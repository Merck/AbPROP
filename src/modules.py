"""
Some code below is modified from:
https://github.com/aws-samples/lm-gvp
Wang, Z., Combs, S.A., Brand, R. et al. LM-GVP: an extensible sequence and structure informed deep learning framework for protein property prediction.
Sci Rep 12, 6832 (2022). https://doi.org/10.1038/s41598-022-10775-y

"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from gvp.models import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv
import ablang
from ablang.pretrained import Pretrained
import numpy as np
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK
from sequence_models.gnn import MPNNLayer, MPNNLayer
from sequence_models.pdb_utils import ab_2_mif_vocab

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def lengths_2_batch_index(lengths):
    batch_index = []
    for i, length in enumerate(lengths):
        batch_index += [i]*length
    return torch.tensor(batch_index, device = 'cuda')
      
        
def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class AbLangGVP(torch.nn.Module):
    #AbLang + GVP Head
    def __init__(self, 
        node_in_dim,
        node_h_dim,
        edge_in_dim,
        edge_h_dim,
        train_params,
        max_length,
        residual=True):
        super(AbLangGVP, self).__init__()
        self.train_params = train_params
        self.residual = residual
        self.max_length = max_length
        self.n = sum(max_length)
        #needs to be 4 to load pretrained weights
        self.num_layers = 4
        self.drop_rate = train_params['drop_rate']
        self.n_hidden = train_params['n_hidden']
        self.eps = train_params['layer_norm_epsilon']
        self.final_layer = torch.nn.Softmax(dim=1) if train_params['outputs'] > 1 else torch.nn.Identity()
        self.tr_matrix = ab_2_mif_vocab().to('cuda')

        if train_params['chains'] == 'both':
            l_params = train_params.copy()
            h_params = train_params.copy()
            l_params['chains'] = 'light'
            h_params['chains'] = 'heavy'
            self.AbLangHeavy = Pretrained(h_params).AbLang
            self.AbLangLight = Pretrained(l_params).AbLang
        else:
            self.AbLang = Pretrained(self.train_params).AbLang
            
        
        vocab_size = 30
        self.W_s = nn.Embedding(vocab_size, vocab_size)
        
        #Getting graph node in dim ready for concatenation with language representation
        node_in_dim = (node_in_dim[0] + vocab_size, node_in_dim[1])
           
        #Deifining GVP Layers
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate= self.drop_rate)
            for _ in range(self.num_layers)
        )
        
        #Because the pretrained weights dont have residual connections
        self.residual = False
        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * self.num_layers,
                node_h_dim[1] * self.num_layers,
            )
            
        self.ns, _ = node_h_dim
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (self.ns, 0))
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.train_params['drop_rate'])

        self.dense = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.ns, eps = self.eps),
            self.dropout,
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), train_params['outputs']),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), train_params['outputs'])
        )
       
                       
    def forward(self, batch, input_ids=None):
        """Perform the forward pass.

        Args:
            batch: (torch_geometric.data.Data, targets)
            input_ids: IDs of the embeddings to be used in the model.

        Returns:
            logits
        """
        
        logits = self._forward(batch, input_ids=input_ids)
        if logits.ndim > 1 and self.train_params['outputs'] == 1: logits = logits.squeeze()
        if logits.ndim == 0: logits = logits.reshape(1)
        if logits.ndim == 1 and self.train_params['outputs'] > 1: logits = logits.reshape(1, self.train_params['outputs'])
        return self.final_layer(logits)

    def _forward(self, batch, input_ids=None):
        """
        Helper function to perform the forward pass.

        Args:
            batch: torch_geometric.data.Data
            input_ids: IDs of the embeddings to be used in the model.
        Returns:
            logits
        """
        h_V = (batch.node_s.to('cuda'), batch.node_v.to('cuda'))
        h_E = (batch.edge_s.to('cuda'), batch.edge_v.to('cuda'))
        
        edge_index = batch.edge_index.to('cuda')
        batch_size = batch.num_graphs
        
        batch_size = batch.num_graphs
        if self.train_params['chains'] == 'both':
            input_ids_heavy, input_ids_light = batch.input_ids[0].reshape(batch_size, -1), batch.input_ids[1].reshape(batch_size, -1)
            attention_mask_heavy, attention_mask_light = batch.attention_mask[0].reshape(batch_size, -1), batch.attention_mask[1].reshape(batch_size, -1)
            heavy_logits = self.AbLangHeavy(input_ids_heavy, attention_mask_light)[:,1:-1,:]
            light_logits = self.AbLangLight(input_ids_light, attention_mask_light)[:,1:-1,:]
            logits = torch.cat([heavy_logits, light_logits],1)
            attention_mask = torch.cat([attention_mask_heavy[:,:-1], attention_mask_light[:,1:]],1)
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            logits = logits.reshape(-1,24)[attention_mask_1d == 0]
        else:
            input_ids = batch.input_ids.reshape(batch_size, -1)
            attention_mask = batch.attention_mask.reshape(batch_size, -1)
            logits = self.AbLang(input_ids, attention_mask)[:,1:-1,:]
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            logits = logits.reshape(-1,24)[attention_mask_1d == 0]  
        
        mif_logits = torch.mm(logits, self.tr_matrix)
        h_V = (torch.cat([h_V[0], mif_logits], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)
            
        out = self.dropout(self.relu(out))
        
        if self.train_params['universal_pooling']:
            padded_out = torch.zeros(batch_size*self.n, self.ns).to('cuda')
            padded_out[attention_mask_1d == 0,:] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            out = scatter_mean(out, batch.batch.to('cuda'), dim=0)
            return self.dense(out).squeeze(-1) + 0.5
    
class AbLangGAT(torch.nn.Module):
    """AbLang + GAT head."""

    def __init__(
        self,
        train_params,
        max_length
    ):

        super(AbLangGAT, self).__init__()
        self.train_params = train_params
        self.max_length = max_length
        if train_params['chains'] == 'both':
            l_params = train_params.copy()
            h_params = train_params.copy()
            l_params['chains'] = 'light'
            h_params['chains'] = 'heavy'
            self.AbLangHeavy = Pretrained(h_params, output_rep = True).AbLang
            self.AbLangLight = Pretrained(l_params, output_rep = True).AbLang
        else:
            self.AbLang = Pretrained(self.train_params, output_rep = True).AbLang        
        self.conv1 = GATConv(768, 128, 4)
        self.conv2 = GATConv(512, 128, 4)
        self.conv3 = GATConv(512, 256, 4)
        self.conv4 = GATConv(1024, 256, 4)
        self.n_hidden = train_params['n_hidden']
        self.n = sum(self.max_length)
        self.relu = nn.ReLU(inplace=True)
        self.eps = train_params['layer_norm_epsilon']
        self.dropout = nn.Dropout(p=self.train_params['drop_rate'])
        self.final_layer = torch.nn.Softmax(dim=1) if train_params['outputs'] > 1 else torch.nn.Identity()
        self.conv_dict = {1:512,
                          2:1024,
                          3:2048,
                          4:3072}
        
        self.conv_out_dim = self.conv_dict[self.train_params['conv_layers']]
        self.dense = nn.Sequential(
            nn.Linear(self.conv_out_dim, int(self.conv_out_dim*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.conv_out_dim*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.conv_out_dim*self.n_hidden), self.train_params['outputs']),
        )
        self.phi = nn.Sequential(
            nn.Linear(self.conv_out_dim, int(self.conv_out_dim*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.conv_out_dim*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.conv_out_dim*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), self.train_params['outputs'])
        )
   
    def forward(self, batch):
        """Does the forward pass through the model for batch[0]
        Args:
            batch: (torch_geometric.data.Data, targets)
        Returns:
            Inferenced logits
        """
        logits = self._forward(batch)
        #dealing with stupid torch dimensionality issues
        if logits.ndim > 1 and self.train_params['outputs'] == 1: logits = logits.squeeze()
        if logits.ndim == 0: logits = logits.reshape(1)
        if logits.ndim == 1 and self.train_params['outputs'] > 1: logits = logits.reshape(1, self.train_params['outputs'])
        return self.final_layer(logits)

    def _forward(self, batch):
        """Does the forward pass through the model for batch
        Args:
            batch: torch_geometric.data.Data
        Returns:
            Inferenced logits
        """
        edge_index = batch.edge_index.to('cuda')
        batch_size = batch.num_graphs
        if self.train_params['chains'] == 'both':
            input_ids_heavy, input_ids_light = batch.input_ids[0].reshape(batch_size, -1), batch.input_ids[1].reshape(batch_size, -1)
            attention_mask_heavy, attention_mask_light = batch.attention_mask[0].reshape(batch_size, -1), batch.attention_mask[1].reshape(batch_size, -1)
            heavy_rep = self.AbLangHeavy(input_ids_heavy, attention_mask_light).last_hidden_states[:,1:-1,:]
            light_rep = self.AbLangLight(input_ids_light, attention_mask_light).last_hidden_states[:,1:-1,:]
            representations = torch.cat([heavy_rep, light_rep],1)
            attention_mask = torch.cat([attention_mask_heavy[:,:-1], attention_mask_light[:,1:]],1)
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            representations = representations.reshape(-1,768)[attention_mask_1d == 0]
        else:
            input_ids = batch.input_ids.reshape(batch_size, -1)
            attention_mask = batch.attention_mask.reshape(batch_size, -1)
            representations = self.AbLang(input_ids, attention_mask).last_hidden_states[:,1:-1,:]
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            representations = representations.reshape(-1,768)[attention_mask_1d == 0]
            
        # GAT forward
        conv1_out=self.conv1(representations, edge_index)
        if self.train_params['conv_layers'] > 1:
            conv2_out = self.conv2(conv1_out, edge_index)
            if self.train_params['conv_layers'] > 2:
                conv3_out = self.conv3(conv2_out, edge_index)
                if self.train_params['conv_layers'] > 3:
                    conv4_out = self.conv4(conv3_out, edge_index)
                    out = torch.cat((conv1_out, conv2_out, conv3_out, conv4_out), dim=-1)
                else:
                    out = torch.cat((conv1_out, conv2_out, conv3_out), dim=-1)
            else:
                out = torch.cat((conv1_out, conv2_out), dim=-1)
        else:    
            out = conv1_out
        out = self.dropout(self.relu(out))  # [n_nodes, 2048]
        if self.train_params['universal_pooling']:
            padded_out = torch.zeros(batch_size*self.n, self.conv_out_dim).to('cuda')
            padded_out[attention_mask_1d == 0,:] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.conv_out_dim)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            out = scatter_mean(out, batch.batch.to('cuda'), dim=0)
            return self.dense(out).squeeze(-1) + 0.5 
        return 

class SGNN(torch.nn.Module):
    """
    Decoder layers from "Generative models for graph-based protein design"
    """

    def __init__(self, train_params, max_length):

        """
        Parameters:
        -----------
        num_letters : int
            len of protein alphabet
        node_features : int
            number of node features
        edge_features : int
            number of edge features
        hidden_dim : int
            hidden dim
        num_encoder_layers : int
            number of encoder layers
        num_decoder_layers : int
            number of decoder layers
        dropout : float
            dropout
        foward_attention_decoder : bool
            if True, use foward attention on encoder embeddings in decoder
        use_mpnn : bool
            if True, use MPNNLayer instead of TransformerLayer
        """

        super(SGNN, self).__init__()

        # Hyperparameters
        self.train_params = train_params
        self.n_hidden = train_params['n_hidden']
        self.eps = train_params['layer_norm_epsilon']
        self.num_letters = 30
        self.decoder_layers = 4
        self.dropout = 0.05
        self.node_features = 10
        self.edge_features = 11
        self.hidden_dim = 256
        self.no_structure = False
        self.n = max_length
        self.dropout_layer = nn.Dropout(p = train_params['drop_rate'])
        self.final_layer = torch.nn.Softmax(dim=1) if train_params['outputs'] > 1 else torch.nn.Identity()
        self.dense = nn.Sequential(
            nn.Linear(self.hidden_dim, int(self.hidden_dim*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.hidden_dim*self.n_hidden), eps = self.eps),
            self.dropout_layer,
            nn.Linear(int(self.hidden_dim*self.n_hidden), train_params['outputs']),
        )
        self.phi = nn.Sequential(
            nn.Linear(self.hidden_dim, int(self.hidden_dim*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.hidden_dim*self.n_hidden), eps = self.eps),
            self.dropout_layer,
            nn.Linear(int(self.hidden_dim*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout_layer,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout_layer,
            nn.Linear(int(self.n*self.n_hidden), self.train_params['outputs'])
        )

        # Embedding layers
        self.W_v = nn.Linear(self.node_features + self.num_letters, self.hidden_dim, bias=True)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)
        self.W_s = nn.Identity()
        layer = MPNNLayer

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            layer(self.hidden_dim, self.hidden_dim * 2, dropout=self.dropout)
            for _ in range(self.decoder_layers)
        ])

    def forward(self, nodes, edges, connections, src, node_mask, lengths, decoder=True):
        """
        Parameters:
        -----------
        nodes : torch.Tensor, (N_batch, L, in_channels)
            Node features
        edges : torch.Tensor, (N_batch, L, K_neighbors, in_channels)
            Edge features
        connections : torch.Tensor, (N_batch, L, K_neighbors)
            Node neighbors
        src : torch.Tensor, (N_batch, L)
            One-hot-encoded sequences
        edge_mask : torch.Tensor, (N_batch, L, k_neighbors)
            Mask to hide nodes with missing features
        Returns:
        --------
        log_probs : torch.Tensor, (N_batch, L, num_letters)
            Log probs of residue predictions
        """
        
        
        
        # Prepare node, edge, sequence embeddings
        h_S = self.W_s(src)  # (N, L, num_letters)
        batch_size, max_length = nodes.shape[0], nodes.shape[1]
        h_V = torch.zeros([batch_size*max_length,30], device = 'cuda')
        h_V[node_mask == 1,:] = h_S
        h_V = h_V.reshape(batch_size,max_length,30)
        h_V = torch.cat([nodes, h_V], dim = -1)
        h_V = self.W_v(h_V)  # (N, L, h_dim - num_letters)
        h_E = self.W_e(edges)  # (N, L, k, h_dim)


        # Run decoder
        for i, layer in enumerate(self.decoder_layers):
            h_EV = cat_neighbors_nodes(h_V, h_E, connections)  # N, L, k, 2 * h_dim
            h_V = layer(h_V, h_EV, mask_V=None)
        
        if self.train_params['universal_pooling']:
            h_V = self.phi(h_V) 
            return self.final_layer(self.rho(h_V.squeeze()) + 0.5)
        else:
            batch_indices = lengths_2_batch_index(lengths)
            h_V = h_V.reshape(batch_size*max_length, self.hidden_dim)[node_mask == 1,:]
            h_V = scatter_mean(h_V, batch_indices, dim=0)
            return self.final_layer(self.dense(h_V).squeeze(-1) + 0.5)
    
class AbLangSGNN(torch.nn.Module):
    'Bert + Pretrained Structured GNN Head'
    def __init__(
        self,
        train_params,
        max_length):

        super(AbLangSGNN, self).__init__()
        self.train_params = train_params
        self.max_length = max_length
        if train_params['chains'] == 'both':
            l_params = train_params.copy()
            h_params = train_params.copy()
            l_params['chains'] = 'light'
            h_params['chains'] = 'heavy'
            self.AbLangHeavy = Pretrained(h_params).AbLang
            self.AbLangLight = Pretrained(l_params).AbLang
        else:
            self.AbLang = Pretrained(self.train_params).AbLang     
        self.n_hidden = train_params['n_hidden']
        self.relu = nn.ReLU(inplace=True)
        self.eps = train_params['layer_norm_epsilon']
        self.dropout = nn.Dropout(p=self.train_params['drop_rate'])
        self.final_layer = torch.nn.Softmax(dim=1) if train_params['outputs'] > 1 else torch.nn.Identity()        
        self.gnn = SGNN(train_params, sum(max_length))
        if train_params['pretrained']:
            print("Loading Pretrained SGNN Weights")
            self.gnn.load_state_dict(torch.load('pretrained_weights/mifst.pt')['model_state_dict'], strict = False)
        self.tr_matrix = ab_2_mif_vocab().to('cuda')
        
        
        
    def forward(self, batch):
        """Does the forward pass through the model for batch[0]
        Args:
            batch: (torch_geometric.data.Data, targets)
        Returns:
            Inferenced logits
        """
        mif_logits, edges, nodes, connections,node_mask, lengths, batch_size = self._forward(batch)
        logits = self.gnn(nodes, edges, connections, mif_logits,node_mask, lengths)
        
        #dealing with stupid torch dimensionality issues
        if logits.ndim > 1 and self.train_params['outputs'] == 1: logits = logits.squeeze()
        if logits.ndim == 0: logits = logits.reshape(1)
        if logits.ndim == 1 and self.train_params['outputs'] > 1: logits = logits.reshape(1, self.train_params['outputs'])
        return logits

    def _forward(self, batch):
        """Does the forward pass through the model for batch
        Args:
            batch: torch_geometric.data.Data
        Returns:
            Inferenced logits
        """
        batch_size = batch.num_graphs
        if self.train_params['chains'] == 'both':
            input_ids_heavy, input_ids_light = batch.input_ids[0].reshape(batch_size, -1), batch.input_ids[1].reshape(batch_size, -1)
            attention_mask_heavy, attention_mask_light = batch.attention_mask[0].reshape(batch_size, -1), batch.attention_mask[1].reshape(batch_size, -1)
            heavy_logits = self.AbLangHeavy(input_ids_heavy, attention_mask_light)[:,1:-1,:]
            light_logits = self.AbLangLight(input_ids_light, attention_mask_light)[:,1:-1,:]
            logits = torch.cat([heavy_logits, light_logits],1)
            attention_mask = torch.cat([attention_mask_heavy[:,:-1], attention_mask_light[:,1:]],1)
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            logits = logits.reshape(-1,24)[attention_mask_1d == 0]
        else:
            input_ids = batch.input_ids.reshape(batch_size, -1)
            attention_mask = batch.attention_mask.reshape(batch_size, -1)
            logits = self.AbLang(input_ids, attention_mask)[:,1:-1,:]
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            logits = logits.reshape(-1,24)[attention_mask_1d == 0]
            
        mif_logits = torch.mm(logits, self.tr_matrix)
        edges = batch.edges
        nodes = batch.nodes
        connections = batch.connections
        node_mask = batch.node_mask
        lengths = batch.lengths
        return (mif_logits, edges, nodes, connections, node_mask, lengths, batch_size)

class AbLangLinear(torch.nn.Module):
    """ BERT + Linear Head"""
    def __init__(
        self,
        train_params,
        max_length
    ):
        super(AbLangLinear, self).__init__()
        self.train_params = train_params
        self.max_length = max_length
        self.AbHead = AbHead(self.train_params, max_length)
        self.random_init = train_params['pretrained'] == False
        if train_params['chains'] == 'both':
            l_params = train_params.copy()
            h_params = train_params.copy()
            l_params['chains'] = 'light'
            h_params['chains'] = 'heavy'
            self.AbLangHeavy = Pretrained(train_params = h_params, random_init = self.random_init, output_rep = True).AbLang
            self.AbLangLight = Pretrained(train_params = l_params, random_init = self.random_init, output_rep = True).AbLang
        else:
            self.AbLang = Pretrained(train_params = self.train_params, random_init = self.random_init, output_rep = True).AbLang
            
    def get_attn_mask(self, tokens, pad_id):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.zeros(*tokens.shape, device=device).masked_fill(tokens == pad_id, 1)
    
    def forward(self, batch):
        """Does the forward pass through the model for batch
        Args:
            batch: torch_geometric.data.Data
        Returns:
            Inferenced logits
        """
        batch_size = batch.shape[0]
        pad_id = 21
        if self.train_params['chains'] == 'both':
            h_length, l_length = self.max_length
            input_ids_heavy = batch[:,0: h_length + 2].reshape(batch_size, -1)
            input_ids_light = batch[:, h_length + 2 :].reshape(batch_size, -1)
            attention_mask_heavy = self.get_attn_mask(batch[:,0: h_length + 2],pad_id).reshape(batch_size, -1)
            attention_mask_light = self.get_attn_mask(batch[:, h_length + 2 :],pad_id).reshape(batch_size, -1)
            heavy_rep = self.AbLangHeavy(input_ids_heavy, attention_mask_light).last_hidden_states[:,1:-1,:]
            light_rep = self.AbLangLight(input_ids_light, attention_mask_light).last_hidden_states[:,1:-1,:]
            representations = torch.cat([heavy_rep, light_rep],1)   
        else:
            attention_mask = self.get_attn_mask(batch,21).reshape(batch_size, -1)
            representations = self.AbLang(batch, attention_mask).last_hidden_states[:,1:-1,:]
        return self.AbHead(representations, batch_size)
    
class AbHead(torch.nn.Module):
    """
    Linear Head for Property Prediction.
    
    """

    def __init__(self, train_params, max_length):
        super().__init__()
        self.train_params = train_params
        self.eps = train_params['layer_norm_epsilon']
        af_choice = train_params['activation']
        self.n = sum(max_length)
        self.full_seq_max_pool = torch.nn.Conv1d(self.n,1,1)
        self.dense = torch.nn.Linear(768, 768)
        self.layer_norm = torch.nn.LayerNorm(768, eps= self.eps)
        self.classifier = torch.nn.Linear(768, train_params['outputs'])
        activations = [torch.nn.ReLU(), torch.nn.LeakyReLU(), torch.nn.Hardswish(), torch.nn.Tanh()]
        helper_dict = {'swish':2, 'relu':0, 'leaky_relu':1, 'tanh':3}
        self.activation = activations[helper_dict[af_choice]]
        self.dropout = torch.nn.Dropout(p=train_params['drop_rate'], inplace=False)
        self.final_layer = torch.nn.Softmax(dim=1) if train_params['outputs'] > 1 else torch.nn.Identity()
        self.universal_norm = torch.nn.LayerNorm(self.n, eps = self.eps)
        self.phi = nn.Sequential(
            nn.Linear(768, int(768*train_params['n_hidden'])),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(768*train_params['n_hidden']), eps = self.eps),
            self.dropout,
            nn.Linear(int(768*train_params['n_hidden']), 1),
        )
        self.rho = nn.Sequential(
            nn.Linear(self.n, int(self.n*train_params['n_hidden'])),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*train_params['n_hidden']), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*train_params['n_hidden']), train_params['outputs'])
        )


    def forward(self, features, bs):
        if self.train_params['universal_pooling']:
            x = self.phi(features)
            x = self.activation(x)
            x = self.universal_norm(x.squeeze())
            x = self.rho(x)
        else:
            x = self.full_seq_max_pool(features)
            x = self.dense(x)
            x = self.activation(x)
            x = self.layer_norm(x)
            x = self.dropout(x)
            x = self.classifier(x)
        if x.ndim > 1 and self.train_params['outputs'] == 1: x = x.squeeze()
        if x.ndim == 0 and self.train_params['outputs'] == 1: x = x.reshape(1)
        if x.ndim != 2 and self.train_params['outputs'] > 1: x = x.reshape(bs,self.train_params['outputs'])
        return self.final_layer(x)
