"""
Some code below is modified from:
https://github.com/aws-samples/lm-gvp
Wang, Z., Combs, S.A., Brand, R. et al. LM-GVP: an extensible sequence and structure informed deep learning framework for protein property prediction.
Sci Rep 12, 6832 (2022). https://doi.org/10.1038/s41598-022-10775-y

"""
import re


def prep_seq(seq):
    """
    Adding spaces between AAs and replace rare AA [UZOB] to X.
    ref: https://huggingface.co/Rostlab/prot_bert.

    Args
        seq: a string of AA sequence.

    Returns:
        String representing the input sequence where U,Z,O and B has been replaced by X.
    """
    seq_spaced = " ".join(seq)
    seq_input = re.sub(r"[UZOB]", "X", seq_spaced)
    return seq_input
