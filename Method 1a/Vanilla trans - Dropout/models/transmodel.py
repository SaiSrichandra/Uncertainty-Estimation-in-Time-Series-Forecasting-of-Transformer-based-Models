import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np


class TransModel(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, embed_type, enc_in, dec_in, c_out, seq_len, label_len, pred_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(TransModel, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Embedding
        if embed_type == 0:     
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        elif embed_type == 1:
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        elif embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        elif embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding_wo_temp(dec_in, d_model, embed, freq, dropout)
        elif embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, infer_noise = False, conv_l_n = 0, params = {'dropout_rate' : 0.3}):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, infer_noise=infer_noise, conv_l_n=conv_l_n, params=params)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]