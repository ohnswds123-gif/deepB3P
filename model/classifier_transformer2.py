# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : classifier.py


import torch.nn.functional as F
from model.transLayers import *

class Classifier(nn.Module):
    def __init__(self, seq_len, vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device,feat_dim=0 ): #
        super(Classifier, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device,feat_dim=feat_dim).to(device)
        self.decoder = Decoder(vocab_size, seq_len, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device,feat_dim=feat_dim).to(device)
        self.attn_pool = nn.Linear(d_model, 1)
        self.projection = nn.Sequential(
            nn.Linear(seq_len * d_model, 256),  # seq_len * d_model 是展平后的输入维度
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x,extra_feats=None):#
        # x = x.long()  # Convert x to LongTensor
        if len(extra_feats.shape) == 2:
            extra_feats = extra_feats.expand(-1, x.size(1), -1)
            x = torch.cat((x, extra_feats), dim=-1)
        enc_outputs, self.enc_att = self.encoder(x) #,extra_feats,enc_outputs: [batch_size, seq_len, d_model], enc_att: [batch_size, n_layers, n_heads, src_len, src_len]
        dec_outputs, self.dec_att, self.dec_enc_att = self.decoder(x, enc_outputs) #dec_outputs: [batch_size, seq_len, d_model], enc_att: [batch_size, n_layers, n_heads, src_len, src_len]
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)
        logists = self.projection(dec_outputs)
        return F.softmax(logists,dim=-1)


    def get_atts(self):
        return self.enc_att, self.dec_att, self.dec_enc_att

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
