"""
TransformerModel copied from
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, noutp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.pos_input_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.input_encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.output_encoder = nn.Embedding(ntoken, noutp)
        self.pos_output_encoder = PositionalEncoding(noutp, dropout)
        decoder_layers = TransformerDecoderLayer(noutp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.linear_decoder = nn.Linear(noutp, ntoken)
        self.softmax_decoder = nn.Softmax(dim=0)
        self.noutp = noutp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_encoder.weight.data.uniform_(-initrange, initrange)
        self.output_encoder.weight.data.uniform_(-initrange, initrange)
        self.linear_decoder.bias.data.zero_()
        self.linear_decoder.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, context, response, src_mask, response_mask):
        context = self.input_encoder(context) * math.sqrt(self.ninp)
        context = self.pos_input_encoder(context)
        encoder_output = self.transformer_encoder(context, src_mask)
        response = self.output_encoder(response) * math.sqrt(self.noutp)
        response = self.pos_output_encoder(response)
        #print('encoder_output:', encoder_output)
        #print('encoder_output.size():', encoder_output.size())
        #print('response:', response)
        #print('response.size():', response.size())
        output = self.transformer_decoder(response, encoder_output, response_mask)
        #print('decoder output:', output)
        #print(output.size())
        output = self.linear_decoder(output)
        #print('linear output:', output)
        #print(output.size())
        output = self.softmax_decoder(output)
        #print('softmax output:', output)
        #print(output.size())
        return output

