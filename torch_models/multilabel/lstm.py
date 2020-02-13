import torch
import torch.nn as nn
import torch.nn.functional as F

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden, num_layers, output_dim, embeddings=None):
         super(BiRNN, self).__init__()
         self.embedding = nn.Embedding(vocab_size, embed_dim)
         if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)

         self.encoder = nn.LSTM(embed_dim, num_hidden, 
            bidirectional=True, num_layers=num_layers)
         self.decoder = nn.Linear(4 * num_hidden, output_dim)


    def forward(self, x_in, apply_sigmoid=False):
        embed = self.embedding(x_in).transpose(0, 1)

        out = self.encoder(embed)

        encoding = torch.cat([out[0][0], out[0][-1]], dim=1)
        output = self.decoder(encoding)


        if apply_sigmoid:
            output = F.sigmoid(output)

        return output

class BiRNNStaticEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden, num_layers, output_dim, embeddings=None):
         super(BiRNN, self).__init__()
         self.embedding = nn.Embedding(vocab_size, embed_dim)
         if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
         for p in self.embedding.parameters():
            p.requires_grad = False
         self.encoder = nn.LSTM(embed_dim, num_hidden, 
            bidirectional=True, num_layers=num_layers)
         self.decoder = nn.Linear(4 * num_hidden, output_dim)


    def forward(self, x_in, apply_sigmoid=False):
        embed = self.embedding(x_in).transpose(0, 1)

        out = self.encoder(embed)

        encoding = torch.cat([out[0][0], out[0][-1]], dim=1)
        output = self.decoder(encoding)


        if apply_sigmoid:
            output = F.sigmoid(output)

        return output