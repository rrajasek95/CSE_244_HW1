import torch
import torch.nn as nn
import torch.nn.functional as F

class OneClassBiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, out_features, embeddings):
         super(BiRNN, self).__init__()

         self.embedding = nn.Embedding(vocab_size, embed_size)
         self.embedding.weight.data.copy_(embeddings)
         for p in self.embedding.parameters():
            p.requires_grad = False
         self.encoder = nn.LSTM(embed_size, num_hiddens, 
            bidirectional=True, num_layers=num_layers)
         self.decoder = nn.Linear(4 * num_hiddens, out_features)


    def forward(self, x_in, apply_softmax=False):
        embed = self.embedding(x_in).transpose(0, 1)

        out = self.encoder(embed)

        encoding = torch.cat([out[0][0], out[0][-1]], dim=1)
        output = self.decoder(encoding)


        if apply_softmax:
            output = F.sigmoid(output)

        return output