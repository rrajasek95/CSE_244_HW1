import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_dim, embeddings = None):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
        
        
        self.conv1 = nn.Conv1d(embed_size, 40, 3, padding=1)
        self.pool = nn.MaxPool1d(3, padding=1)

        self.fc = nn.Linear(40 * 10, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x_in, apply_sigmoid=False):
        # Embed from 30 to batch size * num channels * length
        embed = self.embedding(x_in).transpose(1, 2)
        # Apply convolution to convert from (N, C_in, seq_len) to (N, C_out, L_out)
        conv_out = F.relu(self.conv1(embed))
        # Apply pooling to convert to (N, C_out, seq_len)
        pool_out = self.pool(conv_out)
        pool_view = pool_out.view(-1, 40 * 10)
        # Now map to fully connected layer
        fc_out = self.dropout(F.relu(self.fc(pool_view)))

        output = self.fc2(fc_out)
        
        if apply_sigmoid:
            output = F.sigmoid(output)

        return output

class OneHotCNN(nn.Module):

    def __init__(self, initial_num_channels,output_dim, num_channels):
        super(OneHotCNN, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                out_channels=num_channels, kernel_size=5),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                kernel_size=3, stride=2),
            nn.ELU(),
            )

        self.fc = nn.Linear(num_channels, output_dim)

    def forward(self, x_in, apply_sigmoid=False):
        features = self.convnet(x_in).squeeze(dim=2)
        out = self.fc(features)

        if apply_sigmoid:
            out = F.sigmout(out, dim=1)

        return out