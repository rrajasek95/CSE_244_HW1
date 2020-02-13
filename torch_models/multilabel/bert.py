import torch.nn as nn
from transformers import PreTrainedBertModel

class BertForMultiLabelClassification(PreTrainedBertModel):
    def __init__(self, out_dim):
        super(BertForMultiLabelClassification, self).__init__()

    def forward(self):
        pass