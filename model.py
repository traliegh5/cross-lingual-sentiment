import torch
import torch.nn as nn
from transformers import XLMRobertaModel, AutoModel

class Sentiment_Analysis_Model(nn.Module):
    def __init__(self, vocab_size, model_type, num_classes, device_type):
        super(Sentiment_Analysis_Model, self).__init__()
        '''
        
        '''
        self.device_type = device_type
        if (model_type == 'xlmr'):
            self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        else: 
            self.model = AutoModel.from_pretrained("bert-base-german-cased")
        self.num_classes = num_classes
        self.model.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(p=0.3)
        self.dense = nn.Linear(self.model.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, inputs, lengths):
        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, window_size)

        :return: the logits
        """
        seq_len = list(inputs.size())[1]
        attention_mask = get_attention_mask(seq_len, lengths, self.device_type)
        (_, pooler_output) = self.model(inputs,attention_mask=attention_mask).to_tuple()
        dropout_output = self.dropout(pooler_output)
        dense_output = self.dense(dropout_output)
        to_return = self.sig(dense_output)

        return (dense_output, to_return)

def get_attention_mask(seq_len, lengths, device_type):
    num_in_batch = len(lengths)
    mask = torch.empty((num_in_batch, seq_len))
    for i in range(num_in_batch):
        length = lengths[i]
        ones = torch.ones((1,length))
        zeros = torch.zeros((1,seq_len - length))
        mask_line = torch.cat((ones, zeros),1)
        mask[i] = mask_line

    mask = mask.to(torch.device(device_type))
    return mask
