import torch
import torch.nn as nn
from transformers import XLMRobertaModel, AutoModel

class Sentiment_Analysis_Model(nn.Module):
    def __init__(self, window_size, vocab_size, model_type, device_type):
        super(Sentiment_Analysis_Model, self).__init__()
        '''
        
        '''
        self.window_size = window_size
        self.device_type = device_type
        #self.xlmr_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        if (model_type == 'xlmr'):
            self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        else: 
            self.model = AutoModel.from_pretrained("bert-base-german-cased")
        self.model.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(p=0.3)
        self.dense = nn.Linear(self.model.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.model.eval()

    def forward(self, inputs, lengths):
        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, window_size)

        :return: the logits
        """
        attention_mask = get_attention_mask(self.window_size, lengths, self.device_type)
        (_, pooler_output) = self.model(inputs,attention_mask=attention_mask).to_tuple()
        dropout_output = self.dropout(pooler_output)
        dense_output = self.dense(dropout_output)

        return (dense_output, self.softmax(dense_output))

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
