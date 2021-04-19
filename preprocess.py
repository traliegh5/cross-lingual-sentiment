from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class SentimentData(Dataset):
    def __init__(self, input_file,window_size, tokenizer, dataset_name):
        """
        Read and parse the translation dataset line by line. Make sure you
        separate them on tab to get the original sentence and the target
        sentence. You will need to adjust implementation details such as the
        vocabulary depending on whether the dataset is BPE or not.
        :param input_file: the data file pathname
        :param enc_seq_len: sequence length of encoder
        :param dec_seq_len: sequence length of decoder
        :param bpe: whether this is a Byte Pair Encoded dataset
        :param target: the tag for target language
        :param word2id: the word2id to append upon
        :param flip: whether to flip the ordering of the sentences in each line
        """
        # TODO: read the input file line by line and put the lines in a list.

        # TODO: split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.

        # TODO: create inputs and labels for both training and validation data
        #       and make sure you pad your inputs.

        # Hint: remember to add start and pad to create inputs and labels
        
        
        self.lengths=[]
        self.labels=[]
        self.tokenizer=tokenizer
        self.pad_token=tokenizer("<pad>")["input_ids"][1]

        # Seems to already be inserting the cls token (0) and sep token (2)
        # during tokenization
        self.cls_token=tokenizer("<s>")["input_ids"][1]
        self.sep_token=tokenizer("</s>")["input_ids"][1]

        self.tokens=[]
        self.texts=[]
        # self.start=tokenizer.bos_token_id

        with open(input_file,'r') as f:
            
            while True:
                line=f.readline()
                
                if not line:
                    break

                if (dataset_name == "nlproc"):
                    self.labels.append(int(line[0]))
                    text = line[1:]
                    self.texts.append(text)
                    toke=self.tokenizer(text)['input_ids'][:(window_size)]
                    #toke.insert(0,self.cls_token)
                    #toke.append(self.sep_token)
                    self.tokens.append(torch.as_tensor(toke))
                else:
                    tabbed = line.split("\t")
                    text = tabbed[0]
                    line_labels = tabbed[1]
                    toke=self.tokenizer(text)['input_ids'][:(window_size)]
                    #toke.insert(0,self.cls_token)
                    #toke.append(self.sep_token)
                    num_emotions = 8
                    prepped_label = [0] * num_emotions

                    for i in range(num_emotions):
                        if str(i+1) in line_labels:
                            prepped_label[i] = 1

                    self.labels.append(prepped_label)
                    self.texts.append(text)
                    
                    self.tokens.append(torch.as_tensor(toke))
                # cent.insert(0,'START')
                # token=self.tokenizer(line)['input_ids']
                # token=token[:self.window_size]
                # token.insert(0,self.start)
                # token.append(self.start)
                
                # self.tense.append(torch.as_tensor(token))
                # temp=torch.ones(self.window_size+2)
                # temp[len(token):]=0.0
                # self.masks.append(temp)
                
                self.lengths.append(len(toke))
       
        self.labels=torch.as_tensor(self.labels)
        self.tokens=pad_sequence(self.tokens,batch_first=True,padding_value=self.pad_token)
        # print("Size of tokens: " + str(self.tokens.size()))
        # print("Size of labels: " + str(self.labels.size()))
        # print("Size of lengths: " + str(len(self.lengths)))
        # print("Some tokens")
        # print(self.tokens[:5])
        # print("Some labels")
        # print(self.labels[:5])
        # print("Some lengths")
        # print(self.lengths[:5])
        # print(max(self.lengths))
        # self.masks=pad_sequence(self.masks,batch_first=True,padding_value=padding_value)

    def __len__(self):
        """
        len should return a the length of the dataset
        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        
        # return self.sentences.shape[0]
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.
        :param idx: the index for retrieval
        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
       
        #item={"input":self.tokens[idx,:],"label":self.labels[idx],"pad_token":self.pad_token}
        item={"input":self.tokens[idx,:],"label":self.labels[idx],"lengths":self.lengths[idx]}
        return (self.tokens[idx], self.labels[idx], self.lengths[idx])

if __name__ == "__main__":
    tokenizer=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    german_train_set=SentimentData("Data/de/train.tsv",10,tokenizer)
    german_test_set=SentimentData("Data/de/test.tsv",10,tokenizer)
    japanese_train_set=SentimentData("Data/it/train.tsv",10,tokenizer)
    item=japanese_train_set.__getitem__(0)
    print(item["input"],item["label"])