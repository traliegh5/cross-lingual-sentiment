from comet_ml import Experiment
import torch
import argparse
from torch import nn, optim
from transformers import XLMRobertaTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
#from get_data import load_dataset
from preprocess import SentimentData
from model import Sentiment_Analysis_Model
from tqdm import tqdm
from sklearn.metrics import f1_score
#pip install sentencepiece
#pip install transformers

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

experiment = Experiment(project_name="cross-lingual-sentiment-analysis")

hyperparams = {
"batch_size": 16,
"window_size": 50, # max len is ~ 126
"learning_rate":0.001,
"num_epochs":1
}

def train(model, train_loader, optimizer,experiment,hyperparams, pad_id):
    # Loss 
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = 0
    total_f1 = 0
    total_word_count = 0

    with experiment.train():
        for epoch in range(hyperparams["num_epochs"]):
            batch_num = 0
            for (inputs, labels, lengths) in train_loader:
                num_in_batch = len(lengths)
                if (num_in_batch < hyperparams['batch_size']):
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                batch_num += 1

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                (logits, probs) = model(inputs, lengths)

                loss = loss_fn(logits, labels)
                loss.backward() 
                optimizer.step()

                word_count = sum(lengths)
                total_loss += (loss * word_count)
                total_word_count += word_count
                indices = torch.max(probs, 1)[1]
                total_f1 += f1_score(labels, indices, average='binary')
                print("At batch " + str(batch_num) + " loss is: " + str(loss))

        # Log perplexity to Comet.ml using experiment.log_metric
        mean_loss = total_loss / total_word_count
        perplexity = torch.exp(mean_loss).detach()
        overall_f1 = total_f1/num_in_batch

        print("perplexity:", perplexity)
        print("F1:", overall_f1)
        experiment.log_metric("perplexity", perplexity.cpu())
        experiment.log_metric("F1", overall_f1)

# Test the model on the test set - report perplexity
def test(model, test_loader, experiment, hyperparams, pad_id):
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0
    total_f1 = 0
    total_word_count = 0

    with experiment.test():
        batch_num = 0
        with torch.no_grad():
            for (inputs, labels, lengths) in test_loader:
                num_in_batch = len(lengths)
                if (num_in_batch < hyperparams['batch_size']):
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                batch_num += 1

                (logits, probs) = model(inputs, lengths)
                loss = loss_fn(logits, labels)

                word_count = sum(lengths)
                total_loss += (loss * word_count)
                total_word_count += word_count
                indices = torch.max(probs, 1)[1]
                total_f1 += f1_score(labels, indices, average='binary')
                print("At batch " + str(batch_num) + " loss is: " + str(loss))

        mean_loss = total_loss / total_word_count
        perplexity = torch.exp(mean_loss).detach()
        overall_f1 = total_f1/num_in_batch

        print("perplexity:", perplexity)
        print("F1:", overall_f1)
        experiment.log_metric("perplexity", perplexity.cpu())
        experiment.log_metric("F1", overall_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("train_file")
    # parser.add_argument("test_file")
    parser.add_argument("-m", "--model", type=str, default="",
                        help="xlmr or bert")
    parser.add_argument("-lang", "--language", type=str, default="",
                        help="it, jp or de")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    experiment.log_parameters(hyperparams)
    model_type = args.model
    train_lang = args.language

    # Load the GPT2 Tokenizer, add any special token if needed
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}

    if (model_type == "xlmr"):
        assert train_lang == 'it' or 'jp'
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        vocab_size = len(tokenizer)
        model = Sentiment_Analysis_Model(hyperparams['window_size'], vocab_size, model_type, device_type).to(device)

        if (train_lang == 'it'):
            train_file = "Data/it/train.tsv"
        else: 
            train_file = "Data/jp/train.tsv"

        test_file = "Data/de/test.tsv"
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        vocab_size = len(tokenizer)
        model = Sentiment_Analysis_Model(hyperparams['window_size'], vocab_size, model_type, device_type).to(device)
        train_file = "Data/de/train.tsv"
        test_file = "Data/de/test.tsv"

    # xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    # #xlmr_tokenizer.add_special_tokens(special_tokens_dict)
    # vocab_size = len(xlmr_tokenizer)
    # xlmr_model = Sentiment_Analysis_Model(hyperparams['window_size'], vocab_size, device_type).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    # # Load the train, test DataLoader NOTE: Parse the data using GPT2 tokenizer
    train_dataset = SentimentData(train_file, hyperparams['window_size'], tokenizer)
    test_dataset = SentimentData(test_file, hyperparams['window_size'], tokenizer)
    pad_token = train_dataset.pad_token
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])
                
    if args.load:
        print("loading model...")
        model.load_state_dict(torch.load('./model.pt', map_location=device))
    if args.train:
        # run train loop here
        print("running fine-tuning loop...")
        train(model, train_loader, optimizer, experiment, hyperparams, pad_token)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test(model, test_loader, experiment,hyperparams, pad_token)
