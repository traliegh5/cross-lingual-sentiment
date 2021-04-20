from comet_ml import Experiment
import torch
import argparse
from torch import nn, optim
from transformers import XLMRobertaTokenizer, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,random_split
#from get_data import load_dataset
from preprocess import SentimentData
from model import Sentiment_Analysis_Model
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import gc
#pip install sentencepiece
#pip install transformers

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

experiment = Experiment(project_name="cross-lingual-sentiment-analysis")

hyperparams = {
"batch_size": 16,
"window_size": 60, # max len is ~ 126 (nlproc xlmr), 71 (xed xlmr), 67 (xed bert), 64 (fi xlmr)
"learning_rate":0.001,
"num_epochs":3
}

def train(model, train_loader, optimizer,scheduler,experiment, num_classes,hyperparams, pad_id):
    # Loss 
        
    torch.cuda.empty_cache()
    if (num_classes==2):
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0
    losses = []
    total_f1 = 0
    total_word_count = 0
    correct_predictions = 0
    num_examples = 0

    model = model.train()

    with experiment.train():
        for epoch in range(hyperparams["num_epochs"]):
            batch_num = 0
            for (inputs, labels, lengths) in tqdm(train_loader):
                num_in_batch = len(lengths)
                num_examples += num_in_batch

                inputs = inputs.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                batch_num += 1

                optimizer.zero_grad()
                (logits, probs) = model(inputs, lengths)

                #labels = labels.type_as(logits)
                round_probs = np.round(probs.cpu().data.numpy())
                # print(round_probs[:10])
                # print(labels[:10])

                if (num_classes == 2):
                    labels_for_loss = labels
                else:
                    labels_for_loss = labels.type_as(logits)

                loss = loss_fn(logits, labels_for_loss)
                losses.append(loss.item())
                loss.backward()

                # from post
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() 
                #optimizer.step() <- for regular cross entropy

                word_count = sum(lengths)
                total_loss += (loss * word_count)
                total_word_count += word_count
                indices = torch.max(probs, 1)[1].cpu().data.numpy()
                if (num_classes == 2):
                    f1 = f1_score(labels.cpu().data.numpy(), indices, average='binary')
                    num_correct = np.sum(indices == labels.cpu().data.numpy())
                    accuracy_out_of = num_in_batch
                else:
                    f1 = f1_score(labels.cpu().data.numpy(), round_probs, average='macro')
                    num_correct = np.sum(round_probs == labels.cpu().data.numpy())
                    accuracy_out_of = num_in_batch*8
                total_f1 += f1
                correct_predictions += num_correct

                print("Batch: " + str(batch_num) + " | loss: " + str(loss.item()) + " | accuracy: " 
                    + str(num_correct/accuracy_out_of))
                # print("num correct: " + str(num_correct))
                # print("accuracy out of: " + str(accuracy_out_of))
                # print(indices)
                # print(labels.cpu().data.numpy())

                # print("At batch " + str(batch_num) + " loss is: " + str(loss))
                del inputs
                del labels
                del lengths
                gc.collect()
                torch.cuda.empty_cache()

        #mean_loss = total_loss / total_word_count
        mean_loss = np.mean(losses)
        print("Mean loss: " + str(mean_loss))
        accuracy = correct_predictions / num_examples
        perplexity = np.exp(mean_loss)
        #perplexity = torch.exp(mean_loss).detach()
        overall_f1 = total_f1/batch_num

        print("perplexity:", perplexity)
        print("F1:", overall_f1)
        print("Accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)
        experiment.log_metric("F1", overall_f1)

# Test the model on the test set - report perplexity
def test(model, test_loader, experiment, num_classes, hyperparams, pad_id):
    total_loss = 0
    losses = []
    total_f1 = 0
    total_word_count = 0
    correct_predictions = 0
    num_examples = 0

    if (num_classes == 2):
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    model = model.eval()

    with experiment.test():
        batch_num = 0
        with torch.no_grad():
            for (inputs, labels, lengths) in tqdm(test_loader):
                num_in_batch = len(lengths)
                num_examples += num_in_batch

                inputs = inputs.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                batch_num += 1

                (logits, probs) = model(inputs, lengths)

                if (num_classes == 2):
                    labels_for_loss = labels
                else:
                    labels_for_loss = labels.type_as(logits)

                loss = loss_fn(logits, labels_for_loss)
                losses.append(loss.item())
                round_probs = np.round(probs.cpu().data.numpy())
                
                word_count = sum(lengths)
                total_loss += (loss * word_count)
                total_word_count += word_count
                indices = torch.max(probs, 1)[1].cpu().data.numpy()
                if (num_classes == 2):
                    f1 = f1_score(labels.cpu().data.numpy(), indices, average='binary')
                    num_correct = np.sum(indices == labels.cpu().data.numpy())
                    accuracy_out_of = num_in_batch
                else:
                    f1 = f1_score(labels.cpu().data.numpy(), round_probs, average='macro')
                    num_correct = np.sum(round_probs == labels.cpu().data.numpy())
                    accuracy_out_of = num_in_batch*8
                total_f1 += f1
                correct_predictions += num_correct

                print("Batch: " + str(batch_num) + " | loss: " + str(loss.item()) + " | accuracy: " 
                    + str(num_correct/accuracy_out_of))

        #mean_loss = total_loss / total_word_count
        mean_loss = np.mean(losses)
        #perplexity = torch.exp(mean_loss).detach()
        perplexity = np.exp(mean_loss)
        overall_f1 = total_f1/batch_num
        accuracy = correct_predictions / num_examples

        mean_loss = total_loss / total_word_count
        perplexity = torch.exp(mean_loss).detach()
        # overall_f1 = total_f1/num_in_batch
        overall_f1 = total_f1/batch_num

        print("perplexity:", perplexity)
        print("F1:", overall_f1)
        print("Accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)
        experiment.log_metric("F1", overall_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="",
                        help="xlmr or bert")
    parser.add_argument("-lang", "--language", type=str, default="related",
                        help="related or diff")
    parser.add_argument("-n", "--num_classes", type=str, default="8",
                        help="2 or 8")
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
    num_classes = int(args.num_classes)

    dataset_name = "xed"

    binary = True

    if (model_type == "xlmr"):
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        
        if (dataset_name == "nlproc"):
            if (train_lang == 'related'):
                train_file = "Data/it/train.tsv"
            else: 
                train_file = "Data/jp/train.tsv"

            test_file = "Data/de/test.tsv"
        else: 
            if (train_lang == 'related'):
                train_file = "XED/en-annotated.tsv"
            else: 
                train_file = "XED/fi-annotated.tsv"

            test_file = "XED/de-projections.tsv"
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

        if (dataset_name == "nlproc"):
            train_file = "Data/de/train.tsv"
            test_file = "Data/de/test.tsv"
        else: 
            train_file = "XED/de-projections.tsv"
            test_file = "XED/de-projections.tsv"

    vocab_size = len(tokenizer)
    model = Sentiment_Analysis_Model(vocab_size, model_type, num_classes, device_type).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False) # from post
    
    train_dataset = SentimentData(train_file, tokenizer, dataset_name, num_classes)
    test_dataset = SentimentData(test_file, tokenizer, dataset_name, num_classes)
    pad_token = train_dataset.pad_token

    # if (dataset_name!= "nlproc") and (model_type!="xlmr"):
    if (model_type!="xlmr"):
        length=train_dataset.__len__()
        big=int(.9*length)
        splits=[big,length-big]
        train_dataset,test_dataset=random_split(train_dataset,splits)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    # from post
    total_steps = len(train_dataset) * hyperparams['num_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
                
    if args.load:
        print("loading model...")
        model.load_state_dict(torch.load('./model.pt', map_location=device))
    if args.train:
        # run train loop here
        print("running fine-tuning loop...")
        train(model, train_loader, optimizer, scheduler, experiment, num_classes, hyperparams, pad_token)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test(model, test_loader, experiment, num_classes, hyperparams, pad_token)
