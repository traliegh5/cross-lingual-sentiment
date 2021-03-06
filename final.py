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
# max len is ~ 126 (nlproc xlmr), 71 (xed xlmr), 67 (xed bert), 64 (fi xlmr)
"learning_rate":0.001,
"num_epochs":3
}

def train(model, train_loader, optimizer,experiment, num_classes, pos_weight, hyperparams):        
    torch.cuda.empty_cache()
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses = []
    total_f1 = 0
    correct_predictions = 0
    num_examples = 0
    total_batches = 0

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
                total_batches += 1

                optimizer.zero_grad()
                (logits, probs) = model(inputs, lengths)

                round_probs = np.round(probs.cpu().data.numpy())
                if (batch_num % 100 == 0):
                    print(round_probs)
                    print(labels)

                labels_for_loss = labels.type_as(logits)

                loss = loss_fn(logits, labels_for_loss)
                losses.append(loss.item())
                loss.backward()

                # from post
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                f1 = f1_score(labels.cpu().data.numpy(), round_probs, average='macro',zero_division=1)

                num_correct = np.sum((round_probs == labels.cpu().data.numpy()).all(1))
                accuracy_out_of = num_in_batch
                print("Num correct: " + str(num_correct))
                print("F1: " + str(f1))
                # print(round_probs)
                # print(labels.cpu().data.numpy())

                total_f1 += f1
                correct_predictions += num_correct

                print("Batch: " + str(batch_num) + " | loss: " + str(loss.item()) + " | accuracy: " 
                    + str(num_correct/accuracy_out_of))

                del inputs
                del labels
                del lengths
                gc.collect()
                torch.cuda.empty_cache()

        mean_loss = np.mean(losses)
        print("Mean loss: " + str(mean_loss))
        accuracy = correct_predictions / num_examples
        perplexity = np.exp(mean_loss)
        overall_f1 = total_f1/total_batches

        print("perplexity:", perplexity)
        print("F1:", overall_f1)
        print("Accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)
        experiment.log_metric("F1", overall_f1)

# Test the model on the test set - report perplexity
def test(model, test_loader, experiment, num_classes, pos_weight, hyperparams):
    losses = []
    total_f1 = 0
    correct_predictions = 0
    num_examples = 0

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

                optimizer.zero_grad()
                (logits, probs) = model(inputs, lengths)

                round_probs = np.round(probs.cpu().data.numpy())

                labels_for_loss = labels.type_as(logits)

                loss = loss_fn(logits, labels_for_loss)
                losses.append(loss.item())

                f1 = f1_score(labels.cpu().data.numpy(), round_probs, average='macro',zero_division=1)

                num_correct = np.sum((round_probs == labels.cpu().data.numpy()).all(1))
                accuracy_out_of = num_in_batch
                print("Num correct: " + str(num_correct))
                print("F1: " + str(f1))
                # print(round_probs)
                # print(labels.cpu().data.numpy())

                total_f1 += f1
                correct_predictions += num_correct

                print("Batch: " + str(batch_num) + " | loss: " + str(loss.item()) + " | accuracy: " 
                    + str(num_correct/accuracy_out_of))

        mean_loss = np.mean(losses)
        print("Mean loss: " + str(mean_loss))
        accuracy = correct_predictions / num_examples
        perplexity = np.exp(mean_loss)
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
                        help="1 or 8")
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

    if (model_type == "xlmr"):
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        
        if (train_lang == 'related'):
            train_file = "XED/en-annotated.tsv"
        else: 
            train_file = "XED/fi-annotated.tsv"

        test_file = "XED/de-projections.tsv"
            
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

        train_file = "XED/de-projections.tsv"
        test_file = "XED/de-projections.tsv"    

    vocab_size = len(tokenizer)
    model = Sentiment_Analysis_Model(vocab_size, model_type, num_classes, device_type).to(device)

    optimizer = torch.optim.Adam(model.parameters(), 2e-5)
    #optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False) # from post
    
    train_dataset = SentimentData(train_file, tokenizer, dataset_name, num_classes)
    test_dataset = SentimentData(test_file, tokenizer, dataset_name, num_classes)
    train_pos_weights = train_dataset.pos_weights.to(device)
    test_pos_weights = test_dataset.pos_weights.to(device)
    pad_token = train_dataset.pad_token

    # Seperating all train datasets - we want to split up even for the xlmr model to be able to compare
    # english accuracy and F1 to the values provided in the xed paper
    length=train_dataset.__len__()
    big=int(.9*length)
    splits=[big,length-big]
    train_dataset,eval_dataset=random_split(train_dataset,splits)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=hyperparams['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])
                
    if args.load:
        print("loading model...")
        model.load_state_dict(torch.load('./model.pt', map_location=device))
    if args.train:
        # run train loop here
        print("running fine-tuning loop...")
        train(model, train_loader, optimizer, experiment, num_classes, train_pos_weights, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.test:
        # run test loop here
        print("running testing loop - same language...")
        test(model, eval_loader, experiment, num_classes, test_pos_weights, hyperparams)
        if (model_type == 'xlmr'):
            print("running testing loop - different language...")
            test(model, test_loader, experiment, num_classes, test_pos_weights, hyperparams)
