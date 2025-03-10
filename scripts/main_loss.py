import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, get_scheduler, DataCollatorWithPadding, AutoTokenizer)
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from collections import Counter
import time
import argparse
import os
import re
import datetime

from losses import CustomLoss
from utils import predict, seed_everything

parser = argparse.ArgumentParser()

parser.add_argument('-which_loss','--which_loss', type=str, help='Type of loss', default='softmax',
                    choices=['softmax', 'wsoftmax', 'focalloss', 'classbalancedloss', 'balancedsoftmax', 'equalizationloss', 'ldamloss'])
parser.add_argument('-epochs', '--epochs', type=int, help='Number of epochs', default=10)
parser.add_argument("-batch_size", "--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("-learning_rate", "--learning_rate", default=5e-5, type=float, help="Learning rate.")
parser.add_argument('-input', '--input', type=str, help='What to train on (text/title).', default='title', choices=['title', 'text', 'tt']) #tt = title+text
parser.add_argument('-model_name', '--model_name', type=str, help='Name of model', default='bert-base-uncased')
parser.add_argument('-data_dir', '--data_dir', type=str, help='Path to FHD folder', default='/kaggle/input/fhd-data/')
parser.add_argument('-seed','--seed', type=int, help='Experiment seed', default=786879)
parser.add_argument('-maxlen','--maxlen', type=int, help='maxlen', default=256)
parser.add_argument('-patience','--patience', type=int, help='early stopping patience', default=3)

args = parser.parse_args()

DATADIR = args.data_dir
MODEL_NAME = args.model_name
LOSSFN = args.which_loss
EPOCHS = args.epochs
MAXLEN = args.maxlen
SEED = args.seed

seed_everything(SEED)

def tokenize_function(examples):
    return tokenizer(examples[f'{args.input}'], padding='max_length', max_length=MAXLEN, truncation=True)

# ========================== main run =====================================

if __name__ == "__main__":

    # ======================= dataset loading ==================================

    data = pd.read_csv(DATADIR + 'data/incidents_train.csv', index_col=0)
    valid = pd.read_csv(DATADIR + 'data/incidents_dev.csv', index_col=0)

    # adding title+text columns
    data["tt"] = data["title"] + "[SEP]" + data["text"]
    valid["tt"] = valid["title"] + "[SEP]" + valid["text"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create logdir name
    #args.logdir = os.path.join("logs", "{}-{}-{}".format(
    #    os.path.basename(globals().get("__file__", "notebook")),
    #    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    #    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()) if k != 'data_dir'))
    #))

    #create a LOGS/ folder if doesn't exist

    args.logdir = 'LOGS/'+ datetime.datetime.now().strftime("%Y%m%d%H%M%S") + f'_seed.{SEED}_wl.{LOSSFN}_ep.{EPOCHS}_lr.{args.learning_rate}_plm.{MODEL_NAME}_maxlen.{MAXLEN}_bs.{args.batch_size}_input.{args.input}'

    print(args)

    for label in tqdm(['hazard-category', 'product-category', 'hazard', 'product']):
        label_encoder = LabelEncoder()
        data[f'{label}_label'] = label_encoder.fit_transform(data[label])

    for label in tqdm(['hazard-category', 'product-category', 'hazard', 'product']):
        #label_encoder = LabelEncoder()
        data['label'] = data[f'{label}_label']

        # Data preprocessing
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=SEED)
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        # print(train_dataset)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=MAXLEN)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                   num_labels=len(data[label].unique()) ,
                                                                   output_hidden_states=False)
        model.to('cuda')  # Move model to GPU if available

        # training
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        num_training_steps = EPOCHS * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        dd=dict(Counter(data['label'].values))
        class_count = [dd[i] for i in range(len(dd))]

        custom_loss_fn = CustomLoss(whichloss = LOSSFN,
                                   class_count = class_count)
        # comment it for inference only # """
        model.train()
        # progress_bar = tqdm(range(num_training_steps))
        print("Training starting...")
        total_loss_list = []
        best_val_loss = float("inf")
        patience = args.patience  # Number of epochs to wait for improvement
        patience_counter = 0

        for epoch in (range(EPOCHS)):
            curr_ep_loss = 0
            t1 = time.time()
            for batch in tqdm(train_dataloader, desc=f"epoch={epoch+1}"):
                inputs = {k: v.to('cuda') for k, v in batch.items() if k not in ['labels']}  # Move batch to GPU if available
                labels = {k: v.to('cuda') for k, v in batch.items() if k in ['labels']}
                outputs = model(**inputs)
                logits = outputs.logits  # Raw logits from model
                # Compute custom loss
                loss = custom_loss_fn(logits, **labels)
                curr_ep_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                #progress_bar.update(1)
            t2 = time.time()
            curr_ep_loss /= len(train_dataloader)
            #print(f"Epoch {epoch + 1}, Loss: {curr_ep_loss:.4f} | Time : {(t2-t1):.4f} seconds")

            # assess model
            model.eval()
            val_loss = 0
            total_predictions = []
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs = {k: v.to('cuda') for k, v in batch.items() if k not in ['labels']}  # Move batch to GPU if available
                    labels = {k: v.to('cuda') for k, v in batch.items() if k in ['labels']}
                    outputs = model(**inputs)
                    val_loss += custom_loss_fn(outputs.logits, **labels).item()
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    total_predictions.extend([p.item() for p in predictions])

            val_loss /= len(test_dataloader)
            predicted_labels = label_encoder.inverse_transform(total_predictions)
            gold_labels = label_encoder.inverse_transform(test_df.label.values)
            valf1 = f1_score(gold_labels, predicted_labels, average='macro')

            print(f"Epoch {epoch + 1}, TrainLoss: {curr_ep_loss:.4f} | ValLoss: {val_loss:.4f} | ValF1: {valf1:.4f} | Time : {(t2-t1):.4f} seconds")
            #print(f1_score(gold_labels, predicted_labels, average='macro'))

            #model.save_pretrained(os.path.join(args.logdir, f"bert_{label}"))
            #break
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter if validation loss improves
                print("Validation loss improved, saving model...")
                #torch.save(model.state_dict(), "best_model.pth")  # Save the model
                model.save_pretrained(os.path.join(args.logdir, f"bert_{label}"))
            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} epoch(s).")

            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    # ========================= inference =================================================
    print("********************************* INFERENCE ******************************************")

    os.makedirs(args.logdir, exist_ok=True)

    # Create a txt file listing args and values
    args_txt_path = os.path.join(args.logdir, "args_list.txt")
    with open(args_txt_path, "w") as args_file:
        for k, v in sorted(vars(args).items()):
            args_file.write(f"{k}: {v}\n")

    # prediction ST1
    valid_predictions_category = {}
    for label in tqdm(['hazard-category', 'product-category']):
      # Decode predictions back to string labels
      label_encoder = LabelEncoder()
      label_encoder.fit(data[label])
      if args.input == 'text':
          valid_predictions_category[label] = predict(valid.text.to_list(), os.path.join(args.logdir, f'bert_{label}'), MODEL_NAME)
      elif args.input == 'title':
          valid_predictions_category[label] = predict(valid.title.to_list(), os.path.join(args.logdir, f'bert_{label}'), MODEL_NAME)
      elif args.input == 'tt':
          valid_predictions_category[label] = predict(valid.tt.to_list(), os.path.join(args.logdir, f'bert_{label}'), MODEL_NAME)

      valid_predictions_category[label] = label_encoder.inverse_transform(valid_predictions_category[label])

    # save predictions
    solution = pd.DataFrame({'hazard-category': valid_predictions_category['hazard-category'], 'product-category': valid_predictions_category['product-category']})
    solution.to_csv(os.path.join(args.logdir, "submission_st1.csv"), index=False)
    print("submission ST1 created!")

    # prediction ST2
    valid_predictions = {}
    for label in tqdm(['hazard', 'product']):
      # Decode predictions back to string labels
      label_encoder = LabelEncoder()
      label_encoder.fit(data[label])
      if args.input == 'text':
          valid_predictions[label] = predict(valid.text.to_list(), os.path.join(args.logdir, f'bert_{label}'), MODEL_NAME)
      elif args.input == 'title':
          valid_predictions[label] = predict(valid.title.to_list(), os.path.join(args.logdir, f'bert_{label}'), MODEL_NAME)
      elif args.input == 'tt':
          valid_predictions[label] = predict(valid.tt.to_list(), os.path.join(args.logdir, f'bert_{label}'), MODEL_NAME)

      valid_predictions[label] = label_encoder.inverse_transform(valid_predictions[label])

    # save predictions
    solution = pd.DataFrame({'hazard': valid_predictions['hazard'], 'product': valid_predictions['product']})
    solution.to_csv(os.path.join(args.logdir, "submission_st2.csv"), index=False)
    print("submission ST2 created!")

