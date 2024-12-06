import pandas as pd
from datasets import Dataset
from transformers import (BertTokenizer, BertForSequenceClassification,
            AutoModelForSequenceClassification,
            AdamW, get_scheduler, DataCollatorWithPadding, AutoModel, 
            AutoTokenizer)
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

import seaborn as sns
from collections import Counter
import time

from losses import CustomLoss
from utils import predict, seed_everything



import argparse

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-data_dir','--data_dir', type=str, help='Path to FHD folder', default='/kaggle/input/fhd-data/')
parser.add_argument('-save_dir','--save_dir', type=str, help='Path to LOG folder', default='/kaggle/working/')
parser.add_argument('-whichloss','--whichloss', type=str, help='Type of loss', default='softmax', choices=['softmax', 'wsoftmax', 'focalloss', 
                                                                                                                           'classbalancedloss', 'balancedsoftmax',
                                                                                                                           'equalizationloss', 'ldamloss'])
parser.add_argument('-model_name','--model_name', type=str, help='Name of model', default='bert-base-uncased')
parser.add_argument('-num_epochs','--num_epochs', type=int, help='Number of epochs', default=10)
args = parser.parse_args()

DATADIR = args.data_dir
SAVEDIR = args.save_dir
MODEL_NAME = args.model_name
LOSSFN = args.whichloss
EPOCHS = args.num_epochs
MAXLEN = 256
SEED = 7869786976


seed_everything(SEED)
# ======================= dataset loading ==================================


data = pd.read_csv(DATADIR +'data/incidents_train.csv', index_col=0)
valid = pd.read_csv(DATADIR+'data/incidents_dev.csv', index_col=0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples['title'], padding='max_length', max_length=MAXLEN, truncation=True)
# ==========================================================================

# ========================== main run =====================================
for label in tqdm(['hazard-category', 'product-category', 'hazard', 'product']):
    label_encoder = LabelEncoder()
    data[f'{label}_label'] = label_encoder.fit_transform(data[label])

for label in tqdm(['hazard-category', 'product-category', 'hazard', 'product']):
    #label_encoder = LabelEncoder()
    data['label'] = data[f'{label}_label']

    # Data preprocessing
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #train_df, test_df = train_df.iloc[:100,:], test_df
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    print(train_dataset)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=MAXLEN)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, 
                                                               num_labels=len(data[label].unique()) , 
                                                               output_hidden_states=False)
    #model = AutoModel.from_pretrained('bert-base-uncased')
    model.to('cuda')  # Move model to GPU if available

    # training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = EPOCHS
    num_training_steps = num_epochs * len(train_dataloader)
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

    model.train()
    #progress_bar = tqdm(range(num_training_steps))
    print("training starting ..")
    total_loss_list = []
    
    for epoch in (range(num_epochs)):
        curr_ep_loss = 0
        t1 = time.time()
        for batch in tqdm(train_dataloader, desc=f"epoch={epoch+1}"):
            inputs = {k: v.to('cuda') for k, v in batch.items() if k not in ['labels']}  # Move batch to GPU if available
            labels = {k: v.to('cuda') for k, v in batch.items() if k in ['labels']}
            outputs = model(**inputs)
            #print(outputs.last_hidden_state.shape, outputs.pooler_output.shape)
            #loss = outputs.loss
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
        print(f"Epoch {epoch + 1}, Loss: {curr_ep_loss:.4f} | Time : {(t2-t1):.4f} seconds")

    # assess model
    model.eval()
    total_predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to('cuda') for k, v in batch.items() if k not in ['labels']}  # Move batch to GPU if available
            labels = {k: v.to('cuda') for k, v in batch.items() if k in ['labels']}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_predictions.extend([p.item() for p in predictions])

    predicted_labels = label_encoder.inverse_transform(total_predictions)
    gold_labels = label_encoder.inverse_transform(test_df.label.values)
    print(classification_report(gold_labels, predicted_labels, zero_division=0))

    model.save_pretrained(SAVEDIR+f"bert_{label}")
    #break
    
# ========================= inference =================================================
##### PREDICTIONS #####

# prediction ST1
valid_predictions_category = {}
for label in tqdm(['hazard-category', 'product-category']):
  # Decode predictions back to string labels
  label_encoder = LabelEncoder()
  label_encoder.fit(data[label])
  valid_predictions_category[label] = predict(valid.title.to_list(), SAVEDIR+f'bert_{label}',MODEL_NAME)
  valid_predictions_category[label] = label_encoder.inverse_transform(valid_predictions_category[label])

# save predictions
solution = pd.DataFrame({'hazard-category': valid_predictions_category['hazard-category'], 'product-category': valid_predictions_category['product-category']})
solution.to_csv(SAVEDIR+f'submission_bert_{LOSSFN}_st1.csv', index=False)
print("submission ST1 created!")

# prediction ST2
valid_predictions = {}
for label in tqdm(['hazard', 'product']):
  # Decode predictions back to string labels
  label_encoder = LabelEncoder()
  label_encoder.fit(data[label])
  valid_predictions[label] = predict(valid.title.to_list(), SAVEDIR+f'bert_{label}', MODEL_NAME)
  valid_predictions[label] = label_encoder.inverse_transform(valid_predictions[label])

# save predictions
solution = pd.DataFrame({'hazard': valid_predictions['hazard'], 'product': valid_predictions['product']})
solution.to_csv(SAVEDIR +f'submission_bert_{LOSSFN}_st2.csv', index=False)
print("submission ST2 created!")

