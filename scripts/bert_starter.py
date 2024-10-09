import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm

data = pd.read_csv('../data/incidents/incidents_train.csv', index_col=0)
valid = pd.read_csv('../data/incidents/incidents_dev.csv', index_col=0)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['title'], padding=True, truncation=True)

def predict(texts, model_path, tokenizer_path="/kaggle/input/bert_tokenizer2/pytorch/default/1"):
    # Load the saved tokenizer and the saved model
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    return predictions.cpu().numpy()


def compute_score(hazards_true, products_true, hazards_pred, products_pred):
  f1_hazards = f1_score(
    hazards_true,
    hazards_pred,
    average='macro'
  )

  # compute f1 for products:
  f1_products = f1_score(
    products_true[hazards_pred == hazards_true],
    products_pred[hazards_pred == hazards_true],
    average='macro'
  )

  return (f1_hazards + f1_products) / 2.


for label in tqdm(['hazard-category', 'product-category', 'hazard', 'product']):
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data[label])

    # Data preprocessing
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=16)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)


    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data[label].unique()))
    model.to('cuda')  # Move model to GPU if available

    # training
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # assess model
    model.eval()
    total_predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_predictions.extend([p.item() for p in predictions])

    predicted_labels = label_encoder.inverse_transform(total_predictions)
    gold_labels = label_encoder.inverse_transform(test_df.label.values)
    # print(classification_report(gold_labels, predicted_labels, zero_division=0))

    model.save_pretrained(f"bert_{label}")


##### PREDICTIONS #####

# prediction ST1
valid_predictions_category = {}
for label in tqdm(['hazard-category', 'product-category']):
  # Decode predictions back to string labels
  label_encoder = LabelEncoder()
  label_encoder.fit(data[label])
  valid_predictions_category[label] = predict(valid.title.to_list(), f'bert_{label}')
  valid_predictions_category[label] = label_encoder.inverse_transform(valid_predictions_category[label])

# save predictions
solution = pd.DataFrame({'hazard-category': valid_predictions_category['hazard-category'], 'product-category': valid_predictions_category['product-category']})
solution.to_csv('/kaggle/working/submission_bert_st1.csv', index=False)
print("submission ST1 created!")

# prediction ST2
valid_predictions = {}
for label in tqdm(['hazard', 'product']):
  # Decode predictions back to string labels
  label_encoder = LabelEncoder()
  label_encoder.fit(data[label])
  valid_predictions[label] = predict(valid.title.to_list(), f'bert_{label}')
  valid_predictions[label] = label_encoder.inverse_transform(valid_predictions[label])

# save predictions
solution = pd.DataFrame({'hazard': valid_predictions['hazard'], 'product': valid_predictions['product']})
solution.to_csv('/kaggle/working/submission_bert_st2.csv', index=False)
print("submission ST2 created!")