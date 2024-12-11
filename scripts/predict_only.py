# To be run in the main FHD folder.

import os
import pandas as pd
import argparse

from sklearn.preprocessing import LabelEncoder

from utils import predict

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', '--model_name', type=str, help='Name of model', default='bert-base-uncased')
parser.add_argument('-data_dir', '--data_dir', type=str, help='Path to FHD folder', default='')
parser.add_argument('-input', '--input', type=str, help='What to train on (text/title).', default='title', choices=['title', 'text'])
parser.add_argument('-logdir', '--logdir', type=str, help='Path to an existing logdir to store submissions.', default='..')

args = parser.parse_args()

data = pd.read_csv(args.data_dir + 'data/incidents_train.csv', index_col=0)
valid = pd.read_csv(args.data_dir + 'data/incidents_dev.csv', index_col=0)

print("********************************* INFERENCE ******************************************")

# prediction ST1
valid_predictions_category = {}
for label in ['hazard-category', 'product-category']:
    # Decode predictions back to string labels
    label_encoder = LabelEncoder()
    label_encoder.fit(data[label])
    if args.input == 'text':
        valid_predictions_category[label] = predict(valid.text.to_list(), os.path.join(args.logdir, f'bert_{label}'), args.model_name)
    elif args.input == 'title':
        valid_predictions_category[label] = predict(valid.title.to_list(), os.path.join(args.logdir, f'bert_{label}'), args.model_name)
    valid_predictions_category[label] = label_encoder.inverse_transform(valid_predictions_category[label])

# save predictions
solution = pd.DataFrame({'hazard-category': valid_predictions_category['hazard-category'],
                         'product-category': valid_predictions_category['product-category']})
solution.to_csv(os.path.join(args.logdir, "submission_st1.csv"), index=False)
print("submission ST1 created!")

# prediction ST2
valid_predictions = {}
for label in ['hazard', 'product']:
    # Decode predictions back to string labels
    label_encoder = LabelEncoder()
    label_encoder.fit(data[label])
    if args.input == 'text':
        valid_predictions[label] = predict(valid.text.to_list(), os.path.join(args.logdir, f'bert_{label}'), args.model_name)
    elif args.input == 'title':
        valid_predictions[label] = predict(valid.title.to_list(), os.path.join(args.logdir, f'bert_{label}'), args.model_name)
    valid_predictions[label] = label_encoder.inverse_transform(valid_predictions[label])

# save predictions
solution = pd.DataFrame({'hazard': valid_predictions['hazard'], 'product': valid_predictions['product']})
solution.to_csv(os.path.join(args.logdir, "submission_st2.csv"), index=False)
print("submission ST2 created!")
