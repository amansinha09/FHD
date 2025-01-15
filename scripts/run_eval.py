import pandas as pd
import argparse

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()

parser.add_argument('-gold_file','--gold_file', type=str, help='Path to the gold file', required=True)
parser.add_argument('-pred_file', '--pred_file', type=str, help='Path to the submission file', required=True)

args = parser.parse_args()


goldfile = args.gold_file
predfile = args.pred_file

golddf = pd.read_csv(goldfile)
preddf = pd.read_csv(predfile)

#print( golddf.columns, preddf.columns)

def compute_score(hazards_true, products_true, hazards_pred, products_pred):
  # compute f1 for hazards:
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


for col in preddf.columns:
    
    gold_labels = golddf[col].values
    predicted_labels = preddf[col].values
    print(f"==Macro-f1==> {col} : {f1_score(gold_labels, predicted_labels, average='macro'):.4f}")


if 'hazard-category' in preddf.columns:
   st = 1
   hname = 'hazard-category'
   pname = 'product-category'
else:
   st = 2
   hname = 'hazard'
   pname = 'product'

hazards_true, products_true, hazards_pred, products_pred = golddf[hname], golddf[pname], preddf[hname], preddf[pname]

print(f"Subtask {st} score: {compute_score(hazards_true, products_true, hazards_pred, products_pred):.4f}")
