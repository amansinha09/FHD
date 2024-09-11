
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from utils import compute_score

# Load the data
data = pd.read_csv('../data/incidents_train.csv', index_col=0)

# Split data
trainset, testset = train_test_split(data, test_size=0.2, random_state=2024)
print("#train_sample: ", len(trainset), " |#test_samples: ", len(testset))

#create sklearn pipeline
text_clf_lr = Pipeline([
    ('vect', TfidfVectorizer(strip_accents='unicode', analyzer='char', ngram_range=(2,5), max_df=0.5, min_df=5)),
     ('clf', LogisticRegression(max_iter=1000)),
    ])

#evaluation
predictions = {}
for label in tqdm(['hazard-category', 'product-category', 'hazard', 'product']):
  print(label.upper())
  text_clf_lr.fit(trainset.title, trainset[label])
  predictions[label] = text_clf_lr.predict(testset.title)
  print(f"F1-micro: {f1_score(testset[label], predictions[label], average='micro'):.2f}")
  print(f"F1-macro: {f1_score(testset[label], predictions[label], average='macro'):.2f}")


for label in ('hazard-category', 'product-category', 'hazard', 'product'):
  testset['lr-'+label] = predictions[label]

print('Score Sub-Task 1:', compute_score(testset['hazard-category'], testset['product-category'], testset['lr-hazard-category'], testset['lr-product-category']))
print('Score Sub-Task 2:', compute_score(testset['hazard'], testset['product'], testset['lr-hazard'], testset['lr-product']))

