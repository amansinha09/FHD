import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-hc', '--hazard_category', type=str, help='Path to the hazard-category ensemble file.')
parser.add_argument('-pc', '--product_category', type=str, help='Path to the product-category ensemble file.')
parser.add_argument('-h', '--hazard', type=str, help='Path to the hazard ensemble file.')
parser.add_argument('-p', '--product', type=str, help='Path to the product ensemble file.')
args = parser.parse_args()

hazard_category_df = pd.read_csv(args.hazard_category)
product_category_df = pd.read_csv(args.product_category)
hazard_df = pd.read_csv(args.hazard)
product_df = pd.read_csv(args.product)

final_submission = pd.concat([
    hazard_category_df.rename(columns={'hazard-category': 'hazard-category'}),
    product_category_df.rename(columns={'product-category': 'product-category'}),
    hazard_df.rename(columns={'hazard': 'hazard'}),
    product_df.rename(columns={'product': 'product'})
], axis=1)

final_output_path = 'final_ensembled_submission.csv'
final_submission.to_csv(final_output_path, index=False)

print(f"Final submission file created at {final_output_path}!")
