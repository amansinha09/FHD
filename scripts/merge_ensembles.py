import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hazard_category', type=str, help='Path to the hazard-category ensemble folder.')
parser.add_argument('--product_category', type=str, help='Path to the product-category ensemble folder.')
parser.add_argument('--hazard', type=str, help='Path to the hazard ensemble folder.')
parser.add_argument('--product', type=str, help='Path to the product ensemble folder.')
parser.add_argument('-plm', '--plm', type=str, help='Language model used.', default='bert')
args = parser.parse_args()

hazard_category_df = pd.read_csv(f'./ensembled_results/{args.hazard_category}/ensemble_hazard-category.csv')
product_category_df = pd.read_csv(f'./ensembled_results/{args.product_category}/ensemble_product-category.csv')
hazard_df = pd.read_csv(f'./ensembled_results/{args.hazard}/ensemble_hazard.csv')
product_df = pd.read_csv(f'./ensembled_results/{args.product}/ensemble_product.csv')

final_submission_st1 = pd.concat([
    hazard_category_df.rename(columns={'hazard_category': 'hazard-category'}),
    product_category_df.rename(columns={'product_category': 'product-category'}),
], axis=1)

final_submission_st2 = pd.concat([
    hazard_df.rename(columns={'hazard': 'hazard'}),
    product_df.rename(columns={'product': 'product'})
], axis=1)

final_submission_st1.to_csv(f'ens_{args.plm}_submission_st1.csv', index=False)
print(f"Final submission file for ST1 created!")

final_submission_st2.to_csv(f'ens_{args.plm}_submission_st2.csv', index=False)
print(f"Final submission file for ST2 created!")
