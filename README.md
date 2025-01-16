# FHD
Code repository for project Food hazard detection Challenge


## To convert formatted file to submission zip required file format
```
python submission_folder/converter.py --filename <name_of_csv_file.csv>
```

## To run eval script
```
cd scripts/
python run_eval.py --gold ../data/incidents_valid.csv --pred <path-to-submission-file>
```

## To run ensembling on individual categories
```
python ./scripts/ensemble_individual.py -l <category (e.g. hazard)> -i1 <path/to/folder/of/experiment/1> -i2 <path/to/folder/of/experiment/2> -i3 <path/to/folder/of/experiment/3> -i4 <path/to/folder/of/experiment/4> -i5 <path/to/folder/of/experiment/5>
```

## To merge ensembled results into a single submission file
```
python merge_ensembles.py --hazard_category path/to/ensemble_hazard_category.csv  --product_category path/to/ensemble_product_category.csv --hazard path/to/ensemble_hazard.csv --product path/to/ensemble_product.csv
```
