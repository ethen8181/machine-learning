# Kaggle Rossman Store Sales

Predicting Daily Store Sales. Problem description is available at https://www.kaggle.com/c/rossmann-store-sales/overview/description

## Documentation

- `rossman_data_prep.ipynb` Downloads and prepares the data for downstream modeling. The bulk of the data cleaning and feature engineering is done in this notebook. [[nbviewer](https://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/projects/kaggle_rossman_store_sales/rossman_data_prep.ipynb)][[html](http://ethen8181.github.io/machine-learning/projects/kaggle_rossman_store_sales/rossman_data_prep.html)]
- `rossman_gbt.ipynb` Trains a boosted tree using lightgbm that serves as a baseline model. `gbt_module` and `config` are helper class and configurations that are used in this notebook. [[nbviewer](https://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/projects/kaggle_rossman_store_sales/rossman_gbt.ipynb)][[html](http://ethen8181.github.io/machine-learning/projects/kaggle_rossman_store_sales/rossman_gbt.html)]
- `rossman_deep_learning.ipynb` Trains a fastai deep learning model that showcase the use of embeddings for categorical features. [[nbviewer](https://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/projects/kaggle_rossman_store_sales/rossman_deep_learning.ipynb)][[html](http://ethen8181.github.io/machine-learning/projects/kaggle_rossman_store_sales/rossman_deep_learning.html)]

## Results

Private Leaderboard Score, Root Mean Square Percentage Error (RMSPE):

- boosted tree: 0.1226
- deep learning: 0.1137
- leaderboard score 50th place: 0.1120
- leaderboard score 1st place: 0.1002

Note that the model here is not tuned extensively and no blending/stacking was used.
