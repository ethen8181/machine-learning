"""
Task : Predict if a car purchased at auction is a unfortunate purchase.
Output : .csv file containing the prediction
"""
import os
import logging
import argparse
import numpy as np
from joblib import dump, load
from logzero import setup_logger
from sortedcontainers import SortedSet
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from mlutils.transformers import Preprocesser
from utils import clean, build_xgb, write_output
logger = setup_logger(name = __name__, logfile = 'data_challenge.log', level = logging.INFO)


def main():
    # -----------------------------------------------------------------------------------
    # Adjustable Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action = 'store_true', help = 'training or scoring')
    parser.add_argument(
        '--inputfile', type = str, help = 'input data file name')
    parser.add_argument(
        '--outputfile', type = str, help = 'output prediction file name')
    args = parser.parse_args()

    # directory for the input data and output prediction:
    DATA_DIR = 'data'
    OUTPUT_DIR = 'output'

    # columns used:
    CAT_COLS = ['Auction', 'Transmission', 'WheelType', 'Nationality',
                'Size', 'TopThreeAmericanName', 'IsOnlineSale']
    NUM_COLS = ['VehicleAge', 'VehOdo', 'VehBCost', 'WarrantyCost',
                'MMRCurrentAuctionAveragePrice', 'MMRAcquisitionAuctionAveragePrice',
                'MMRCurrentAuctionCleanPrice', 'MMRAcquisitionAuctionCleanPrice',
                'MMRCurrentRetailAveragePrice', 'MMRAcquisitionRetailAveragePrice',
                'MMRCurrentRetailCleanPrice', 'MMRAcquisitonRetailCleanPrice']
    DATE_COLS = ['PurchDate']
    LABEL_COL = 'IsBadBuy'
    IDS_COL = 'RefId'

    # current time for computing recency feature
    NOW = '2010-12-31'

    # modeling step:
    # model checkpoint for future scoring
    MODEL_DIR = 'model'
    CHECKPOINT_XGB = 'xgb.pkl'
    CHECKPOINT_PREPROCESS = 'preprocess.pkl'

    # parameter that only relevant for training stage and not scoring
    if args.train:
        # number of cross validation and hyperparameter settings to try
        CV = 10
        N_ITER = 5
        MODEL_RANDOM_STATE = 4321

        # train/validation stratified split
        VAL_SIZE = 0.1
        TEST_SIZE = 0.1
        SPLIT_RANDOM_STATE = 1234

    # -----------------------------------------------------------------------------------
    logger.info('preprocessing')
    checkpoint_preprocess = os.path.join(MODEL_DIR, CHECKPOINT_PREPROCESS)
    checkpoint_xgb = os.path.join(MODEL_DIR, CHECKPOINT_XGB)
    input_path = os.path.join(DATA_DIR, args.inputfile)

    if args.train:
        data = clean(input_path, NOW, CAT_COLS, NUM_COLS, DATE_COLS, IDS_COL, LABEL_COL)
        ids = data[IDS_COL].values
        label = data[LABEL_COL].values
        data = data.drop([IDS_COL, LABEL_COL], axis = 1)

        # train/test split twice to achieve train/validation/test three way split
        df_train, df_test, y_train, y_test, ids_train, ids_test = train_test_split(
            data, label, ids, test_size = TEST_SIZE,
            random_state = SPLIT_RANDOM_STATE, stratify = label)

        df_train, df_val, y_train, y_val, ids_train, ids_val = train_test_split(
            df_train, y_train, ids_train, test_size = VAL_SIZE,
            random_state = SPLIT_RANDOM_STATE, stratify = y_train)

        # obtain finalized columns
        num_cols_cleaned = list(SortedSet(df_train.columns) - SortedSet(CAT_COLS))
        preprocess = Preprocesser(num_cols = num_cols_cleaned, cat_cols = CAT_COLS)
        X_train = preprocess.fit_transform(df_train)
        X_val = preprocess.transform(df_val)
        X_test = preprocess.transform(df_test)

        logger.info('modeling')
        eval_set = [(X_train, y_train), (X_val, y_val)]
        xgb_tuned = build_xgb(N_ITER, CV, MODEL_RANDOM_STATE, eval_set)
        xgb_tuned.fit(X_train, y_train)

        if not os.path.isdir(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        dump(preprocess, checkpoint_preprocess)
        dump(xgb_tuned, checkpoint_xgb)

        # model evaluation metric reporting
        y_pred = []
        xgb_best = xgb_tuned.best_estimator_
        zipped = zip(
            ('train', 'validation', 'test'),
            (X_train, X_val, X_test),
            (y_train, y_val, y_test))
        for name, X, y in zipped:
            xgb_pred = xgb_best.predict_proba(
                X, ntree_limit = xgb_best.best_ntree_limit)[:, 1]
            score = round(roc_auc_score(y, xgb_pred), 3)
            logger.info('{} AUC: {}'.format(name, score))
            y_pred.append(xgb_pred)

        ids = np.hstack((ids_train, ids_val, ids_test))
        y_pred = np.hstack(y_pred)
    else:
        data = clean(input_path, NOW, CAT_COLS, NUM_COLS, DATE_COLS, IDS_COL)
        ids = data[IDS_COL].values
        data = data.drop(IDS_COL, axis = 1)

        logger.info('scoring')
        preprocess = load(checkpoint_preprocess)
        xgb_tuned = load(checkpoint_xgb)
        X = preprocess.transform(data)
        xgb_best = xgb_tuned.best_estimator_
        y_pred = xgb_best.predict_proba(
            X, ntree_limit = xgb_best.best_ntree_limit)[:, 1]

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, args.outputfile)
    write_output(ids, IDS_COL, y_pred, LABEL_COL, output_path)


if __name__ == '__main__':
    main()
