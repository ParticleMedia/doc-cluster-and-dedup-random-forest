import itertools
from pathlib import Path
import random
import time

import numpy as np
from sklearn.model_selection import *
import xgboost as xgb

data_path = Path('/mnt/nlp/albert/clustering/data/dedup_train_data_v3')

def get_data(data_path, format='arff', val=0.1, seed=None, remove_feat=None):
    if remove_feat is None:
        remove_feat = set()
    with open(data_path) as fin:
        arff_data = fin.read().split('@data')[1].strip('\n')
    features = []
    labels = []
    lines = arff_data.split('\n')
    random.shuffle(lines)
    for line in lines:
        row = []
        label = None
        for el in line.split(','):
            if el == '?':
                row.append(np.nan)
            elif el in ('DUP', 'EVENT', 'DIFF'):
                if label is not None:
                    raise ValueError('multiple labels')
                label = 0 if el == 'DIFF' else 1 if el == 'EVENT' else 2
            else:
                row.append(float(el))
        if label is None:
            raise ValueError('label not found')
        row = [n for i, n in enumerate(row) if i not in remove_feat]
        features.append(row)
        labels.append(label)
    if val == 0:
        X_train, y_train = np.array(features), np.array(labels)
        return X_train, y_train
    else:
        X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), random_state=seed, test_size=val)
        return X_train, X_test, y_train, y_test


def train_xgboost(data_path, seed=42):
    remove_feat = None # {48, 50, 43, 39}
    X_train, X_test, y_train, y_test = get_data(data_path/'train.arff', val=0.1, seed=seed, remove_feat=remove_feat)
    print(X_train.shape)
    num_class = len(np.unique(y_train))
    params = {
        "max_depth": [7, 8],
        "subsample": [0.6, 0.8],
        "gamma": [1],
        "learning_rate": [0.05],
    }
    keys = list(params.keys())
    permutations = itertools.product(*list(params.values()))
    best_score = 0
    for perm in permutations:
        kwargs = {k: v for k, v in zip(keys, perm)}
        model = xgb.XGBClassifier(
            n_estimators=1000,
            objective="multi:softprob",
            silent=True,
            nthread=2,
            missing=np.nan,
            eval_metric="auc",
            early_stopping_rounds=5,
            random_state=seed,
            verbosity=0,
            num_class=num_class,
            **kwargs
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=None)
        print(kwargs, model.best_iteration, model.best_score)
        if model.best_score > best_score:
            print('    ^ new best')
            best_score = model.best_score
            model.save_model(data_path/"xgb_model.json")
    model.load_model(data_path/"xgb_model.json")
    predict(data_path, model, remove_feat=remove_feat)

def train_xgboost_xgb(data_path, seed=42):
    X_train, X_test, y_train, y_test = get_data(data_path/'train.arff', val=0.1, seed=seed)
    dtrain = xgb.DMatrix(X_train, y_train, weight=None, missing=np.nan)
    dval = xgb.DMatrix(X_test, y_test, weight=None, missing=np.nan)
    print(dtrain.num_row(), dtrain.num_col())
    num_class = len(np.unique(dtrain.get_label()))
    param = {
        "max_depth": 6,
        "subsample": 0.8,
        "gamma": 1,
        "eta": 0.1,
        "seed": seed,
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "auc",
    }
    watchlist = [(dtrain, "train"), (dval, "eval")]
    bst = xgb.train(
        param, dtrain,
        num_boost_round=100,
        evals=watchlist,
        verbose_eval=True,
        early_stopping_rounds=3,
    )
    
    bst.save_model(data_path/"xgb_model.json")
    bst.dump_model(data_path/"xgb_dump.txt")
    predict(data_path, bst)

def predict(data_path, bst, remove_feat=None):
    fmt = 'xgb' if isinstance(bst, xgb.Booster) else 'sk'
    X_test, y_test = get_data(data_path/'test.arff', val=0, remove_feat=remove_feat)
    # run prediction
    if fmt == 'xgb':
        dval = xgb.DMatrix(X_test, y_test, weight=None, missing=np.nan)
        preds = bst.predict(dval)
    else:
        preds = bst.predict_proba(X_test)
    correct = 0
    confusion = [[0]*3 for _ in range(3)]
    with open(data_path/'badcase', 'w') as fout:
        for pred, label in zip(preds, y_test):
            difScr, evtScr, dupScr = pred
            if dupScr >= evtScr and dupScr >= difScr:
                pCls = "DUP"
            elif evtScr >= dupScr and evtScr - difScr >= 0.2:
                pCls = "EVENT"
            else:
                pCls = "DIFF"
            pIndex = 0 if pCls == "DIFF" else 1 if pCls == "EVENT" else 2
            rIndex = int(label)
            confusion[rIndex][pIndex] += 1
            rCls = "DIFF" if label == 0 else "EVENT" if label == 1 else "DUP"
            if pCls == rCls:
                correct += 1
            # else:
            #     fout.write()
    
    summary = f"Accuracy: {correct/len(preds)}"
    summary += "\nReal\\Predicted\tDIFF\tEVENT\tDUP"
    summary += "\nDIFF\t" + "\t".join(map(str, confusion[0]))
    summary += "\nEVENT\t" + "\t".join(map(str, confusion[1]))
    summary += "\nDUP\t" + "\t".join(map(str, confusion[2]))
    print(summary)
    with open(data_path/"summary", "w") as fout:
        fout.write(summary)

def run_xgboost(data_path):
    bst = xgb.Booster(model_file=data_path/"xgb_model.json")
    # bst.load_model(data_path/"xgb_model.json")
    l = sorted(bst.get_fscore().items(), key=lambda t: -t[1])
    for t in l:
        print(t)

if __name__ == '__main__':
    st = time.time()
    train_xgboost(data_path)
    run_xgboost(data_path)
    
    et = time.time()
    print(f'took {et-st:.2f}s')