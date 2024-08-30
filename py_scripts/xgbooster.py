from pathlib import Path
import random
import time

import numpy as np
import xgboost as xgb

data_path = Path('/mnt/nlp/albert/clustering/data/dedup_train_data_v3')

def get_dmatrix(data_path, format='arff', val=0.1):
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
        features.append(row)
        labels.append(label)
    split = int(len(lines) * (1 - val))
    dtrain = xgb.DMatrix(features[:split], labels[:split], weight=None, missing=np.nan)
    if val > 0:
        dval = xgb.DMatrix(features[split:], labels[split:], weight=None, missing=np.nan)
        return dtrain, dval
    else:
        return dtrain
        

def train_xgboost(data_path, seed=42):
    random.seed(seed)
    dtrain, dval = get_dmatrix(data_path/'train.arff', val=0.1)
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

def predict(data_path, bst, dval=None):
    if dval is None:
        dval = get_dmatrix(data_path/'test.arff', val=0)
    # run prediction
    preds = bst.predict(dval)
    labels = dval.get_label()
    correct = 0
    confusion = [[0]*3 for _ in range(3)]
    with open(data_path/'badcase', 'w') as fout:
        for pred, label in zip(preds, labels):
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
    # train_xgboost(data_path)
    run_xgboost(data_path)