import json
from pathlib import Path

data_path = Path('/mnt/nlp/albert/clustering/data')

def fields_to_stitle():
    for pre in ('dup', 'evt'):
        old_fields = data_path / f'dedup_train_data/{pre}_train_fields'
        old_stitles = data_path / f'{pre}_train_stitles.tsv'
        with (
                open(old_fields) as fin,
                open(old_stitles, 'w') as fout
                ):
            for line in fin:
                id1, id2, label, succ, feat1, feat2 = line.split('\t')
                if succ != 'SUCCESS':
                    continue
                stitle1 = json.loads(feat1).get('stitle', '')
                stitle2 = json.loads(feat2).get('stitle', '')
                fout.write('\t'.join((id1, id2, label, stitle1, stitle2))+'\n')


if __name__ == '__main__':
    fields_to_stitle()