from collections import defaultdict
import datetime
import json
from pathlib import Path
import time

from gpt_util import gpt4_prompt
from mongo_util import get_document_col

data_path = Path('/mnt/nlp/albert/clustering/data')
mongo_path = Path('/mnt/nlp/albert/data/mongo_cache')
document_col = get_document_col()
FIELDS = [
    '_id', 'stitle', 'seg_title', 'src', 'seg_content', 'domain',
    'source', 'c_word', 'paragraph_count', 'epoch', 'insert_time', 'simhash',
    'kws', 'kw_title', 'highlightkeyword_list', 'channels', 'channels_v2',
    "ne_content_organization", "spacy_content_org", 
    "ne_content_location", "spacy_content_loc", 
    "ne_content_person", "spacy_content_per", 
    "spacy_content_num", 
    "spacy_content_tim", 
    "ne_title_organization", "spacy_title_org", 
    "ne_title_location", "spacy_title_loc", 
    "ne_title_person", "spacy_title_per", 
    "spacy_title_num", 
    "spacy_title_tim", 
    "text_category", "text_category_v2", 
    "geotag", "geotag_v2", 
    "url", 
    "images_phash", "faces_phash"
]

def loki_get_search_pairs():
    n = 0
    masters = defaultdict(list)
    with open(data_path/'doc_clustering_server_loki.txt') as fin:
        for line in fin:
            n += 1
            # if n < 5000000:
            #     continue
            # if n >= 20:
            #     break
            spl = line.split()
            master = spl[7]
            searcher = spl[9]
            if len(spl) < 12:
                continue
            candidates = spl[11].split(',')
            if master in candidates:
                continue
            if len(masters[master]) == 2:
                masters[master] = (None, None, None)
            elif len(masters[master]) < 2:
                masters[master](searcher, candidates))
    for master, searches in masters.items():
        if (len(searches) == 2
                and (searches[0][0] == 'FaissSearcher' and searches[1][0] == 'ESSearcher'
                     or searches[1][0] == 'FaissSearcher' and searches[0][0] == 'ESSearcher')
                ):
            a = 0 if searches[0][0] == 'FaissSearcher' else 1
            yield master, searches[a][1], searches[1-a][1]

def loki_filter_disjoint():
    # gen = loki_get_search_pairs()
    # i = 0
    # d = {}
    # for master, faiss_cdds, es_cdds in gen:
    #     d[master] = {'master': master, 'faiss': faiss_cdds, 'es': es_cdds}
    #     # print(master, len(faiss_cdds+es_cdds))
    with open(data_path/'v2_loki_map.json') as f:
        data = json.load(f)
    all_search_docids = set()
    for d in data.values():
        for docid in d['faiss'] + d['es']:
            all_search_docids.add(docid)
    for master in list(data.keys()):
        if master in all_search_docids:
            del data[master]
    with open(data_path/'v2_loki_map_disjoint.json', 'w') as f:
        json.dump(data, f)

def cache_article_content(doc_id, projection):
    doc_path = mongo_path / f'{doc_id}.json'
    sdoc = {}
    if doc_path.exists():
        with open(doc_path) as fin:
            sdoc = json.load(fin)
    missing = False
    for k in projection:
        if k not in sdoc:
            missing = True
            break
    if missing or not sdoc:
        doc = document_col.find_one({'_id': doc_id}, projection=projection)
        if doc:
            sdoc.update(doc)
            for k in projection:
                if k not in sdoc:
                    sdoc[k] = None
            with open(doc_path, 'w') as fout:
                json.dump(sdoc, fout)
    return sdoc

def one_article(doc_id, index=None, return_doc=False, chars=1e6):
    print(doc_id)
    doc = cache_article_content(doc_id, FIELDS)
    title = doc.get('stitle', doc.get('seg_title', '')).strip()
    content = doc.get('seg_content', '').strip()
    space_index = f' {index}' if index else ''
    article = f'**Article {index}**\n' if index else ''
    article += f'Title{space_index}: {title}\n'
    article += f'Content{space_index}: {content}'
    if len(article) > chars:
        article = article[:chars-3] + '...'
    return (article, doc) if return_doc else article

def gpt_label_75(master, faiss, es):
    print(len(set(faiss)), len(set(es)), len(set(faiss+es)))
    faissset = set(faiss)
    candidates = faiss + [d for d in es if d not in faissset]
    candidates = candidates[:3] + candidates[5:7]
    master_article, master_doc = one_article(master, index='NEW', return_doc=True)
    master_title = master_doc.get('stitle')
    if not master_title or not master_doc.get('seg_content'):
        return 'missing'
    user = master_article + '\n----------\n----------\n'
    user += '\n----------\n'.join(
        one_article(doc_id, index=i+1, chars=1500)
        for i, doc_id in enumerate(candidates)
    )
    user += '\nWrite your JSON answer now.'
    system = f'''There is a NEW article titled "{master_title}"
Compare the NEW article to each of the old articles, which are numbered starting from 1. Label if the NEW article is a "duplicate", "same event", "different", or "not enough information" compared to each old article. Judge the old articles independently of each other, only relative to the NEW article.

Follow these definitions:
"duplicate" means the articles are the same, or large sections are the same.
"same event" means the articles are reporting the same incident, news, or event. Unlike duplicates, they contain unique details or are written differently.
"different" means the articles are about different incidents, news, or events.
"not enough information" means that the articles are missing too much context for you to judge between duplicate, same event, or different.

Answer with JSON format like this.'''
    system += '''
{
  "article_1_label": "...",
  "article_2_label": "...",
  ...
}'''
    with open('/mnt/nlp/albert/output.txt', 'w') as fout:
        fout.write(system+'\n\n')
        fout.write(user)
    answer = gpt4_prompt([system, user], max_tokens=2000, model='gpt-4o-mini', timeout=30, is_json=True)
    print(answer)

def create():
    with open(data_path/'v2_loki_map_disjoint.json') as f:
        lokis = json.load(f)
    to_delete = set()
    for k, d in lokis.items():
        result = gpt_label_75(d['master'], d['faiss'], d['es'])
        if result == 'missing':
            to_delete.add(k)
            print('deleted', k)
        else:
            break
    if to_delete:
        for k in to_delete:
            del lokis[k]
        with open(data_path/'v2_loki_map_disjoint2.json', 'w') as f:
            json.dump(lokis, f)

if __name__ == '__main__':
    st = time.time()
    
    create()
    
    et = time.time()
    print(et-st, 'seconds')