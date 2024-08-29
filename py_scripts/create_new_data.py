from collections import defaultdict
import datetime
from hashlib import md5
import json
from pathlib import Path
import pickle
import random
import time
import traceback

import requests
import tqdm

from gpt_util import gpt4_prompt, print_usage
from mongo_util import get_document_col

data_path = Path('/mnt/nlp/albert/clustering/data')
mongo_path = Path('/mnt/nlp/albert/data/mongo_cache')
document_col = get_document_col()
FIELDS = [
    '_id', 'stitle', 'seg_title', 'src', 'seg_content', 'domain', 'content_type',
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
                masters[master].append((searcher, candidates))
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

def cache_article_content(doc_ids, projection, skip_cache=False):
    sdocs = {k: dict() for k in doc_ids}
    if not skip_cache:
        mongo_docids = []
        for doc_id in doc_ids:
            doc_path = mongo_path / f'{doc_id}.json'
            sdoc = {}
            if doc_path.exists():
                with open(doc_path) as fin:
                    try:
                        sdoc = json.load(fin)
                    except json.decoder.JSONDecodeError:
                        sdoc = {}
                        doc_path.unlink()
            missing = False
            for k in projection:
                if k not in sdoc:
                    missing = True
                    break
            sdocs[doc_id] = sdoc
            if missing or not sdoc:
                mongo_docids.append(doc_id)
            else:
                sdocs[doc_id] = sdoc
    else:
        mongo_docids = doc_ids
    
    for d in document_col.find({'_id': {'$in': mongo_docids}}, projection=projection):
        sdocs[d['_id']].update(d)
        for k in projection:
            if k not in sdocs[d['_id']]:
                sdocs[d['_id']][k] = None
        if not skip_cache:
            doc_path = mongo_path / f'{d["_id"]}.json'
            with open(doc_path, 'w') as fout:
                json.dump(sdocs[d['_id']], fout)
    
    return [sdocs[k] for k in doc_ids]

def one_article(doc_id, return_doc=False, chars=1e6, doc=None):
    doc = cache_article_content([doc_id], FIELDS)[0] if doc is None else doc
    title = doc.get('stitle', doc.get('seg_title', '')).strip()
    content = doc.get('seg_content', '').strip()
    article = '**Article{space_index}**\n'
    article += f'Title{{space_index}}: {title}\n'
    article += f'Content{{space_index}}: {content}'
    if len(article) > chars:
        article = article[:chars-3] + '...'
    return (article, doc) if return_doc else article

def gpt_label_75(master, faiss, es, batch_size=5):
    master_article, master_doc = one_article(master, return_doc=True)
    master_article = master_article.format(space_index=' NEW')
    master_title = master_doc.get('stitle')
    if not master_title or not master_doc.get('seg_content'):
        return 'missing'
    labels = {}
    print(f'master {master} {master_title}')
    
    faissset = set(faiss)
    candidates = faiss + [d for d in es if d not in faissset]
    candidates.sort()
    docid_to_article = {}
    docid_to_doc = {}
    for docid in candidates:
        article, doc = one_article(docid, chars=1500, return_doc=True)
        docid_to_article[docid] = article
        docid_to_doc[docid] = doc
    index_to_article = {}
    index_to_docids = defaultdict(list)
    article_to_index = {}
    for docid, article in docid_to_article.items():
        if article in article_to_index:
            index = article_to_index[article]
            index_to_docids[index].append(docid)
        else:
            index = len(article_to_index) + 1
            article_to_index[article] = index
            index_to_docids[index].append(docid)
            index_to_article[index] = article
    
    system = f'''There is a NEW article titled "{master_title}"
Compare the NEW article to each of the old articles, which are numbered starting from 1. Label if the NEW article is a "not enough information", "different", "same event", or "duplicate" compared to each old article. Judge the old articles independently of each other, only relative to the NEW article.

Follow these definitions:
"not enough information" means that the articles are missing too much context for you to judge accurately between duplicate, same event, or different.
"different" means the articles are about different incidents, news, or events.
"same event" means the articles are reporting the same incident, news, or event. Unlike duplicates, they contain unique details or are written differently.
"duplicate" means the articles are the same, or large sections are the same.

Answer with JSON format like this.'''
    system += '''
{
  "article_1_label": "...",
  "article_2_label": "...",
  ...
}

'''
    system += master_article
    
    for start_index in range(0, len(index_to_article), batch_size):
        batch_articles = []
        for i in range(1, batch_size + 1):
            if i+start_index >= len(index_to_article):
                break
            batch_articles.append(index_to_article[i+start_index].format(space_index=f' {i}'))
        user = '\n----------\n'.join(batch_articles)
        user += '\n----------\nWrite your JSON answer now.'
        answer = gpt4_prompt([system, user], max_tokens=2000, model='gpt-4o-mini', timeout=30, is_json=True)
        print(answer)
        answer = json.loads(answer)
        for i in range(1, len(batch_articles)+1):
            key = f'article_{i}_label'
            if key in answer:
                label = answer[key]
                for docid in index_to_docids[i + start_index]:
                    labels[docid] = {
                        'label': label,
                        'features': docid_to_doc[docid]
                    }
                    print(f'{docid}: {label} {docid_to_doc[docid].get("stitle")}')
    return labels

def gpt_label_RF(master, faiss, es):
    master_article, master_doc = one_article(master, return_doc=True)
    master_title = master_doc.get('stitle')
    master_content = master_doc.get('seg_content')
    if not master_title or not master_content:
        return None, 'missing'
    if master_doc.get('domain') == 'msn.com':
        return None, 'skip msn.com'
    # print(f'master {master} {master_title}')
    
    faissset = set(faiss)
    candidates = faiss + [d for d in es if d not in faissset]
    docid_to_article = {}
    docid_to_doc = {}
    features = cache_article_content(candidates, projection=FIELDS)
    for docid, doc in zip(candidates, features):
        article = one_article(docid, return_doc=False, doc=doc)
        if not doc.get('stitle') or not doc.get('seg_content') or doc.get('domain') == 'msn.com':
            continue
        docid_to_article[docid] = article
        docid_to_doc[docid] = doc
    
    # grab one random EVENT or DUP from prod v2 label
    results = call_prod_RF(master_doc, list(docid_to_doc.values()))
    candidates_RF = [res for res in results if res.get('label') in ('EVENT', 'DUP')]
    if not candidates_RF:
        return None, 'all DIFF'
    candidates_RF.sort(key=lambda d: md5((master+d['doc']).encode()).hexdigest())
    candidate_RF = candidates_RF[0]
    candidate_docid = candidate_RF['doc']
    label, timeline = gpt_label_1(master_article, docid_to_article[candidate_docid])
    c = docid_to_doc[candidate_docid]
    return (master, candidate_docid,
            candidate_RF['label'], candidate_RF.get('score', ''),
            label, timeline,
            master_doc['content_type'], master_title, master_content,
            c['content_type'], c['stitle'], c['seg_content'])

def gpt_label_1(article1, article2, model='gpt-4o'):
    system = f'''Compare the two articles. Choose a label from "not enough information", "different", "same event", or "duplicate". Consider the main focuses of the articles. Do not consider the articles' minor mentions.

Follow these definitions for the label:
"not enough information" means that the articles are missing too much context for you to judge accurately between duplicate, same event, or different.
"different" means the articles are about different incidents, news, or events.
"same event" means the articles are reporting the same incident, news, or event. Unlike duplicates, they contain unique details or are written differently.
"duplicate" means the articles are the same, or large sections are the same. For example, plagiarized articles are duplicates.

Then, if the label is "same event" or "duplicate", try to decide if one article contains updates superseding the other. Fill in the "timeline" field:
"N/A": The label is "not enough information" or "different"
"uncertain": It is difficult to compare the articles.
"both contain important information": Each article contains important information.
"A replaces B": All information in Article B is also in Article A, and Article A contains more information.
"B replaces A": All information in Article A is also in Article B, and Article B contains more information.

Answer with JSON format like this, and write nothing else.'''
    system += '''
{
  "label": "...",
  "timeline": "..."
}'''
    article1 = article1.format(space_index=' A')
    article2 = article2.format(space_index=' B')
    user = article1 + '\n----------\n' + article2
    answer = gpt4_prompt([system, user], temperature=0.01, timeout=15, max_tokens=40, is_json=True,
                         model=model)
    # with open('/mnt/nlp/albert/output.txt', 'w') as fout:
    #     fout.write(system+'\n\n')
    #     fout.write(user)
    if not answer:
        return None, None
    answer = json.loads(answer) if isinstance(answer, str) else answer
    slabel = answer.get('label', '')
    if slabel == 'duplicate':
        label = 'DUP'
    elif slabel == 'same event':
        label = 'EVENT'
    elif slabel == 'different':
        label = 'DIFF'
    elif slabel == 'not enough information':
        label = 'NEI'
    else:
        label = ''
    stime = answer.get('timeline', '')
    return label, stime

def gpt_label_1_v2(article1, article2, model='gpt-4o'):
    system = '''Compare the two articles. Choose a label from "not enough information", "different", "same event", or "duplicate". Consider the main focuses of the articles. Do not consider the articles' minor mentions.

Follow these definitions for the label:
"not enough information" means that the articles are missing too much context for you to judge accurately between duplicate, same event, or different.
"different" means the articles are about different incidents, news, or events.
"same event" means the articles are reporting the same incident, news, or event. Unlike duplicates, they contain unique details or are written differently.
"duplicate" means the articles are the same, or large sections are the same. For example, plagiarized articles are duplicates. Or one article may be very short and was taken from the other article.

Answer with JSON format like this, and write nothing else.
{"label": "..."}'''
    article1 = article1.format(space_index=' A')
    article2 = article2.format(space_index=' B')
    user = f'<ArticleA>{article1}</ArticleA>\n\n<ArticleB>{article2}</ArticleB>'
    answer = gpt4_prompt([system, user], temperature=0.01, timeout=15, max_tokens=20,
                         is_json=True, model=model)
    # with open('/mnt/nlp/albert/output.txt', 'w') as fout:
    #     fout.write(system+'\n\n')
    #     fout.write(user)
    if not answer:
        return None, None
    answer = json.loads(answer) if isinstance(answer, str) else answer
    slabel = answer.get('label', '')
    if slabel == 'duplicate':
        label = 'DUP'
    elif slabel == 'same event':
        label = 'EVENT'
    elif slabel == 'different':
        label = 'DIFF'
    elif slabel == 'not enough information':
        label = 'uncertain'
    else:
        label = None
    
    timeline = None
    if label in ('EVENT', 'DUP'):
        system = '''Compare the two articles. Decide if one article contains updates superseding the other. Fill in the "label" field with one of these four choices:
    "unique details": Each article contains important information that does not appear in the other article.
    "identical details": The articles contain the same information.
    "A replaces B": All information in Article B is also in Article A, and Article A contains more information. After reading Article A, you would gain nothing from Article B.
    "B replaces A": All information in Article A is also in Article B, and Article B contains more information. After reading Article B, you would gain nothing from Article A.

    Answer with JSON format like this, and write nothing else.
    {"label": "..."}'''
        answer = gpt4_prompt([system, user], temperature=0.01, timeout=15, max_tokens=20,
                            is_json=True, model=model)
        if answer:
            answer = json.loads(answer) if isinstance(answer, str) else answer
            timeline = answer.get('label', None)
    else:
        timeline = ''
    return label, timeline

valid_labels = ('DUP', 'EVENT', 'DIFF')
valid_timelines = ('both contain important information', 'B replaces A', 'A replaces B')
def gpt_check_2(article1, article2, label1, timeline1, model='gpt-4o', give_wrong_answer=False):
    if label1 not in valid_labels:
        return label1, timeline1
    if give_wrong_answer:
        rn = ord(article1[999 % len(article1)])
        labels_choices = [l for l in valid_labels if l != label1]
        time_choices = [l for l in valid_timelines if l != timeline1]
        label1 = labels_choices[rn % len(labels_choices)]
        timeline1 = 'N/A' if label1 == 'DIFF' else time_choices[rn % len(time_choices)]
    
    system = '''You are an English teacher who will grade your student's work. The student has read two articles and analyzed them. Your job is to judge whether the student's answers are correct.

The two questions are
1. Are the articles identical, about the same event, or about different events? Choose between "DIFFERENT", "SAME EVENT", and "DUPLICATE".
2. Does one article contain all the details of the other article, plus additional details? Choose "A supersedes B", "B supersedes A", "Both contain unique details", or "Both contain identical details".

Answer with JSON format like this, and write nothing else.
{
  "question 1": {"student is correct": true/false, "correct answer": "..."},
  "question 2": {"student is correct": true/false, "correct answer": "..."},
}'''
    article1 = article1.format(space_index=' A')
    article2 = article2.format(space_index=' B')
    student_analysis = '**Student Analysis**\n'
    if label1 == 'DIFF':
        student_analysis += '1. DIFFERENT. The articles are about different incidents, news, or events.\n'
    elif label1 == 'EVENT':
        student_analysis += '1. SAME EVENT. The articles are reporting the same incident, news, or event. However, they are distinct enough to not be considered as duplicate.\n'
    elif label1 == 'DUP':
        student_analysis += '1. DUPLICATE. The articles are the same, or large sections are the same. One source likely copied the other.\n'
    else:
        raise ValueError()
    timeline_map = {
        'both contain important information': ('Both contain unique details', 'Readers can get a more complete picture by reading both of them, as opposed to reading only one article.'),
        'uncertain': ('Both contain identical details', 'The articles are very similar and have overlapping information. Neither supersedes the other.'),
        'B replaces A': ('B supersedes A', 'Every important detail in Article A is also in Article B, and Article B contains more details. Therefore, readers can save time by reading only Article B and still get the full picture.'),
        'A replaces B': ('A supersedes B', 'Every important detail in Article B is also in Article A, and Article A contains more details. Therefore, readers can save time by reading only Article A and still get the full picture.'),
        'N/A': ('Both contain unique details', 'Article A and Article B each mention some unique details, so the reader can read both to get a more complete picture.'),
    }
    student_analysis += f'2. {timeline_map[timeline1][0]}. {timeline_map[timeline1][1]}'
    user = '\n----------\n'.join([article1, article2, student_analysis])
    
    answer = gpt4_prompt([system, user], temperature=0.01, timeout=15, max_tokens=200, is_json=True,
                         model=model)
    with open('/mnt/nlp/albert/output.txt', 'w') as fout:
        fout.write(system+'\n\n')
        fout.write(user)
    print(answer)
    if not answer:
        return '', ''
    answer = json.loads(answer) if isinstance(answer, str) else answer
    
    def parse_answer(answer, choices):
        for c in choices:
            if answer.startswith(c):
                return c
        for c in choices:
            if c in answer:
                return c
        return ''
    
    if answer['question 1']['student is correct']:
        label = label1
    else:
        label = parse_answer(answer['question 1']['correct answer'], ('DIFF', 'DUP', 'EVENT'))
    if answer['question 2']['student is correct']:
        timeline = timeline_map[timeline1][0]
    else:
        timeline = parse_answer(answer['question 2']['correct answer'], ('A supersedes B', 'B supersedes A', 'Both contain unique details', 'Both contain identical details'))
    map2 = {'A supersedes B': 'A replaces B', 'B supersedes A': 'B replaces A', 'Both contain unique details': 'unique details', '': '', 'Both contain identical details': 'identical details'}
    timeline = map2[timeline]
    if timeline == 'both contain important information' and label == 'DIFF':
        timeline = 'N/A'
    return label, timeline

last_RF_time = 0
def call_prod_RF(master, candidates):
    path = data_path / f'RF_results/{master["_id"]}.json'
    if path.exists():
        with open(path) as fin:
            return json.load(fin)
    
    global last_RF_time
    now = time.time()
    if now - last_RF_time < 1:
        time.sleep(1 - (now - last_RF_time))
    last_RF_time = now
    
    data = {'master': master, 'candidates': candidates}
    url = 'http://doc-clu-dedup-random-forest.k8s.nb-prod.com/document'
    res = requests.post(url, timeout=10, json=data).json()
    
    if res:
        with open(path, 'w') as fout:
            json.dump(res, fout)
    return res


def gpt_label_pairs():
    with open(data_path/'v2_loki_map_disjoint.json') as f:
        lokis = json.load(f)
    to_delete = set()
    with open(data_path/'v2_loki_gpt_label_RF.tsv', 'w') as fout:
        i = 0
        for k, d in tqdm.tqdm(lokis.items()):
            i += 1
            if i % 10 == 0:
                fout.flush()
            try:
                result = gpt_label_RF(d['master'], d['faiss'], d['es'])
                if not result or result[0] == None:
                    to_delete.add(k)
                    print('delete', k)
                else:
                    row = '\t'.join(map(str, result)) + '\n'
                    fout.write(row)
            except Exception as e:
                traceback.print_exc()
    if to_delete:
        for k in to_delete:
            del lokis[k]
        with open(data_path/'v2_loki_map_disjoint2.json', 'w') as f:
            json.dump(lokis, f)
    print_usage()

class Dupidv2Pairs:
    articles_path = Path('/mnt/nlp/albert/articles_cache/more_by_dup_id_v2')
    pairs_path = data_path/'v2_mongo_pairs.tsv'
    label_path = data_path/'v2_mongo_label.tsv'
    days_3 = datetime.timedelta(days=3)
    
    def label(self):
        with (open(self.pairs_path) as fin, open(self.label_path, 'w') as fout):
            for li, line in enumerate(fin):
                try:
                    if li % 2 == 1:
                        continue
                    master, candidate = line.strip().split('\t')
                    master_article, master_doc = one_article(master, return_doc=True)
                    cand_article, cand_doc = one_article(candidate, return_doc=True)
                    if any((not doc.get('stitle')
                            or not doc.get('seg_content')
                            or doc.get('domain') == 'msn.com'
                            ) for doc in (master_doc, cand_doc)):
                        continue
                    
                    result = call_prod_RF(master_doc, [cand_doc])[0]
                    model = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
                    label, timeline = gpt_label_1(master_article, cand_article, model=model)
                    if label is not None:
                        row = (master, candidate,
                                result['label'], result.get('score', ''),
                                label, timeline,
                                master_doc['content_type'], cand_doc['content_type'],
                                master_doc['stitle'], master_doc['seg_content'],
                                cand_doc['stitle'], cand_doc['seg_content'])
                        fout.write('\t'.join(map(str, row)) + '\n')
                        fout.flush()
                except Exception as e:
                    print(e)
    
    def _time_check(self, a, b):
        return abs(datetime.datetime.fromisoformat(a['insert_time']) - datetime.datetime.fromisoformat(b['insert_time'])) < self.days_3
    
    def sample(self):
        gen = self.read_articles(articles_path=self.articles_path, use_pickle=False)
        with open(self.pairs_path, 'w') as f:
            for mapa in gen:
                c = 0
                pairs = []
                docids = set()
                diff_docs = {}
                for dup_id_v2, docs in mapa.items():
                    if len(docs) < 2:
                        continue
                    master, candidate = random.sample(docs, 2)
                    if master['c_word'] == candidate['c_word']:
                        continue
                    if master['doc_id'] in docids or candidate['dup_id_v2'] in docids:
                        continue
                    if not self._time_check(master, candidate):
                        continue
                    roll = random.random()
                    if roll > 0.02:
                        continue
                    elif roll > 0.01:
                        # for DIFF pairs
                        diff_docs[master['doc_id']] = master
                    else:
                        docids.add(master['doc_id'])
                        docids.add(candidate['doc_id'])
                        pairs.append([master['doc_id'], candidate['doc_id']])
                
                diff_list = list(diff_docs.keys())
                random.shuffle(diff_list)
                diff_count = 0
                for i, master in enumerate(diff_list):
                    if diff_count >= 100:
                        break
                    if master not in diff_docs:
                        continue
                    for j in range(i+1, len(diff_list)):
                        candidate = diff_list[j]
                        if candidate not in diff_docs:
                            continue
                        if self._time_check(diff_docs[master], diff_docs[candidate]):
                            pairs.append([master, candidate])
                            diff_count += 1
                            docids.add(master)
                            docids.add(candidate)
                            del diff_docs[master]
                            del diff_docs[candidate]
                            break
                
                features = cache_article_content(list(docids), projection=FIELDS, skip_cache=False)
                features = {doc['_id']: doc for doc in features}
                c = 0
                for master, candidate in pairs:
                    if master not in features or candidate not in features:
                        continue
                    mdoc = features[master]
                    cdoc = features[candidate]
                    if mdoc['stitle'] == cdoc['stitle']:
                        continue
                    if mdoc['seg_content'] == cdoc['seg_content']:
                        continue
                    # if abs(datetime.datetime.fromisoformat(mdoc['insert_time']) - datetime.datetime.fromisoformat(cdoc['insert_time'])) > datetime.datetime.
                    f.write(f'{master}\t{candidate}\n')
                    c += 1
                f.flush()
                print(c)
    
    
    def read_articles(self, articles_path=articles_path, use_pickle=True):
        for fpath in articles_path.glob('*.pkl' if use_pickle else '*.jsonl'):
            px = fpath.stem
            if len(px) == 1:
                c = 0
                articles = defaultdict(list) # dup_id_v2 : list of articles
                if use_pickle:
                    with open(fpath, 'rb') as f:
                        articles = pickle.load(f)
                        c = sum(len(v) for v in articles.values())
                else:
                    with open(fpath) as f:
                        for line in f:
                            if not line.strip():
                                continue
                            d = json.loads(line)
                            for dupid, docs in d.items():
                                articles[dupid] += docs
                                c += len(docs)
                print(f'loaded {c} articles in {len(articles)} dupids')
                yield articles

def do_one():
    model = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    model = 'gpt-4o'
    with (open('/mnt/nlp/albert/input.txt') as fin, open('/mnt/nlp/albert/output.tsv', 'w') as fout):
        for i, line in enumerate(fin):
            if i != 0: continue
            t1, c1, t2, c2 = line.rstrip('\n').split('\t')[:4]
            a1 = one_article(None, doc={'stitle': t1, 'seg_content': c1})
            a2 = one_article(None, doc={'stitle': t2, 'seg_content': c2})
            if False:
                label, timeline = gpt_label_1(a1, a2, model)
                fout.write(f'{label}\t{timeline}\n')
            else:
                label, timeline = gpt_label_1_v2(a1, a2, model)
                fout.write(f'{label}\t{timeline}\n')
            fout.flush()

def merge_pairs():
    pairs = []
    docids = set()
    with open(data_path/'v2_mongo_pairs.tsv') as fin:
        for line in fin:
            d1, d2 = line.rstrip('\n').split('\t')
            if d1 in docids or d2 in docids:
                continue
            docids.add(d1)
            docids.add(d2)
            pairs.append((d1, d2, 'mongo'))
    with open(data_path/'v2_loki_pairs.tsv') as fin:
        for line in fin:
            d1, d2 = line.rstrip('\n').split('\t')
            if d1 in docids or d2 in docids:
                continue
            docids.add(d1)
            docids.add(d2)
            pairs.append((d1, d2, 'loki'))
    random.shuffle(pairs)
    with open(data_path/'docid_pairs.tsv', 'w') as fout:
        for pair in pairs:
            fout.write('\t'.join(pair)+'\n')

class Labeler:
    def gpt_claude_check(self, ifname, ofname):
        model1 = 'gpt-4o'
        model2 = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        modelc = 'gpt-4o'
        with (open(ifname) as fin, open(ofname, 'w') as fout):
            for li, line in enumerate(fin):
                if li % 100 == 0:
                    print(li, datetime.datetime.now().isoformat(' ', 'seconds'))
                line = line.rstrip('\n')
                try:
                    row = line.split('\t')
                    idA, idB = row[:2]
                    articleA, docA = one_article(idA, return_doc=True)
                    articleB, docB = one_article(idB, return_doc=True)
                    row += [docA['stitle'], docA['seg_content'], docB['stitle'], docB['seg_content']]
                    label1, timeline1 = gpt_label_1_v2(articleA, articleB, model1)
                    if label1 is None or timeline1 is None:
                        fout.write('\t'.join(row)+'\n')
                        continue
                    row += [label1, timeline1]
                    label2, timeline2 = gpt_label_1_v2(articleA, articleB, model2)
                    if label2 is None or timeline2 is None:
                        fout.write('\t'.join(row)+'\n')
                        continue
                    row += [label2, timeline2]
                    # if label1 == label2:
                    #     labelc = label1
                    # else:
                    #     labelc = self.label_tiebreak(articleA, articleB, label1, label2, modelc)
                    fout.write('\t'.join(row)+'\n')
                except Exception as e:
                    print(e)
                    fout.write(line+'\n')
                fout.flush()
    
    def do_tiebreak(self, ifname, ofname):
        modelc = 'gpt-4o'
        with (open(ifname) as fin, open(ofname, 'w') as fout):
            for li, line in enumerate(fin):
                if li % 500 == 0:
                    print(li, datetime.datetime.now().isoformat(' ', 'seconds'))
                fout.write(line.rstrip('\n'))
                try:
                    row = line.rstrip('\n').split('\t')
                    id1, id2, src, t1, c1, t2, c2, label1, time1, label2, time2 = row
                    a1 = one_article(None, doc={'stitle': t1, 'seg_content': c1})
                    a2 = one_article(None, doc={'stitle': t2, 'seg_content': c2})
                    lreason, label = self.label_tiebreak(a1, a2, label1, label2, modelc)
                    treason, timeline = self.time_tiebreak(a1, a2, label, time1, time2, modelc)
                    row = [label, timeline, lreason, treason]
                    fout.write('\t'+'\t'.join(row)+'\n')
                except Exception as e:
                    print(e)
                    fout.write('\n')
                fout.flush()
                
    
    label_desc = {
        'DUP': 'The articles are the same or almost the same. So it is not useful to read both articles.',
        'EVENT': 'The articles are reporting the same incident, news, or event. They each contain unique details or are written differently.',
        'DIFF': 'The articles are about different incidents, news, or events.'
    }
    def label_tiebreak(self, article1, article2, label1, label2, model):
        if label1 not in self.label_desc or label2 not in self.label_desc:
            return 'missing', ''
        if label1 == label2:
            return 'match', label1
        if label1 == 'DUP' and label2 == 'DIFF' or label1 == 'DIFF' and label2 == 'DUP':
            system = '''You are an English teacher. Two of your students, Taylor and Avery, disagree with each other about a critical reading question. Who is correct?
It is possible that both of them are wrong, and the articles are reporting the same incident, news, or event, but they each contain unique details or are written differently.
Read the articles carefully. Then, write down your thinking, and make your decision.
Answer with JSON format like this, and write nothing else.
{"thinking": "one sentence", "correct": "Taylor or Avery or neither"}'''
        else:
            system = '''You are an English teacher. Two of your students, Taylor and Avery, disagree with each other about a critical reading question. Who is correct?
Read the articles carefully. Then, write down your thinking, and make your decision.
Answer with JSON format like this, and write nothing else.
{"thinking": "one sentence", "correct": "Taylor or Avery"}'''
        article1 = article1.format(space_index=' A')
        article2 = article2.format(space_index=' B')
        user = f'<ArticleA>{article1}</ArticleA>\n\n<ArticleB>{article2}</ArticleB>\n\n'
        user += f'<Taylor>Taylor wrote, "{self.label_desc[label1]}"</Taylor>\n\n'
        user += f'<Avery>Avery wrote, "{self.label_desc[label2]}"</Avery>'
        answer = gpt4_prompt([system, user], temperature=0.01, timeout=20, max_tokens=100,
                            is_json=True, model=model)
        with open('/mnt/nlp/albert/output.txt', 'w') as fout:
            fout.write(system+'\n\n')
            fout.write(user)
        if not answer:
            return 'no answer', ''
        answer = json.loads(answer) if isinstance(answer, str) else answer
        reason = answer.get('thinking', '')
        person = answer.get('correct')
        if person == 'Taylor':
            return reason, label1
        elif person == 'Avery':
            return reason, label2
        else:
            return reason, 'neither'
    
    time_desc = {
        'A replaces B': 'All of the information in Article B is also in Article A. So after reading Article A, you can skip Article B.',
        'B replaces A': 'All of the information in Article A is also in Article B. So after reading Article B, you can skip Article A.',
        'unique details': 'Each article has some unique details that are not in the other article. You can read both articles.',
        'identical details': 'The articles contain the same details. You only need to read one of them.'
    }
    def time_tiebreak(self, article1, article2, label, time1, time2, model):
        if label == 'DIFF':
            return 'diff', 'diff'
        elif time1 == time2:
            return 'match', time1
        elif label not in self.label_desc:
            return 'no label', ''
        if not time1 and time2:
            return 'one', time2
        elif time1 and not time2:
            return 'one', time1
        if time1 not in self.time_desc or time2 not in self.time_desc:
            return 'missing', ''
        choices = '\n'.join(self.time_desc.values())
        system = f'''You are an English teacher. Two of your students, Taylor and Avery, disagree with each other about a critical reading question. Who is correct? It is also possible that both of them are wrong.

There are four possible answers:
{choices}

Read the articles carefully. Then, write down your thinking, and make your decision.
Answer with JSON format like this, and write nothing else.
{{"thinking": "one sentence", "correct": "Taylor or Avery or neither"}}'''
        article1 = article1.format(space_index=' A')
        article2 = article2.format(space_index=' B')
        user = f'<ArticleA>{article1}</ArticleA>\n\n<ArticleB>{article2}</ArticleB>\n\n'
        user += f'<Taylor>Taylor wrote, "{self.time_desc[time1]}"</Taylor>\n\n'
        user += f'<Avery>Avery wrote, "{self.time_desc[time2]}"</Avery>'
        answer = gpt4_prompt([system, user], temperature=0.01, timeout=20, max_tokens=100,
                            is_json=True, model=model)
        with open('/mnt/nlp/albert/output.txt', 'w') as fout:
            fout.write(system+'\n\n')
            fout.write(user)
        if not answer:
            return 'no answer', ''
        answer = json.loads(answer) if isinstance(answer, str) else answer
        reason = answer.get('thinking', '')
        person = answer.get('correct')
        if person == 'Taylor':
            return reason, time1
        elif person == 'Avery':
            return reason, time2
        else:
            for k, v in self.time_desc.items():
                if reason == v:
                    return reason, k
            return reason, 'neither'
    

def step2_check():
    # model = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    model = 'gpt-4o'
    with (open('/mnt/nlp/albert/input.txt') as fin, open('/mnt/nlp/albert/output.tsv', 'w') as fout):
        for i, line in enumerate(fin):
            # if i != 0: continue
            t1, c1, t2, c2, _, _, label1, time1 = line.rstrip('\n').split('\t')[:8]
            a1 = one_article(None, doc={'stitle': t1, 'seg_content': c1})
            a2 = one_article(None, doc={'stitle': t2, 'seg_content': c2})
            label, timeline = gpt_check_2(a1, a2, label1, time1, model, True)
            fout.write(f'{label}\t{timeline}\n')
            fout.flush()

def do_dupidv2():
    Dupidv2Pairs().sample()

class FileUtils:
    @staticmethod
    def to_pairs_file(ifname, ofname1, ofname2, r):
        one, two = 0, 0
        with (open(ifname) as fin,
                open(ofname1, 'w') as fout1,
                open(ofname2, 'w') as fout2
            ):
            for line in fin:
                row = line.split('\t')
                if len(row) >= 12 and row[11]:
                    if row[11] not in ('DUP', 'EVENT', 'DIFF'):
                        continue
                    out = '\t'.join((row[0], row[1], row[11])) + '\n'
                    if one == 0 or one / (one + two) <= r:
                        fout1.write(out)
                        one += 1
                    else:
                        fout2.write(out)
                        two += 1
    

if __name__ == '__main__':
    st = time.time()
    # Labeler().gpt_claude_check(data_path/'docid_pairs.tsv', data_path/'gpt_claude_label.tsv')
    # Labeler().do_tiebreak(data_path/'gpt_claude_label.tsv', data_path/'tiebroken_label.tsv')
    FileUtils.to_pairs_file(data_path/'tiebroken_label.tsv', data_path/'dedup_train_data_v3/train', data_path/'dedup_train_data_v3/test', 0.9)
    et = time.time()
    print(et-st, 'seconds')