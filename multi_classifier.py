# -*- coding: utf-8 -*-
'''
Chinese text multiple classification.
Using DSTI data.

date: 2016/03/18 周五
author: hiber.niu@gmai.com
'''
from util.mongo_util import MongoUtil
import html2text
import jieba
from time import time
import codecs

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn import cross_validation

import cPickle


def dump_contents():
    corpus, categories = get_content_category()

    with open('multi_corpus_dump.dat', 'wb') as fid:
        cPickle.dump(corpus, fid)
    with open('multi_categories_dump.dat', 'wb') as fid:
        cPickle.dump(categories, fid)
    return


def cn_tokenize(text):
    seg_list = jieba.cut(text)
    seg_list = [item for item in seg_list if len(item)>1 and not item.isdigit()]
    return seg_list


def get_cn_stopwords():
    filename = 'cn_stopwords.txt'
    stopwords = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        stopwords.extend(f.read().splitlines())

    return stopwords


def get_content_category():
    contents = []
    categories = []

    results = MongoUtil('cn_classify', 'dsti', '192.168.*.*').find()
    for row in results:
        if len(row['text']) <= 0:
            continue
        contents.append(row['text'])
        category = row['source']
        if category == u'核工业':
            category = 0
        elif category == u'航天工业':
            category = 1
        elif category == u'航空工业':
            category = 2
        elif category == u'船舶工业':
            category = 3
        elif category == u'兵器工业':
            category = 4
        else:  # 电子工业
            category = 5
        categories.append(category)

    return contents, categories

'''
def get_content_category():
    # get non-space articles
    cur = MysqlUtil('*.*.*.*', 'crawl', 'user', '****')
    query = 'SELECT * FROM base3 WHERE site_name="DSTI"'
    results = cur.get_query_results(query)
    contents = []
    categories = []
    for row in results:
        # read article
        content = get_text(row[11])
        content = content.split('\n')[5:-4]
        content = '\n'.join(content)
        contents.append(content)

        # read category
        category = get_text(row[4])
        category = category.strip('\n')
        if category == u'核工业':
            category = 0
        elif category == u'航天工业':
            category = 1
        elif category == u'航空工业':
            category = 2
        elif category == u'船舶工业':
            category = 3
        elif category == u'兵器工业':
            category = 4
        else: # 电子工业
            category = 5
        categories.append(category)

    return contents, categories
'''

def get_text(html):
    h = html2text.HTML2Text()
    return h.handle(html)


def train_multi_classifier():
    with open('multi_corpus_dump.dat', 'rb') as fid:
        corpus = cPickle.load(fid)
    with open('multi_categories_dump.dat', 'rb') as fid:
        categories = cPickle.load(fid)

    vectorizer = TfidfVectorizer(max_df=1.0, max_features=1000, min_df=1,
                                 stop_words=get_cn_stopwords(),
                                 encoding='utf-8', decode_error='ignore',
                                 analyzer='word', tokenizer=cn_tokenize)

    X = vectorizer.fit_transform(corpus)
    with open('multi_module_vector.pkl', 'wb') as fid:
        cPickle.dump(vectorizer, fid)
    Y = categories

    clf = Pipeline([('classification', svm.LinearSVC())])

    print('_' * 80)
    t0 = time()
    clf.fit(X, Y)

    with open('multi_module_dump.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    kfold = cross_validation.KFold(len(categories), n_folds=10)
    accuracy = cross_validation.cross_val_score(clf, X, categories, cv=kfold)
    print("average accuracy:  %0.3f" % accuracy.mean())


def test_module():
    corpus = []
    categories = []
    titles = []

    results = MongoUtil('cn_classify', 'test', '*.*.*.*').find()
    for row in results:
        if len(row['text']) <= 0:
            continue
        corpus.append(row['text'])
        category = row['source']
        titles.append(row['title'])
        if category == u'核工业':
            category = 0
        elif category == u'航天工业':
            category = 1
        elif category == u'航空工业':
            category = 2
        elif category == u'船舶工业':
            category = 3
        elif category == u'兵器工业':
            category = 4
        else:  # 电子工业
            category = 5
        categories.append(category)

    with open('multi_module_vector.pkl', 'rb') as fid:
        vectorizer = cPickle.load(fid)
    X = vectorizer.transform(corpus)
    with open('multi_module_dump.pkl', 'rb') as fid:
        clf = cPickle.load(fid)
    pred = clf.predict(X)

    print('='*80)
    print('showing test results')
    print('test  ||  orig  ||  title')
    for index, result in enumerate(pred):
        print('   %d  ||  %d  || %s' %(result, categories[index],
                                       titles[index]))

    accuracy = metrics.accuracy_score(categories, pred)
    print("mannul accuracy:  %0.3f" % accuracy)


if __name__ == '__main__':
    # dump_contents() # only need run once
    train_multi_classifier() # train multiple classify module
    test_module()
