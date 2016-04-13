# -*- coding: utf-8 -*-
'''
Chinese text multiple classification.
Using space data.

    1. classify space and other chinese texts.
    2. classify detail categories using space texts.

date: 2016/03/23 周三
author: hiber_niu@163.com
'''

from util.mongo_util import MongoUtil
from util.mysql_util import MysqlUtil

import html2text
import jieba
from time import time
import codecs
import numpy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier

import cPickle


def dump_contents():
    corpus, categories = get_content_category()

    with open('space_corpus_dump.dat', 'wb') as fid:
        cPickle.dump(corpus, fid)
    with open('space_categories_dump.dat', 'wb') as fid:
        cPickle.dump(categories, fid)
    return


def cn_tokenize(text):
    seg_list = jieba.cut(text)
    seg_list = [item for item in seg_list if len(item) > 1 and not item.isdigit()]
    seg_list = [item for item in seg_list if '.' not in item]
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

    results = MongoUtil('cn_classify', 'dsti', '192.168.200.10').find()
    for row in results:
        if len(row['text']) <= 0:
            continue
        contents.append(row['text'])
        category = row['source']
        if category == u'航天工业':
            category = 1
        else:
            category = 0
        categories.append(category)

    return contents, categories


def get_detail_content_category():
    contents = []
    categories = []

    results = MongoUtil('cn_classify', 'space', '192.168.200.10').find()
    for row in results:
        contents.append(row['content'])
        category = row['category']
        category = category[1:]
        if category == u'战略综合':
            category = 0
        elif category == u'进入空间':
            category = 1
        elif category == u'利用空间':
            category = 2
        elif category == u'控制空间':
            category = 3
        elif category == u'载人航天':
            category = 4
        else:  #'前沿技术':
            category = 5
        categories.append(category)

    return contents, categories


def get_test_mrdt():
    cur = MysqlUtil('192.168.200.10', 'bdscontent', 'bdsdata', '357135')
    query = 'SELECT * FROM mrdt'
    results = cur.get_query_results(query)
    contents = []
    titles = []

    for row in results:
        content = get_text(row[4].decode('utf-8'))
        contents.append(content)
        titles.append(row[1])

    return contents, titles


def get_text(html):
    h = html2text.HTML2Text()
    return h.handle(html)


def train_space_classifier():
    with open('space_corpus_dump.dat', 'rb') as fid:
        corpus = cPickle.load(fid)
    with open('space_categories_dump.dat', 'rb') as fid:
        categories = cPickle.load(fid)

    vectorizer = TfidfVectorizer(max_df=1.0, max_features=1000, min_df=1,
                                 stop_words=get_cn_stopwords(),
                                 encoding='utf-8', decode_error='ignore',
                                 analyzer='word', tokenizer=cn_tokenize)

    X = vectorizer.fit_transform(corpus)
    with open('space_module_vector.pkl', 'wb') as fid:
        cPickle.dump(vectorizer, fid)
    y = categories

    clf = svm.LinearSVC()

    print('_' * 80)
    t0 = time()
    clf.fit(X, y)

    feature_names = vectorizer.get_feature_names()
    with codecs.open('space_features.txt', 'w', 'utf-8') as fid:
        for i in range(clf.coef_.shape[0]):
            coefs_with_fns = sorted(zip(clf.coef_[i], feature_names), reverse=True)
            for (coef_1, fn_1) in coefs_with_fns:
                fid.write('%s\n' % (fn_1))

    with open('space_module_dump.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    kfold = cross_validation.KFold(len(categories), n_folds=10)
    accuracy = cross_validation.cross_val_score(clf, X, categories, cv=kfold)
    print("average accuracy:  %0.3f" % accuracy.mean())


def train_space_multiple_classifier():
    corpus, categories = get_detail_content_category()

    vectorizer = TfidfVectorizer(max_df=1.0, max_features=10000, min_df=1,
                                 stop_words=get_cn_stopwords(),
                                 encoding='utf-8', decode_error='ignore',
                                 analyzer='word', tokenizer=cn_tokenize)

    X = vectorizer.fit_transform(corpus)
    with open('space_multiple_module_vector.pkl', 'wb') as fid:
        cPickle.dump(vectorizer, fid)
    y = categories

    clf = svm.LinearSVC()
    ch2 = SelectKBest(chi2, k=800)
    selected_X = ch2.fit_transform(X, y)

    print('_' * 80)
    t0 = time()
    clf.fit(selected_X, y)

    '''
    # get chi2 selected feature names).
    top_ranked_features = sorted(enumerate(ch2.scores_),key=lambda
                                 x:x[1], reverse=True)[:3000]
    top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
    with codecs.open('selected_space_features.txt', 'w', 'utf-8') as fid:
        for feature_pvalue in zip(numpy.asarray(vectorizer.get_feature_names())[top_ranked_features_indices], ch2.pvalues_[top_ranked_features_indices]):
            # fid.write('%s  #  %0.5f' % (feature_pvalue[0], feature_pvalue[1]))
            fid.write('%s' % (feature_pvalue[0]))
            fid.write('\n')

    '''
    '''
    # get feature names of each category.
    feature_names = vectorizer.get_feature_names()
    with codecs.open('space_categories_features.txt', 'w', 'utf-8') as fid:
        for i in range(clf.coef_.shape[0]):
            fid.write('='*80)
            coefs_with_fns = sorted(zip(clf.coef_[i], feature_names), reverse=True)
            for (coef_1, fn_1) in coefs_with_fns:
                fid.write('\t%.4f\t%-15s\n' % (coef_1, fn_1))
    '''


    with open('space_multiple_module_dump.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    kfold = cross_validation.KFold(len(categories), n_folds=10)
    accuracy = cross_validation.cross_val_score(clf, selected_X, categories, cv=kfold)
    print("average accuracy:  %0.3f" % accuracy.mean())


def train_dl_space_multiple_classifier():
    corpus, categories = get_detail_content_category()

    vectorizer = TfidfVectorizer(max_df=1.0, max_features=10000, min_df=1,
                                 stop_words=get_cn_stopwords(),
                                 encoding='utf-8', decode_error='ignore',
                                 analyzer='word', tokenizer=cn_tokenize)

    X = vectorizer.fit_transform(corpus)
    with open('space_multiple_module_vector.pkl', 'wb') as fid:
        cPickle.dump(vectorizer, fid)
    y = categories

    clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5,
                        hidden_layer_sizes=(15,), random_state=1)
    print('_' * 80)
    t0 = time()
    clf.fit(X, y)

    '''
    # get feature names of each category.
    feature_names = vectorizer.get_feature_names()
    with codecs.open('space_dl_categories_features.txt', 'w', 'utf-8') as fid:
        for i in range(clf.coefs_.shape[0]):
            fid.write('='*80)
            coefs_with_fns = sorted(zip(clf.coefs_[i], feature_names), reverse=True)
            for (coef_1, fn_1) in coefs_with_fns:
                fid.write('\t%.4f\t%-15s\n' % (coef_1, fn_1))
    '''


    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    kfold = cross_validation.KFold(len(categories), n_folds=10)
    accuracy = cross_validation.cross_val_score(clf, X, categories, cv=kfold)
    print("average accuracy:  %0.3f" % accuracy.mean())  # 0.638


def test_module():
    corpus = []
    titles = []

    corpus, titles = get_test_mrdt()

    with open('space_module_vector.pkl', 'rb') as fid:
        vectorizer = cPickle.load(fid)
    X = vectorizer.transform(corpus)
    with open('space_module_dump.pkl', 'rb') as fid:
        clf = cPickle.load(fid)
    pred = clf.predict(X)

    with open('space_test.txt', 'w') as fid:
        fid.write('showing test results\n')
        fid.write('test  ||  title\n')
        for index, result in enumerate(pred):
            fid.write('  %d  ||  %s \n' % (result, titles[index].encode('utf-8')))


if __name__ == '__main__':
    # dump_contents() # only need run once
    # train_space_classifier() # train space classifier
    # test_module()
    train_space_multiple_classifier()
    # train_dl_space_multiple_classifier()
