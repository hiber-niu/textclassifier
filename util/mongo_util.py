# -*-: coding:utf-8 -*-
'''
mongodb相关函数。

date: 2015/12/29 周二
author: hiber.niu@gmail.com
'''

from pymongo import MongoClient

class MongoUtil():
    def __init__(self, database, collection, host='localhost', port=27017):
        client = MongoClient(host, port)
        db = client[database]
        self.coll = db[collection]

    def find(self, query=None):
        '''
        query is certain dict like this:
            {"birthday":{"$lt":new Date("1990/01/01")}}
        '''
        return self.coll.find(query)
