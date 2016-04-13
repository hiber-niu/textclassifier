# -*-: coding:utf-8 -*-
'''
mysql相关函数。

date: 2015/12/28 周一
author: hiber_niu@163.com
'''

import MySQLdb


class MysqlUtil():
    def __init__(self, host, db, user, pw, charset='utf8'):
        self.host = host
        self.db = db
        self.user = user
        self.passwd = pw
        self.charset = charset

        database = MySQLdb.connect(host=self.host, user=self.user, passwd=self.passwd,
                                   db=self.db, charset=self.charset)
        cursor = database.cursor()
        # cursor.execute('SET NAMES "utf8";')
        self.cursor = cursor

    def get_query_results(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def insert_many(self, query, rows):
        """
        query is certain string like this:
        '''INSERT INTO mrdt (docno, title, source, publishdate, ftext,
        department, author, sourcetitle, sourceurl) VALUES (%s, %s, %s, %s,
        %s, %s, %s, %s, %s)''', rows)
        """
        return self.cursor.executemany(query, rows)
