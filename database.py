# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 15:06:40 2022

@author: fdost
"""

import sqlite3
import json
import time

con = sqlite3.connect('dataset.db')
cur = con.cursor()

def getRelease(idNum):
    r = cur.execute("SELECT jsonData FROM releases WHERE id=?", [idNum])
    r = r.fetchone()
    if r is None:
        return None
    else:
        r = r[0]
        r = json.loads(r)
        return r
    
def getAllReleaseIds():
    r = cur.execute("SELECT id FROM releases")
    r = [k[0] for k in r.fetchall()]
    return r

def addRelease(idNum, data):
    cur.execute("INSERT INTO releases VALUES (?, ?)", (idNum, json.dumps(data)))
    while True:
        try:
            con.commit()
            break
        except sqlite3.OperationalError as e:
            time.sleep(1)

def removeBadReleases():
    for idNum in getAllReleaseIds():
        try:
            getRelease(idNum)
        except Exception as e:
            print(idNum)
            r = cur.execute("DELETE FROM releases WHERE id=?", [idNum])
