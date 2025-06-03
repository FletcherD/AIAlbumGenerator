import sqlite3
import json
import time

DB_FILE_PATH = 'DiscogsDataset.db'
IMAGE_DIR_PATH = '/media/media/Storage/Media/DiscogsImageDataset'
IMAGE_DIR_PATH_UNPROCESSED = '/media/media/Storage/Media/DiscogsImageDatasetRaw'
DATASET_CSV_PATH = 'albumTextToImageDataset.csv'

con = sqlite3.connect(DB_FILE_PATH)
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

def deleteRelease(idNum) :
    r = cur.execute("DELETE FROM releases WHERE id=?", [idNum])


def removeBadReleases():
    for idNum in getAllReleaseIds():
        try:
            getRelease(idNum)

        except Exception as e:
            print(idNum)
            deleteRelease(idNum)
