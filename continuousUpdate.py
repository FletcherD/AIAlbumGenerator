# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:46:56 2022

@author: fdost
"""

import time

import getImages
import writeDataset

while True:
    try:
        writeDataset.writeAllTextFiles()
        getImages.getAllImages()
        print('Done')
        time.sleep(600)
    except Exception as e:
        print(e)