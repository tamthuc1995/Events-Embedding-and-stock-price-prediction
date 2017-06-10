#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:58:54 2017

@author: red-sky
"""

import sys
import json


def findDate(news_body, list_news):
    date = ""
    for ind, new in enumerate(list_news):
        if news_body in new["body"]:
            date = new["time"]
            break
    return date


def extractAllDate(list_events, list_news, choosedInfor=[1, 2, 3, 0, 6]):
    list_result = []
    N = len(list_events)
    i = 0.0
    for event in list_events:
        i += 1
        if i % 1000 == 0:
            print("Done %f percents" % (i/N*100))
        date = [findDate(event[6], list_news)]
        infor = date + [event[i] for i in choosedInfor]
        list_result.append(infor)
    return list_result

if __name__ == "__main__":
    events = open(sys.argv[1], "r").read().strip().splitlines()
    events = [event.split("\t") for event in events
              if len(event.split("\t")) > 5]
    news = json.load(open(sys.argv[2], "r"))
    result = extractAllDate(events, news)

    with open(sys.argv[3], "w") as W:
        for line in result[1:]:
            W.write("\t".join(line)+"\n")
