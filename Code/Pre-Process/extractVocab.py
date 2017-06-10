#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:52:11 2017

@author: red-sky
"""
import sys
import json
import numpy as np


def updateDict(words, dictUp):
    # update word dictionary with given "words" and the dict "dictUp"
    for w in words:
        if w in dictUp:
            dictUp[w] += 1
        else:
            dictUp[w] = 0
    return dictUp


def extractVocab(eventsFile, fromIndex=0, toIndex="END"):
    # from Events file, extract infor about words and create a mapping
    vocab = dict()
    with open(eventsFile, "r") as file:
        list_events = file.read().strip().splitlines()
        if toIndex == -1:
            list_events = list_events[fromIndex:]
        else:
            list_events = sorted(set(list_events[fromIndex:toIndex]))
    for i, event in enumerate(list_events):
        if event[0] != "\t":
            index = i
            break
    list_events = list_events[index:]
    for event in list_events:
        event = event.split("\t")
        words = event[1].split(" ") + \
            event[2].split(" ") + \
            event[3].split(" ")
        vocab = updateDict(words, vocab)
    vocab_words = vocab.keys()
    support_words = ["NOISEWORDS"]
    vocab_words = support_words + \
        sorted(vocab_words, key=lambda x: vocab[x], reverse=True)
    IndexWords = range(len(vocab_words))
    Count = ["NOISEWORDS"] + [vocab[w] for w in vocab_words[1:]]
    result = [dict(zip(vocab_words, Count)),
              dict(zip(IndexWords, vocab_words)),
              dict(zip(vocab_words, IndexWords))]
    return result, list_events


def convertEvent(eventsFile, vocabMapping, countMin=5):
    # convert all Events to index for training
    wordCount, _, word2index = vocabMapping
    Events = []
    with open(eventsFile, "r") as file:
        list_events = file.read().strip().splitlines()

    for event in list_events:
        event = event.split("\t")
        list_obj = [event[1].split(" "),
                    event[2].split(" "),
                    event[3].split(" ")]

        # Covert only words that appear more than countMin
        wordsIndexed = []
        for obj in list_obj:
            objIndex = []
            for w in obj:
                if wordCount[w] >= countMin:
                    objIndex.append(word2index[w])
                else:
                    objIndex.append(0)
            wordsIndexed.append(objIndex)
        Events.append(wordsIndexed)
    return Events


if __name__ == "__main__":
    # in
    EventPath = sys.argv[1]
    fromIndex = int(sys.argv[3])-1
    toIndex = int(sys.argv[4])-1
    minCountWord = int(sys.argv[5])
    # out
    EventNewPath = sys.argv[2] + "Events_for_training.txt"
    VocabPath = sys.argv[2] + "Vocab_in_events_for_training.json"
    IndexdEventPath = sys.argv[2] + "IndexedEvents_for_training.npy"

    vocabMapping, EventNew = extractVocab(EventPath, fromIndex, toIndex)
    with open(VocabPath, "w") as W:
        json.dump(vocabMapping, W, indent=2)

    with open(EventNewPath, "w") as W:
        W.write("\n".join(EventNew))

    indexed_events = convertEvent(EventNewPath, vocabMapping, minCountWord)
    np.save(arr=np.array(indexed_events), file=IndexdEventPath)
