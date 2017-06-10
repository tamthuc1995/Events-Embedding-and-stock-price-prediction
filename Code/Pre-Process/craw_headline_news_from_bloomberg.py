#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:57:57 2017

@author: red-sky
"""

import bs4
import json
import sys
import urllib.request as urlreq
from bs4 import BeautifulSoup

BLOOMBERG_params = {
    "sort_by_newest": "time:desc",
    "sort_by_oldest": "time:asc",
    "source_from_bloomberg": "sites=bview",
    "end_time": "2017-03-12T15:20:16.240Z"
}

DATA_TO_EXTRACT = {
    "query_list_news": ["div", {"class": "search-result-story__container"}],
    "query_headline": ["h1", {"class": "search-result-story__headline"}],
    "query_time_published": ["time", {"class": "published-at"}],
    "query_body": ["div", {"class": "search-result-story__body"}]
}


def parser_url(query_string, page,
               sort_by="sort_by_oldest",
               source="source_from_bloomberg"):
    url = "https://www.bloomberg.com/"
    # add search query
    url = url + "search?query=" + query_string + "&"
    # add sort by
    url = url + "sort=" + BLOOMBERG_params[sort_by] + "&"
    # add time to query -- use present time
    url = url + "sites=" + BLOOMBERG_params[source] + "&"
    # add page number
    url = url + "page=" + str(page)
    return url


def get_rid_off_key(list_contents):
    body_string = ""
    for substring in list_contents:
        if (type(substring) == bs4.element.Tag):
            # join all body string and
            # eliminate highlight query string key
            body_string += substring.string
        else:
            if (type(substring.string) == bs4.element.NavigableString):
                body_string += substring.string
    return(body_string)


def extract_from_url(url):
    try:
        with urlreq.urlopen(url) as response:
            html_of_page = response.read()
            soup_object = BeautifulSoup(html_of_page, "lxml")
        # Extract list of news in soup object
        param_to_find = DATA_TO_EXTRACT["query_list_news"]
        list_of_news = soup_object.find_all(param_to_find[0],
                                            attrs=param_to_find[1])
        if (len(list_of_news) == 0):
            return None
        # create list result extracted
        result = []
        for block_new in list_of_news:
            # extract time from block
            param_to_find = DATA_TO_EXTRACT["query_time_published"]
            time = block_new.find_all(param_to_find[0],
                                      attrs=param_to_find[1])
            time = time[0]["datetime"]

            # extract new headline
            param_to_find = DATA_TO_EXTRACT["query_headline"]
            headline = block_new.find_all(param_to_find[0],
                                          attrs=param_to_find[1])
            headline = get_rid_off_key(headline[0].a.contents)

            # extract new body list if string
            param_to_find = DATA_TO_EXTRACT["query_body"]
            body = block_new.find_all(param_to_find[0],
                                      attrs=param_to_find[1])

            body_string = get_rid_off_key(body[0].contents)
            extracted_from_block = {"time": time,
                                    "headline": headline,
                                    "body": body_string}
            # for debug :
            # print("\t".join(extracted_from_block))
            if len(body_string) >= 5:
                result.append(extracted_from_block)
    except Exception as inst:
        print("Something whenwrong :)", inst)
        print("ULR: ", url)
        result = []
    return(result)


def Query(key, max_page=5000):
    # Init page and looping until return None
    page = 1
    result = "not None"
    all_result_query = []
    error = 0
    while True and page < max_page:
        print("Colected: %d articles" % len(all_result_query))
        new_url = parser_url(key, page)
        result = extract_from_url(new_url)
        if len(result) > 0 or error > 10:
            page += 1
            error = 0
        else:
            error += 1

        if result is not None:
            all_result_query += result
        else:
            break
    return(all_result_query)


if __name__ == "__main__":
    print("Begin query information about: ", sys.argv[1])
    print("Then will save result in: ", sys.argv[2])

    News = Query(sys.argv[1], int(sys.argv[4]))
    file_name1 = sys.argv[2]

    with open(file_name1, "w") as W:
        json.dump(News, W, indent=1)

    file_name2 = sys.argv[3]
    with open(file_name2, "w") as W:
        W.write("\n".join([new["body"] for new in News]))
