#!/usr/bin/python3
# coding=UTF-8

__author__		= "Daaje Meiners"

from examine_sentiment import *

if __name__ == "__main__":
	effectwordnet = import_lexicon()	
	corpus = load_selected_subcorpus("Corpus_6.txt") #, max_lines = 50)
	select_event_sentences(corpus, effectwordnet, export_to = "___Corpus_7.txt")
