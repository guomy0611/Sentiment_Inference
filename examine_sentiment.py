#!/usr/bin/python3
# coding=UTF-8

from sst_parser import *
from sentiment_lexicon import *
from LBP import *

__author__		= "Daaje Meiners"
		
class SentimentEnv:
	"""Wrapper class. Stores an SST_Corpus object and a lexicon list."""
	def __init__(self, corpus = "stanfordSentimentTreebank/", gfbf_lexicon = "effectwordnet/goldStandard.tff"):
		if type(corpus) is str:
			corpus = SST_Corpus(corpus)
		if type(gfbf_lexicon) is str:
			gfbf_lexicon = import_lexicon(gfbf_lexicon)
		
		if type(corpus) != SST_Corpus:
			raise TypeError("Invalid corpus")
		if type(gfbf_lexicon) != list:
			raise TypeError("Invalid lexicon")
		self.corpus = corpus
		self.lex = gfbf_lexicon	
		
	def get_phrase_sentiments(self, sentence_obj):
		""" Takes a SST_Sentence object and its associated datasets, computes subj + obj sentiments with graphs.
		
		Arguments:
		sentence_obj -- An SST_Sentence object.
		corpus -- An SST_Corpus object.
		gfbf_lexicon -- A list of LexiconEntry objects.
		also_return_graph -- If true, returns a tuple of (sentiment data, graph object)."""
		event_data = sentence_obj.get_event_subtrees(write=False)
		# print(event_data)
		event_graphs = self.make_event_graphs(event_data)
		# FIXME: kludge
		if not(event_graphs):
			return False
		out_sent = tuple([g.get_senti() for g in event_graphs])
		# memory management
		for g in event_graphs:
			del(g)
		return out_sent
		
	def make_event_graphs(self, event_subtrees):
		"""Takes a list of [subj, pred, obj, event] subtrees, generates Subj, Obj node objects from them."""
		# An issue: if we just use the whole sentence/event phrase, it doesn't necessarily match the concat of the event subtrees...
		# but the SST sentiment data has no entry for the <Subject Predicate Object> "phrases".
		# So we can only compare a whole-sentence sentiment with an event sentiment. this will worsen our accuracy
		subj_senti = self.corpus.lookup_sentiment(event_subtrees[0])
		obj_senti = self.corpus.lookup_sentiment(event_subtrees[2])
		# FIXME: kludge to skip broken lines
		if len([e for e in event_subtrees if not(e)]):
			print("Missing info for graph from", [tree_to_string(phrase) if phrase else phrase for phrase in event_subtrees])
			return False
		phrase_strings = [tree_to_string(phrase) for phrase in event_subtrees]
		pred_gfbf = get_gfbf(phrase_strings[1], self.lex, True)
		# new graph format from LBP.py: dict of string to ( [(neighbour, edge), ...], (es_pos, es_neg) )
		"""graph_input = {phrase_strings[0]:([(phrase_strings[0], pred_gfbf)], (subj_senti, 1-subj_senti)),
				phrase_strings[2]:([(phrase_strings[0], pred_gfbf)], (obj_senti, 1-obj_senti))}"""
		# print("sentiments:", phrase_strings[0], subj_senti, phrase_strings[2], obj_senti)
		graph_unknown_subj = {phrase_strings[0]:([(phrase_strings[2], pred_gfbf)], (0.5, 0.5)),
							  phrase_strings[2]:([(phrase_strings[0], pred_gfbf)], (obj_senti, 1-obj_senti))}
		graph_unknown_obj =  {phrase_strings[0]:([(phrase_strings[2], pred_gfbf)], (subj_senti, 1-subj_senti)),
							  phrase_strings[2]:([(phrase_strings[0], pred_gfbf)], (0.5, 0.5))}
		# print(graph_input)
		return (Node(phrase_strings[0], graph_unknown_subj), Node(phrase_strings[2], graph_unknown_obj) )
		
	def select_event_sentences(self, export_to):
		return select_event_sentences(self.corpus, self.lex, export_to)

	# Old:
	"""def compute_sentiments(self, export_to = "sentiments_corpus7.txt"):
		outdoc = open(export_to, "w")
		outdoc.write("# index, predicate, sys senti, gold senti, sys label correctness, detected event, beginning of sentence\n")	
		for sentence in corpus.treebank:
			senti_and_graph = self.get_phrase_sentiments(sentence, True)
			# convert tuple of (pos/neg, intensity) to SST-like value
			comparable_senti_val = senti_and_graph[0][1]
			if senti_and_graph[0][0] == "negative":
				comparable_senti_val = 1 - comparable_senti_val
			sys_subj_senti = 
			gold_senti_val = self.corpus.lookup_sentiment(sentence.sentence_tree)		
			label_correctness = (comparable_senti_val > 0.5) == (gold_senti_val > 0.5)
			
			outdoc.write( str(sentence.index) + " \t " + tree_to_string(sentence.get_SVO()[1]) + "\t" + str(comparable_senti_val) + "\t" + str(gold_senti_val) + "\t" + str(label_correctness) + "\t" + senti_and_graph[1].event + "\t" + cap_string(tree_to_string(sentence.sentence_tree), 25) + "\n" )
			print(tree_to_string(sentence.sentence_tree), comparable_senti_val, " --- ", senti_and_graph[1].event)
		outdoc.close()"""

	def compute_sentiments(self, export_to = "sentiments_corpus7.txt"):
		outdoc = open(export_to, "w")
		outdoc.write("# index, predicate, subj sys senti, subj gold senti, subj label correctness, obj sys, obj gold, obj correctness, predicate gfbf, detected event, beginning of sentence\n")	
		for sentence in corpus.treebank:
			sys_sents = self.get_phrase_sentiments(sentence)
			# FIXME: kludge
			if not(sys_sents):
				print("Sentence didn't parse: ", tree_to_string(sentence.sentence_tree))
				continue
			"""# convert tuple of (pos/neg, intensity) to SST-like value
			comparable_senti_val = senti_and_graph[0][1]
			if senti_and_graph[0][0] == "negative":
				comparable_senti_val = 1 - comparable_senti_val"""
			gold_subj_sent = self.corpus.lookup_sentiment(sentence.subject_st)
			gold_obj_sent = self.corpus.lookup_sentiment(sentence.object_st)
			subj_correctness = ( (sys_sents[0] == "positive") == (gold_subj_sent > 0.5) )
			obj_correctness =  ( (sys_sents[1] == "positive") == (gold_obj_sent > 0.5) )
			gfbf = get_gfbf(tree_to_string(sentence.get_event_subtrees()[1]), self.lex, True)
			if not(gfbf):
				print("Unknown predicate in sentence ", sentence.index, cap_string(tree_to_string(sentence.sentence_tree), 25))
				continue
			line =  str(sentence.index) + " \t " + tree_to_string(sentence.get_SVO()[1])
			line += "\t" + str(sys_sents[0]) + "\t" + str(gold_subj_sent) + "\t" + str(subj_correctness)
			line += "\t" + str(sys_sents[1]) + "\t" + str(gold_obj_sent) + "\t" + str(obj_correctness)
			line += "\t" + gfbf
			line += "\t" + " ".join([tree_to_string(p) for p in sentence.get_event_subtrees()]) + "\t" + cap_string(tree_to_string(sentence.sentence_tree), 25) + "\n" 
			outdoc.write( line )
			print(tree_to_string(sentence.get_SVO()[1]), sys_sents, "---", sentence.index, tree_to_string(sentence.get_SVO()[3]))
		outdoc.close()

# For old LBP script:
'''def get_phrase_sentiment(sentence_obj, corpus, gfbf_lexicon, also_return_graph = False):
	""" Takes a SST_Sentence object and its associated datasets, computes sentiment with a graph.
	
	Arguments:
	sentence_obj -- An SST_Sentence object.
	corpus -- An SST_Corpus object.
	gfbf_lexicon -- A list of LexiconEntry objects.
	also_return_graph -- If true, returns a tuple of (sentiment data, graph object)."""
	event_data = sentence_obj.get_event_subtrees(write=False)
	# print(event_data)
	event_graph = make_event_graph(event_data, corpus, gfbf_lexicon)
	if also_return_graph:
		return (event_graph.get_senti(event_graph.event), event_graph)
	else:
		return event_graph.get_senti(event_graph.event)'''

# For old LBP script:
'''def make_event_graph(event_subtrees, corpus, gfbf_lexicon):
	"""Takes a list of [subj, pred, obj, event] subtrees, generates a Graph object from them."""
	# An issue: if we just use the whole sentence/event phrase, it doesn't necessarily match the concat of the event subtrees...
	# but the SST sentiment data has no entry for the <Subject Predicate Object> "phrases".
	# So we can only compare a whole-sentence sentiment with an event sentiment. this will worsen our accuracy
	subj_senti = corpus.lookup_sentiment(event_subtrees[0])
	obj_senti = corpus.lookup_sentiment(event_subtrees[2])
	phrase_strings = [tree_to_string(phrase) for phrase in event_subtrees]
	# graph format from LBP.py: tuples of (string, (pos sent, neg sent) ) for subj, pred, obj, whole event
	graph_input = [ (phrase_strings[0], (subj_senti, 1-subj_senti) ), 
			(phrase_strings[1], get_gfbf(phrase_strings[1], gfbf_lexicon, True) ),
			( phrase_strings[2],  (obj_senti, 1-obj_senti) ), 
			(" ".join(phrase_strings), (0.5, 0.5) )       ]
	# print(graph_input)
	return Graph( graph_input )'''

def select_event_sentences(from_corpus, with_lexicon, export_to):
	"""Returns a pruned list of sentences from a corpus with recognizable predicates from a lexicon"""
	# selection of sentences with a recognizable SVO structure
	sub_corpus = [sentence for sentence in from_corpus.treebank if sentence.get_SVO()]
	# print predicates
	print([tree_to_string(s.svo_data[1]) for s in sub_corpus])
	# selection of sentences where the predicate or its stemmed form is in the gfbf lexicon
	sub_corpus = [sentence for sentence in sub_corpus if tree_to_string(sentence.svo_data[1]) in get_known_words(with_lexicon) or stem_word(tree_to_string(sentence.svo_data[1])) in get_known_words(with_lexicon)]
	print([tree_to_string(s.svo_data[1]) for s in sub_corpus])
	if not export_to:
		return sub_corpus
	outfile = open(export_to, "w")
	for s in sub_corpus:
		w = str(s.index) + "\t" + tree_to_string(s.sentence_tree) + "\t" + str(get_gfbf(s.svo_data[1], with_lexicon, True)) + "\n"
		print(w)
		outfile.write(w)
	outfile.close()
	return sub_corpus



# i wanted explicit numbers to compare with sst, but i figured out too late that this needs normalization
# normalization:
# this presumes sentiment values between 0 to 1.
# intuition: >.5 should increase the val and <0.5 should decrease it
# but i didn't think through the math and i think this can leave the boundaries so it doesn't work this way
'''def __get_explicit_senti(self):
	senti_pos=self.es[0]
	senti_neg=self.es[1]


	for neib,v in self.connections:
		m_pos = self.calc_message(neib)[0]
		m_neg = self.calc_message(neib)[1]

		senti_pos*=m_pos
		senti_neg*=m_neg

	# print("senti and msg:", self.es[0], m_pos)
	return senti_pos * 2
	
# appended extra method (is that unpythonic?)
Node.get_explicit_senti = __get_explicit_senti'''





def cap_string(s, l):
	"""Returns string s, cropped and ended by "..." if longer than legnth l."""
	# Credit to http://stackoverflow.com/questions/11602386/python-function-for-capping-a-string-to-a-maximum-length.
	# i checked if there was a pythonic way to do this, and decided i might as well use what's there
	return s if len(s)<=l else s[0:l-3]+'...'

	# Retrieve object sentiment from SST.
	# Disambiguate predicate sense (in sentiment_lexicon/get_gfbf).
	# Retrieve predicate GFBF status.
	# Apply inference rules.
	# Compare resulting sentiment to whole-phrase sentiment 
	
	
	# Your suggestion for explicit phrase sentiment sounds good, but the SST already comes with sentiment data for each phrase.
	# using your algorithm could improve results, but i'll use the pre-existing data for now so we can try out LBP...

if __name__ == "__main__":
	corpus = load_selected_subcorpus("Corpus_7.txt")
	env_wrapper = SentimentEnv(corpus, import_lexicon())
	"""corpus = load_selected_subcorpus("Corpus_5.txt", max_lines = 10)
	select_event_sentences(corpus, effectwordnet, export_to = "___Corpus_7.txt")"""
	env_wrapper.compute_sentiments()
