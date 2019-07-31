#!/usr/bin/python3
from sst_parser import *

__author__		= "Daaje Meiners"

# swn = nltk.corpus.sentiwordnet # alias for convenience
from random import choice
# nltk imported by sst_handler

class LexiconEntry:
	"""LexiconEntry: Stores an entry from the Deng/Wiebe sentiment lexicon.
	
	Variables:
	entry_id --
	effect --
	synset --
	definition --
		
	Methods:
	from_string ---
	quantify_sentiment---"""
	def __init__(self, entry_id, effect, synset, definition):
		"""Constructs 
		
		Keyword arguments:
		entry_id
		effect
		synset
		definition"""
		self.entry_id = entry_id
		self.effect = effect
		if type(synset) == str:
			synset = synset.split(",")
		self.synset = synset
		self.definition = definition
		self.known_words = None
		
	@classmethod
	def from_string(cls, input_string):
		"""Class method. Creates a LexiconEntry from a lexicon string."""
		spl = input_string.rstrip().split("\t")
		# empty lines are ok
		if len(spl) == 1:
			return None
		if len(spl) != 4:
			print("Malformed entry", input_string)
			return None
		return cls(*spl)

	def shorten_gfbf(self, effect = None):
		"""Converts a +Effect/-Effect string to "gf" or "bf"
		Keyword arguments
		sentiment --- If none given, falls back to object's own sentiment"""
		if effect == None:
			effect = self.effect
		if effect == "+Effect":
			return "gf"
		if effect == "-Effect":
			return "bf"
		return 0		

	def quantify_gfbf(self, effect = None):
		"""Converts a +Effect/-Effect string to a number (+1 or -1).
		Keyword arguments
		sentiment --- If none given, falls back to object's own sentiment"""
		if effect == None:
			effect = self.effect
		if effect == "+Effect":
			return 1
		if effect == "-Effect":
			return -1
		return 0
	
	def __str__(self):
		return str(self.synset) + " (" + self.definition + ")"
	
	def __repr__(self):
		return self.entry_id + " " + str(self.synset)
		
def get_known_words(in_lexicon):
	"""Returns a set of verb lemmata that exist in entries' synsets."""
	"""# only compute once
	if self.known_words:
		return known_words"""
	known_words = set()
	for entry in in_lexicon:
		known_words |= set(entry.synset)
	return known_words

'''def make_senti_corpus(lexicon, complete_sst, size=100):
	"""Pick n (default 100) sentences from the SST whose predicates exist in the lexicon."""
	# Expected to be horribly slow.
	# Placeholder: just checks for any matching verb, instead of the predicate/all verbs
			
	for sentence in complete_sst:
		if [token for token in sentence if token.pos.startswith("VB") and token.lemma in self.known_words]:
			matching_sentences.append(sentence)
	if size == 0:
		return matching_sentences
	else:
		out = []
		while len(out) < size:
			remaining_matches = [sentence for sentence in matching_sentences if sentence not in out]
			# stop if we've run out of sentences
			if not(remaining_matches):
				print("Insufficient sentences ("+str(len(out))+"/"+str(size)+")")
				return out
			out.append(random.choice([sentence for sentence in matching_sentences if sentence not in out]))
		return out'''
	

def import_lexicon(filename = "effectwordnet/goldStandard.tff"):
	"""Imports a lexicon text file as a list of LexiconEntry objects."""
	with open(filename) as input_file:
		# Datenstruktur? Dictionary waere schnell zu durchsuchen, aber unhandlich fuer Ambiguitaet.
		lexicon = [LexiconEntry.from_string(line) for line in input_file.readlines()]

	return lexicon

def get_gfbf(token, in_lexicon, avoid_neutrals = False, no_more_fallback = False):
	"""Look up a token's GFBF status."""
	
	# print(token, type(token))
	if type(token) == SST_Token:
		lemma = token.lemma
	elif type(token) == nltk.Tree:
		# if this receives a Tree, it'll be a parse tree, without lemma information -> lemmatize now
		lemma = tree_to_string(token)
		# FIXME: the stemmer will try to stem what's already a stem ("create" -> "creat") 
	else:
		lemma = token
	if no_more_fallback:
		lemma = stem_word(lemma)
	matches = [entry for entry in in_lexicon if lemma in entry.synset]
	if(avoid_neutrals):
		polar_matches = [m for m in matches if m.effect != "Null"]
		if polar_matches:
			matches = polar_matches
	if not(matches):
		# First failure: retry with stemmed form. Afterwards: stop
		if not no_more_fallback:
			return get_gfbf(stem_word(lemma), in_lexicon, avoid_neutrals, no_more_fallback = True)
		else:
			print("Unknown verb", lemma)
			return False
	# TODO: apply WSD algorithm to matches and pick likeliest
	# placeholder: pick first sense
	selected_entry = matches[0]
	return selected_entry.effect
	# return selected_entry.quantify_gfbf(selected_entry.effect)

'''def verb_wsd(lemma, sentence):
	# new problem: matching the output (synset object) with the synsets in EffectWordNet
	return nltk.wsd.lesk(sentence, lemma, "v")'''

def export_lexicon():
	effectwordnet = import_lexicon()
	# sst = SST_Corpus()
	# corpus = make_senti_corpus(effectwordnet, sst, size=0)
	# print(len(corpus))
	out = open("sent_lexicon_print.txt", "w")
	out.write(str( list( [(e.entry_id, e.synset, e.effect) for e in effectwordnet] ) ))
	out.close()

if __name__ == "__main__":
	pass
