#!/usr/bin/python3
# coding=UTF-8

__author__		= "Daaje Meiners"

import nltk
import nltk.parse.stanford
"""from os import environ
environ['STANFORDTOOLSDIR'] = '$HOME/programs/stanford_tools'
environ['CLASSPATH'] = '$STANFORDTOOLSDIR/stanford-parser-full-2016-10-31'"""


_stemmer = nltk.stem.snowball.EnglishStemmer()
"""https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software"""
# """stanford-parser-full-2016-10-31/"""

# parser on ella
# _parser = nltk.parse.stanford.StanfordParser(path_to_jar = "/mnt/resources/processors/parser/stanfordparser-3.6.0/stanford-parser.jar", path_to_models_jar = "/mnt/resources/processors/parser/stanfordparser-3.6.0/stanford-parser-3.6.0-models.jar")

# parser on laptop
_parser = nltk.parse.stanford.StanfordParser("/home/elaine/programs/stanford_tools/stanford-parser-full-2016-10-31/stanford-parser.jar", path_to_models_jar = "/home/elaine/programs/stanford_tools/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar")

#_dep_parser = nltk.parse.stanford.StanfordDependencyParser()
# path_to_jar = "stanford-parser.jar", path_to_models_jar = "stanford-parser-3.70-models.jar"

class SST_Corpus:
	"""Container for treebank data.
	
	Variables:
	treebank --
	id_dictionary --
	label_dictionary --
	
	Class methods:
	read_treebank --
	read_dictionary --
	"""
	
	# to do: import datasetSplit
	def __init__(self, path = "stanfordSentimentTreebank/", only_lines = None):
		self.treebank = SST_Corpus.read_treebank(path, only_lines)
		
		# to do: iterate over enumerated sentences + put them in training or test data according to datasetSplit
		# print("Allocated datasets")
		self.id_dictionary = SST_Corpus.read_dictionary(path, "dictionary.txt")
		print("Imported phrase dictionary")
		self.label_dictionary = SST_Corpus.read_dictionary(path, "sentiment_labels.txt")
		print("Imported sentiment dictionary")
		print("Successfully imported corpus!!")
		
	'''def parse_sentences(self):
		"""Adds phrase structure and PoS data for each sentence in corpus. """
		for sentence_tree in self.treebank:
			tag_phrase_structure(sentence_tree, write = True)'''
	
	@classmethod
	def read_treebank(cls, path = "stanfordSentimentTreebank/", only_lines = None):
		""" Imports sentences from SST as trees.
		
		Arguments:
		path       --- Folder containing the SST files. 'stanfordSentimentTreebank/' by default.
		only_lines --- Optional. List of sentence indexes. If given, loads the numbered sentences and skips the rest."""
		with open(path+"SOStr.txt", mode = "rt") as sostr:
			with open(path+"STree.txt", mode = "rt") as stree:
				# zip sentences and tree structures
				tree_data = list( zip(sostr.readlines(), stree.readlines()) )
		imported_treebank = []
		print("Loaded sentence data")
		
		# DEBUG
		import time
		start_time = time.time()
		cnt = 0
		
		for index, line in enumerate(tree_data):
			# 
			
			if only_lines:
				# index+1: because dataset_sentences starts at 1.
				# 
				if index+1 > max(only_lines):
					break
				if not(index+1 in only_lines):
					continue
			imported_treebank.append( SST_Sentence.from_corpus_lines(line[0], line[1], index+1) )
			
			# DEBUG
			cnt += 1
			if not cnt % 2:
				print(str(cnt)+ ",", str(time.time() - start_time) + "s")
				
		print("Imported trees")
		return imported_treebank
		
	@classmethod
	def read_dictionary(cls, path = "stanfordSentimentTreebank/", docname = "dictionary.txt"):
		""" Imports a dictionary (phrase ID or label) from a document. """
		out_dict = dict()
		with open(path+docname, mode = "rt") as id_doc:
			for l in id_doc.readlines():
				if l.rstrip():
					(key, val) = l.rstrip().split("|")
					out_dict[key] = val
		return out_dict
		
	def lookup_sentiment(self, phrase):
		"""Looks up the sentiment of the given phrase tree or string in the corpus's dictionary.
		
		Arguments:
		phrase -- The phrase to be looked up.
		Returns a floating point number."""
		if type(phrase) == nltk.ParentedTree:
			phrase = tree_to_string(phrase)
		phrase_id = self.id_dictionary.get(phrase)
		sentiment = self.label_dictionary.get(phrase_id)
		# print(phrase_id, sentiment)
		if not(sentiment):
			print("No sentiment for phrase", phrase)
			return 0.5
		else:
			# dictionary entry is a number string -> convert to float
			return float(sentiment)
		
class SST_Sentence:
	"""Container for a sentence tree.
	
	Contains the tree, its index and parse information. """
	
	def __init__(self, sentence_tree, index = None):
		self.sentence_tree = sentence_tree
		self.parse_tree = tag_phrase_structure(sentence_tree)
		self.index = index
		self.subject_st = None
		self.predicate_st = None
		self.object_st = None
		self.svo_data = None	
		self.get_event_subtrees(write=True)
		
	@classmethod
	def from_corpus_lines(self, sostr_line, stree_line, index = None):
		"""Generates a sentence object from an SOStr line and an STree line."""
		tree = load_sst_tree(sostr_line, stree_line)
		return SST_Sentence(tree, index)
		
	def __str__(self):
		return tree_to_string(self.parse_tree)
	
	def get_event_subtrees(self, write = True):
		# find a sentence's event structure: 
		# retrieve subject, predicate, object  via get_SVO.
		# search the sentences's SST tree for the subtrees for those phrases, using find_phrase.
		# that way we can access the phrase (and token) sentiments from SST, the effect data, and the lexicon entry.
		svo_data = self.get_SVO()
		if not(svo_data):
			return False
		# print([bool(elm) for elm in svo_data])
		# if tree_to_string doesn't work: " ".join([l for l in svo_data[0].leaves()])
		subject_st = find_phrase(self.sentence_tree, tree_to_list(svo_data[0]))
		predicate_st = find_phrase(self.sentence_tree, tree_to_list(svo_data[1]))
		object_st = find_phrase(self.sentence_tree, tree_to_list(svo_data[2]))
		if(write):
			self.subject_st = subject_st
			self.predicate_st = predicate_st
			self.object_st = object_st
		return (subject_st, predicate_st, object_st)

	def get_SVO(self):
		if self.svo_data:
			return self.svo_data
		
		phrase_structure_tree = self.parse_tree
		
		main_vp = SST_Sentence.find_phrase_of_type(phrase_structure_tree, "VP")
		if not main_vp:
			print("no vp")
			return False
		
		"""# initialize index to ensure that the first attempt happens
		np_index = float("inf")"""
		# regular tree doesn't store parents -> manually retrieve the parent.
		# subj_np = SST_Sentence.find_phrase_of_type(main_vp.parent(), "NP")
		vp_parent = get_parent(main_vp, phrase_structure_tree)
		subj_np = SST_Sentence.find_phrase_of_type(vp_parent, "NP")
		if not subj_np:
			print("no np")
			return False
		# if the first np wasn't to the left of the vp, there's no subject
		if list(phrase_structure_tree.subtrees()).index(subj_np) > list(phrase_structure_tree.subtrees()).index(main_vp):
			print("no subject np")
			return False
		obj_np = SST_Sentence.find_phrase_of_type(main_vp, "NP")
		predicate = SST_Sentence.find_phrase_of_type(main_vp, "VB")
		"""if obj_np.parent() != main_vp:
			return False"""
		if not obj_np:
			print("no object np")
			return False
		if not predicate:
			print("no predicate")
			return False
			
		self.svo_data = (subj_np, predicate, obj_np, main_vp)
		return (subj_np, predicate, obj_np, main_vp)
		
		# Predicate: Topmost VB in topmost VP.
		# Subject: NP that's a left-side sibling of the topmost VP. i. e. NP that has a VP to the right of it.
		# (i'm not sure how to compare tree positions and ensure that it's on the left side)
		# Object: NP in the predicate's VP.
	
	# Not needed, since trees are now ordered from the start
	'''@classmethod
	def sort_tree(cls, phrase_tree):
		"""Generates an ordered version of a tree."""
		#TODO
		pass'''
		
	'''@classmethod
	def get_argument_structure(cls, sentence):
		"""DEPRECATED. Identifies subject, predicate, object of a sentence. Requires PoS tags."""
		# descend through tree. find token 
		# to do: account for composite nouns ("Arnold Schwarzenegger")
		"""subject = cls.find_part_of_speech(sentence, "NN")"""
		# VP is sibling of subject
		print(find_siblings(sentence, subject))
		predicate = None
		for s in find_siblings(sentence, subject):
			predicate = cls.find_part_of_speech(assumed_vps, "VB")
			if predicate:
				break
		# here's an underscore because "object" is unsurprisingly reserved
		object_ = None
		for s in find_siblings(sentence, subject):
			object_ = cls.find_part_of_speech(assumed_vps, "VB")
			if object_:
				break
		if not(subject) or not(predicate) or  not(object_):
			print("Failed to find arguments")
			return None
		return (subject, predicate, object_)
			# noun: NN*
			# verb: VB*'''
	
	@classmethod
	def find_phrase_of_type(cls, sentence, phrase_type, is_sst_tree = False):
		"""Finds the topmost leftmost node of the given phrase type in a parse tree.
		Arguments:
		sentence -- A sentence/phrase tree.
		phrase_type -- phrase tag. Partial matching with start of tag. NP, VP, S, SBAR, ...
		is_sst_tree -- False by default. Set to true if browsing a tree of SST_Tokens.
		Descends through a sentence tree, finding the topmost, leftmost token of the given type."""
		# leafs can't be phrases! the branch leading to the leaf should be labelled with the leaf's PoS tag.
		if not isinstance(sentence, nltk.Tree):
			raise TypeError("Invalid tree type", type(sentence))
			return None
		try:
			if is_sst_tree:
				candidates = [phrase for phrase in sentence.subtrees() if phrase.label().part_of_speech.startswith(phrase_type)]
			else:
				# candidates = [phrase for phrase in sentence.subtrees() if type(phrase.label()) is str]
				candidates = [phrase for phrase in sentence.subtrees() if phrase.label().startswith(phrase_type)]
		except AttributeError:
			print("Attribute error when searching for", phrase_type)
			# idr why i did this: # print(pos, [l.part_of_speech for l in sentence.leaves()])
			return None
		if not(candidates):
			print(phrase_type, "not found in", [phrase.label() for phrase in sentence.subtrees()])
			print(sentence)
			return None
			
		# Cancelled: treeposition only works for parented tree
		# order by height, then position.
		# candidates = sorted(candidates , key=lambda c: (get_subtree_height(sentence, c)) )
		# return leftmost candidate
		return candidates[0]
		"""
		for elm in sentence:
			
			cls.find_part_of_speech(subtree, pos)"""
	
	'''@classmethod		
	def find_part_of_speech(cls, sentence, pos):
		"""Finds the topmost leftmost node of the given PoS type in a sentence tree.
		Arguments:
		sentence -- A sentence/phrase tree.
		pos -- part-of-speech tag. Partial matching with start of tag. NN for noun, VB for verb.
		find_leaves -- False by default. If false, searches for tags in 
		Descends through a sentence tree, finding the topmost, leftmost token of the given type."""
		# if we've reached a leaf: check whether it matches, stop if not
		if type(sentence) == SST_Token:
			if sentence.part_of_speech.startswith(pos):
				return sentence
			else:
				return None
		try:
			candidates = [phrase for phrase in sentence.subtrees() if phrase.label().part_of_speech.startswith(pos)]
		except AttributeError:
			print("Attribute error when searching for", pos)
			# idr why i did this: # print(pos, [l.part_of_speech for l in sentence.leaves()])
		if not(candidates):
			return None
		# order by height, then position.
		candidates = sorted(candidates, key=lambda c: (get_subtree_height(sentence, c), c.node_id) )
		# return leftmost candidate
		return candidates[0]
		"""
		for elm in sentence:
			
			cls.find_part_of_speech(subtree, pos)"""'''

# retrieve parent for subtree of a non-Parented tree
def get_parent(subtree, containing_tree):
	"""Find a subtree's parent in a Tree.""" 
	"""for i, phrase_candidate in enumerate( containing_tree.subtrees() ):
		if phrase_candidate == subtree:
			position = containing_tree.leaf_treeposition(i)
			break
	# subtree not found in containing tree? it's a failure.
	else:
		print(phrase, "not found")
		return None"""
	subtree_list = list(containing_tree.subtrees())
	if type(containing_tree) is nltk.ParentedTree:
		return subtree.parent()
	i = subtree_list.index(subtree) # subtrees() is a generator, must be converted
	path = containing_tree.treepositions()[i]
	depth = len(path)
	# iterate backwards and check subtrees higher up. the parent's among them
	for j in reversed(range(i+1)):
		parent_candidate = subtree_list[j]
		if len( containing_tree.treepositions()[j] ) < depth:
			if subtree in list(parent_candidate.subtrees()):
				# print("found parent:", j, containing_tree.treepositions()[i], "{", containing_tree.treepositions()[j])
				return parent_candidate
			"""else:
				# print("Non-parent:", parent_candidate, "of", subtree, containing_tree.treepositions()[i], "{", containing_tree.treepositions()[j])"""
	return None

def tree_to_string(phrase_tree):
	"""Converts a tree to a phrase string to look up in the dictionary. """
	return " ".join([word_string for word_string in tree_to_list(phrase_tree)])

def tree_to_list(phrase_tree):
	"""Converts a tree to a list of its tokens. """
	leaves = phrase_tree.leaves()
	# Kludge: put scrambled tokens in order
	# (now tokens aren't scrambled anymore so we don't need this)
	# leaves = sorted(leaves, key = lambda l: l.node_id)
	return [leaf.surface_form if type(leaf) is SST_Token else leaf for leaf in leaves]	
	
def pos_tag(sentence, write = True):
	"""Supplies part of speech tags for a sentence tree, via NLTK."""
	tagged_sentence = nltk.pos_tag(tree_to_list(sentence))
	# print(tagged_sentence)
	if write:
		for token, tagged in zip(sorted(sentence.leaves(), key = lambda l: l.node_id), tagged_sentence):
			"""if type(token) != SST_Token:
				continue"""
			token.part_of_speech = tagged[1] # tagged is a tuple of (surface form, tag)
			# print(token.surface_form, token.part_of_speech)
	return tagged_sentence

"""def get_dependency_graph(sentence):
	parses = _dep_parser.raw_parse(tree_to_string(sentence.sentence_tree))
	# TODO: desambiguation
	# tree() can make a tree, but it loses its labels
	return parses[0]
	# to use dependency graph, align the tokens with the ones from the treebank.
	# the dependency parser scrambles the order of the tokens, so how do we do that?
	# then we can take subj, pred, obj from the nsubj and dobj connections"""
	
def get_parent_index(subtree, in_tree):
	i = list(containing_tree.subtrees()).index(subtree)
	path = containing_tree.treepositions()[i]
	return path[-1]

'''def get_subtree_height(in_tree, subtree):
	"""	Returns the height of a subtree in a tree.
	
	Analogous to get_leaf_height (or not.)
	Credit to http://stackoverflow.com/questions/25815002/nltk-tree-data-structure-finding-a-node-its-parent-or-children"""
	subtree_index = list(in_tree.subtrees()).index(subtree)
	# return subtree.treeposition

def get_leaf_height(in_tree, leaf):
	"""	Returns the height of a leaf in a tree.
	Credit to http://stackoverflow.com/questions/25815002/nltk-tree-data-structure-finding-a-node-its-parent-or-children"""
	leaf_index = list(in_tree.leaves()).index(leaf)

def find_leaf_siblings(in_tree, leaf):
	"""Lists the elements in a tree that share a level with a leaf.
	Credit to http://stackoverflow.com/questions/25815002/nltk-tree-data-structure-finding-a-node-its-parent-or-children"""
	return [elm for elm in in_tree.subtrees(lambda t: t.height() == get_leaf_height(in_tree, leaf))]'''


def find_phrase(sentence, phrase):
	"""Searches a sentence for a subtree matching a given phrase. """
	if isinstance(phrase, nltk.Tree):
		phrase = tree_to_list(phrase)
	if not len(phrase):
		return False
	matches = [s for s in sentence.subtrees() if tree_to_list(s) == phrase]
	# i know finding the best partial match is its own can of worms.
	# so, slow placeholder implementation for now:
	# shorten phrase 
	if not(matches):
		return find_phrase(sentence, phrase[:-1])
	if len(matches) > 2:
		# multiple matches: TODO: disambiguation instead of selecting first match
		# how2: remember how 
		print("Ambiguous phrase", phrase, ":", matches)
	return matches[0]


class Tree_Segment:
	def get_leaf_indices(self):
		return {l.node_id for l in self.get_leaves()}
	

class SST_Token(Tree_Segment):
	"""SST_Token: Stores a token from an SST sentence.
	
	Variables:
	node_id -- The index in the sentence. Starts at 0 unlike STree data
	surface_form -- The token's surface form.
	part_of_speech -- Optional, not given by corpus.
	lemma -- Optional, not given by corpus.
		
	Methods:
	---"""
	def __init__(self, node_id, surface_form, part_of_speech = None, lemma = None):
		self.node_id = node_id
		self.surface_form = surface_form
		self.part_of_speech = part_of_speech
		if(lemma):
			self.lemma = lemma
		else:
			self.lemma = stem_word(surface_form)
		
	def __str__(self):
		s = self.surface_form
		if self.part_of_speech:
			s += " (" + self.part_of_speech +")"
		return s
	
	# breaks a PEP: repr should be unambiguous
	def __repr__(self):
		return self.__str__() # + " " + str(id(self))
	
	# a leaf's leaves are only the leaf itself
	def get_leaves(self):
		return set([self])
	
	def get_leaf_indices(self):
		return {self.node_id}
		
	def get_children(self):
		return []

class Node_Label(Tree_Segment):
	"""A label for a non-leaf tree node. Stores the index and the parser data.
	
	Variables:
	node_id --- The index of this node.
	part_of_speech --- Phrase annotation, usually from Stanford parser.
	assoc_subtree --- The tree structure labelled with this node.
	"""
	def __init__(self, node_id, part_of_speech = ""):
		self.node_id = node_id
		self.part_of_speech = part_of_speech
		self.assoc_subtree = None
		self.contained_leaves = set()
		self.children = []
	
	def get_leaves(self):
		return self.contained_leaves
		
	def get_children(self):
		return self.children
		
	def __str__(self):
		# return "<" + self.part_of_speech + " " + str([c.node_id for c in self.children]) + ">"
		return self.part_of_speech + " " + str(self.node_id)
	
	def __repr__(self):
		return self.__str__()
		


def tag_phrase_structure(sentence, write = False):
	parse = _parser.raw_parse(tree_to_string(sentence))
	# this iter seems to contain just a single tree.
	# why.
	# (probably for handling multiple sentences)
	parse_tree = parse.__next__()
	"""for token_subtree, parse_subtree in zip(sentence.subtrees(), parse_tree.subtrees()):
		# print(parse_subtree.label())
		token_subtree.label().part_of_speech = parse_subtree.label()
		# print(subtree.label(), phrase_parse)"""
	# TODO: ensure overlapping alignment.
	# for each phrase: check whether phrase text matches. track which subtrees we're currently in.
	# in case of mismatch/repetition, try to tag a phrase lower in the tree.
	# partial matches are better than nothing (in case of displaced quotation marks)
	return parse_tree

def stem_word(word):
	"""Stems a word via the snowball algorithm from nltk."""
	return _stemmer.stem(word)

def load_sst_tree(sentence, nodes):
	# wrapper method for the currently preferred implementation.
	return grow_ordered_tree(sentence, nodes)

def grow_ordered_tree(sentence, nodes):
	""" builds an nltk ParentedTree from an SOStr line and an STree line.
	Top down, stores and maintains sentence order. """
	sentence = sentence.rstrip()
	tokens_list = sentence.split("|")
	nodes_list = [int(i) for i in nodes.split("|")] # number strings to ints
	# list of entries. parents-to-children relations are stored in node label objects
	# tree_data = [None] * len(tokens_list) + [Node_Label(i) for i in range(len(tokens_list), len(nodes_list))]
	# tree should have the leaves on extra branches, bc the annotation tree does too
	tree_data = [Node_Label(i) for i in range(len(nodes_list))]
	# dict: children to parents
	parent_data = dict()
	s_node = None
	
	# print(nodes_list)
	for i, parent_id in enumerate(nodes_list):
		# STree starts with 1, python indexing with 0
		parent_data[i] = parent_id-1
		if i < len(tokens_list):
			# index also gets stored in token object, so they can be ordered
			new_leaf = SST_Token(i, tokens_list[i])
			tree_data[i].children.append(new_leaf)
			tree_data[i].contained_leaves.add(new_leaf)
		
		if parent_id != 0:
			# store as child if the parent exists
			tree_data[parent_id-1].children.append(tree_data[i])
		else:
			# otherwise make a root node
			if s_node:
				# 2 roots? that shouldn't happen
				raise ValueError("Root node conflict!")
			s_node = tree_data[i]
			# print(i, parent_id, s_node, type(s_node))
	# after all node objects have been constructed: propagate the left-to-right order upwards
	# print(tree_data, "\n", parent_data)
	for node in tree_data:
		try:
			parent = tree_data[parent_data[node.node_id]]
		except:
			print("Tree construction error ", node.node_id)
		parent.contained_leaves |= node.get_leaves()
		
	rootnode = Node_Label(-1)
	rootnode.children = [s_node]
	sentence_tree = grow_branches(rootnode, tree_data)
	return sentence_tree

def grow_branches(starting_node, from_list):
	""" Recursively grows top-down subtree for an SST phrase.
	
	If the node label objects contain information about contained leaves,
	the branches will be ordered left-to-right to maintain sentence order."""
	
	# tokens are leaves, no more growth from there
	"""if not isinstance(starting_node, nltk.Tree):
		print(type(starting_node), starting_node)"""
	if type(starting_node) is SST_Token:
		return starting_node
	
	# print(len(from_list), starting_node)
	children = starting_node.children
	# if contained-leaf data was collected, order the subtrees left-to-right.
	children = sorted(children, key=lambda c: min(c.get_leaf_indices()))
	
	# recursively grow subtrees
	branches = list(grow_branches(child, from_list) for child in children)
	subtree = nltk.ParentedTree(starting_node, branches)
	return subtree

# test methods:
# compare each character in a string
def compare_strings(stra, strb):
	return "".join([str(int(x == y)) for x,y in zip(stra, strb)])
	
def reverse_lookup(val, dictionary):
	return [key for key, value in dictionary.items() if value == val][0]



def load_selected_subcorpus(selection_file = "Corpus_5.txt", max_lines = None, path = "stanfordSentimentTreebank/"):
	with open(selection_file) as corpus_selection:
		index_list = []
		elines = enumerate(corpus_selection.readlines())
	for i, line in elines:
		spl = line.rstrip().split("\t")
		"""if len(spl) < 3:
			continue"""
		# parse valid lines, skip the rest
		try:
			index_list.append( int(spl[0]) )
		except ValueError:
			continue
		# only load first n lines of corpus: tentative downsizing for runtime purposes
		if max_lines:
			if i+1 >= max_lines:
				break
	return SST_Corpus(path, only_lines = index_list)

if __name__ == "__main__":
	
	corpus = load_selected_subcorpus("Corpus_7.txt")
	s1 = corpus.treebank[0]
	example_tree = s1.sentence_tree
	print(example_tree)
	
	# print(tree_to_string(example_tree))
	print(corpus.lookup_sentiment(example_tree))
	
	print(pos_tag(example_tree, False))
	# print(tag_phrase_structure(example_tree))
	s1.parse_tree.draw()
	example_tree.draw()
	
	
	# dictionary debugging:
	# print([(key, corpus.id_dictionary[key]) for key in corpus.id_dictionary.keys() if key.startswith("The Rock")])
	# print(compare_strings(tree_to_string(example_tree), "The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."))
	# why does the string not match what it should match? (it was a line break)
	# print(compare_strings(tree_to_string(example_tree), reverse_lookup(corpus.id_dictionary, '226166') ) )


# Development nodes - ordered trees:
# 
# Approach 0: recursive ascent!
#   iterate over all children, build their parenting subtrees, repeat with those subtrees
#   EXCEPT children can have different layers so there's always holes in the trees
#   what iteration order can prevent this? growing bottom-up, finishing each lower subtree before ascending?
# Approach 1: 
#   from child node (starting leaf): request construction of parents.
#   parent (downwards): request construction of children, in correct order of children.
#   Conundrum: how do we find out which subtrees contain the leftmost children? (we probably don't.)
# Approach 2:
#   Build top-down, somehow learn in advance which leaves are the leftmost.
# Approach 3 (implemented):
#   Bottom-up propagation of the numbers of associated leafs for each node.
#   Then build top-down, growing the subtrees that contain the leftmost leaves, first.
# Solution 1, bottom-up:
#   for each child node, request construction of parents.
#   If the child has no "nieces/nephews", subtree construction can be completed.
#   Otherwise, postpone the construction.
#   Repeat, ascending from new subtrees, go left to right in each iteration.
#   This way, we end up building subtrees, finishing the lowest/smallest first.
#   Seems inefficient, does a number of redundant checks.
#   Also kind of a brain-twister to implement.
