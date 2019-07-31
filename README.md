Introduction
The goal of our project is to apply the EffectWordNet +/- effect lexicon on sentences from the SST (Standford Sentiment Treebank) to get fine-grained, context based sentiment for subjects and objects of sentences.
In composite phrases, modifying words such as adjectives and adverbs alter the sentiment of a phrase - "a novel" is a neutral phrase, while "a bad novel" is clearly negative. Also, negation can invert the sentiment of a phrase, though the specific connotation of a negated phrase varies. These kinds of sentiment are explicit, easy to get, but usually not adequate, since the sentiment of a single word/phrase doesn’t necessarily stand for the sentiment of other parts or the entire sentence.
A new idea was brought up to get finer grained sentiment by inferring implicit sentiment from the context:
Verbs can be considered to have a negative or a positive effect on their objects and trigger a sentiment propagation. In a recent work: Deng/Choi/Wiebe, 2014, Lexical Acquisition for Opinion Inference, a verb is considered having a + effect when the subject does something good for the object (hence, also called GoodFor event), for example “promote” and vice versa, a - effect verb (BadFor event) indicates the subject does something bad to it’s object, for example “devastate”.
Similarly to negation, combining a malefactive verb with a negative object can lead to a positive sentiment: If a BadFor event happens to an object that's considered bad, the event's sentiment will be positive.
An event's sentiment passes to its subject, as the cause of a good or bad thing is expected to be good or bad, respectively, as well.
So if we know the author’s opinion towards the subject or object and the effect of the verb, we can infer the sentiment of the verb's other argument from this.
One example given by Deng explains the idea clearly:
Why would [President Obama] support [health care reform]? Because [reform] could lower [skyrocketing health care costs], and prohibit [private insurance companies] from overcharging [patients].
Suppose a sentiment analysis system recognises only one explicit sentiment expression, skyrocketing. According to the annotations, there are several gfbf events. Each is listed below in the form <agent, gfbf, object>
E1: <reform, lower, costs>
E2: <reform, prohibit, E3>
E3: <companies, overcharge, patients>
E4: <Obama, support, reform>
From [skyrocketing] we know the author’s opinion toward [costs] is negative, and lower is a 'bad for' verb, indicating that reform is bad for cost, which is negative, so the author’s attitude toward reform must be positive. And “Obama supports (goodfor) the reform” means [Obama] has a positive implicit sentiment. The same analysis process goes for other events. This kind of inference intuition are summarised by Deng in 4 formal rules: GFBF inference rules.
With the help of these rules we can propagate sentiment through the sentence, as long as the subjects and objects are connected by verbs with known gfbf effects.
Our group's task is to examine verb implicature, by using a lexicon of (benefactive or malefactive) verb effects and applying inference rules. We've applied the verb data from Choi/Deng/Wiebe's EffectWordNet lexicon (Deng/Choi/Wiebe, 2014, Lexical Acquisition for Opinion Inference: A Sense-Level Lexicon of Benefactive and Malefactive Events) to sentences from the Stanford Sentiment Treebank.
Resources
a. The Stanford Sentiment Treebank The Stanford Sentiment Treebank is the first corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews.
It was parsed with the Stanford parser (Klein and Manning, 2003) and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges. Sentiments are stored for each separate phrase. This new dataset allows us to analyze the intricacies of sentiment and to capture complex linguistic phenomena. However, the corpus does not contain part-of-speech, constituent, or argument labels for the parse trees, which posed a challenge to our project.
b. EffectWordNet EffectWordNet is a corpus of verbs, stored on a synset basis, and their classification as benefactive (goodFor/+ effect), malefactive (badFor/-effect) or neutral. These verbs are extracted from a manually annotated corpus and expanded to sense-level. In total, the gf lexicon contains 4,157 senses and the bf lexicon contains 5,071 senses.
We disregarded neutral verbs because our project focuses on sentiment changes caused by verbs, and only polar verbs usually cause changes in a predictable way.
Related Work
The Corpuses mentioned above are used/created in the following researches, to which we orient ourselves for this project.
1. Socher, Perelygin, Wu, Chuang, Manning, Ng, Potts, 'Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank', 2013
In this work, Recursive Neural Tensor Network was trained on Stanford Sentiment Treebank. This model outperforms all previous methods on several metrics. It pushes the state of the art in single sentence positive/negative classification from 80% up to 85.4%. The accuracy of predicting fine-grained sentiment labels for all phrases reaches 80.7%, an improvement of 9.7% over bag-of-features baselines.
Lastly, it is the only model that can accurately capture the effect of contrastive conjunctions as well as negation and its scope at various tree levels for both positive and negative phrases, which point out the direction for us of solving sentiment composition problem.
2. Deng/Choi/Wiebe, 'Lexical Acquisition for Opinion Inference: A Sense-Level Lexicon of Benefactive and Malefactive Events', 2014
A corpus of blogs and editorials about the Affordable Care Act, a controversial topic, was manually annotated with gfbf information by Deng et al. (2013). It provides annotated gfbf events and the agents and objects of the events. It consists of 134 blog posts and editorials.
Because the Affordable Health Care Act is a controversial topic, the data is full of opinions.
In this corpus, 1,411 gfbf instances are annotated, each including a gfbf event, its agent, and its object (615 gf instances and 796 bf instances). 196 different words appear in gf instances and 286 different words appear in bf instances; 10 words appear in both.
Deng/Choi/Wiebe extracted the verbs and their effects and expanded the to sense-level with the help of WordNet. In total, the gf lexicon contains 4,157 senses and the bf lexicon contains 5,071 senses.
3. Deng, Wiebe, 'Sentiment Propagation via Implicature Constraints', 2014
Using the manually annotated corpus from 2) as gold-standard, Deng and Wiebe pass the sentiment from one argument to another via Loopy Belief Propagation. The precision of the new approach is 10% higher than the old ways using only explicit sentiment, and the recall is 7% higher.
We basically followed the work to implement our project, so more of the work will be elaborated in introduction and implementation part of the documentation.
Corpus
We downsized the SST to the corpus we use step by step to get relevant results and faster runtimes.
First off we filtered it to sentences containing non-neutral verbs from the SentiWordNet lexicon.
Then we removed sentences longer than 60 characters, to get rid of sentences with infinite clauses because they introduce complications beyond our scope.
Finally, we only kept sentences with full SVO structures.
Because there's no parametrization for the Loopy Belief Propagation algorithm, separation into training and test data was unnecessary and we only evaluated once.
Implementiation
We implemented our project in Python, for convenience and access to powerful NLTK tools.
Data structures
Importing various sentiment dictionaries was a non-issue, but the SST's sparsely documented tree format proved a bit of a challenge.
NLTK's Tree classes were a convenient asset to our project. We chose the ParentedTree subclass to store hierarchy data for imported NLTK trees, but the Stanford parser's generated trees are unfortunately un-Parented.
Maintaining word order while generating the trees, and writing hierarchy-check methods for non-Parented trees, took up a surprising amount of time.
Determining sentence structure
Unfortunately, the phrase labels and part-of-speech labels that were used to generate the SST trees aren't stored in the corpus data, so we had to reconstruct them.
We used the Stanford parser to generate new parse trees. Even though the SST trees were made with the same parser, the generated trees sometimes differ from the ones in the corpus. This is due to ambiguity, potential version differences, and possible manual corrections to the SST data.
Because getting phrase structure data from constituent trees (though they lack phrase labels) seemed easier than aligning unordered dependency trees with the sentences, we computed argument structure using the reconstructed parse trees.
Our argument detector was very makeshift, looking for verb phrases and then using their sibling/child NPs as subject and object.
This rudimentary system cannot account for advanced constructions, such as idiomatic predicates, infinite clauses, coordination etc, and one or the other occurs in most sentences. But building a more powerful sentence parser would've been too much of a detour from our core task, and our scripts are sufficient as a proof-of-concept.
Side tasks
Aside from the aforementioned sentence parsing, we needed to deal with ambiguity, negation and stemming.
Since the dictionary contains base forms, and the SST data does not contain lemma information (like a CoNLL format corpus would), we needed to isolate verb stems. This was easy with NLTK's Snowball stemmer. If the surface form is not found in the dictionary, the method tries to look up its stemmed form. The unstemmed form is attempted first because the stemmer has false positives, and will further shorten stems that look like inflected forms.
As side tasks, mostly unrelated to our project, we saved word sense disambiguation and negation detection for last, and we didn't get around to implementing them.
WSD would've let us draw on the fact that EffectWordNet is a sense-level lexicon, and improve the results. The tentative baseline chooses the first verb sense found. We tried preferring non-neutral senses but this introduced more new errors.
NLTK features a module for the Lesk algorithm, but matching the used WordNet senses to the ones from EffectWordNet seemed like a rather unrewarding extra task.
The lack of a model for negation results in some of the more glaring classification errors. A rudimentary negation detection could match the adverbs in a VP with a keyword list, and reverse the predicate's GF/BF effect if a negation was found.
Algorithm - Loopy Belief Propagation
Belief propagation, also known as sum-product message passing, is a message passing algorithm for performing inference on graphical models. It calculates the marginal distribution for each unobserved node, conditional on any observed nodes. LBP iterates through the whole graph to propagate message. In our case, a LBP graph looks like:
      AGENT ----  VERB  ----  OBJECT  (--- VERB --- OBJECT)

    nodes: agent, object
          !! each node carries two main infos:
                1. it's neighbors(s)
                2. it's own explicit sentiment, acquired from SST
    edge: verb (gf/bf)
          !! for each edge, the verb's gfbf status must be retrieved from the +/- effect lexicon
A message from n_i to n_j over edge e_i_j has two values: m_i_j(pos) is how much information from node n_i indicates node n_j is positive, and m_i_j(neg) is how much information from node n_i indicates node n_j is negative.
In LBP, each node has a score, n_i(y), and each edge has a score, e_i_j(yi,yj). In our case, n_i(y) represents the writer’s explicit sentiment toward ni. e_i_j (yi, yj ) is the score on edge e_i_j , representing the likelihood that node n_i has polarity yi and n_j has polarity yj
To implement the LBP we need the following information:
Argument structure recognition to determine each word’s position in the graph and if they are each other’s neighbour In Deng’s work, a manually annotated small corpus(143 snippets) was not only used as a gold-standard. The corpus also provided subject, object and predicate spans for the algorithm, which is a very ideal situation. Yet in our case, SST is a corpus of 11855 sentences without argument structure annotations. It’s not only impossible to manually tag the corpus, but also meaningless, because in the end, the problem has to be faced. Hence we decided to exploit the SST’s tree structure and write a argument structure detector ourselves.
Our system runs the Stanford constituency grammar parser on each sentence of the corpus, detects predicate, subject, object by finding VP and NP phrases, and aligns those subtrees with the ones from the SST. The known phrases' sentiments are retrieved from SST data. We are unable to handle modal verbs, passive structures, coordination and many other complex constructions due to time and scope of the project, so we had to restrict ourselves to a small portion of the corpus.
explicit sentiment In LBP, each node may/can have it’s own explicit sentiment (coming from the noun itself or it’s adjectives). To some extend, explicit sentiment of one node is exactly what we propagate to other nodes. Now the question is, where do we get explicit sentiment. In Deng’s work a voting schema was developed using widely used and recognised sentiment extracting systems including opinion finder and other independent sentiment lexicons. In our case, explicit sentiment is directly extracted from SST. 
The SST format stores a phrase ID and a sentiment label for (the string value) of each phrase in the SST sentence trees. Thus, each argument phrase could be simply looked up.
If the subject/object phrase of a sentence, detected by the Stanford parser, has no matching subtree in the SST data because of mismatched parses, the sentence was filtered out of the corpus.
the effect of the edge the four rules mentioned in the introduction are formalised in the 8 formulae below:
if the verb is a GoodFor verb (with a + effect), it makes both nodes it connects have the same polarity, so the score of same polarity is set to 1 and the opposite polarity to 0.
            e_p_p = 1
            e_n_n = 1
            e_p_n = 0
            e_n_p = 0
if the verb is a BadFor verb (wit a - effect), it makes both nodes it connects have the opposite polarity, so the score of same polarity is set to 0 and the opposite polarity to 1.
            e_p_p = 0
            e_n_n = 0
            e_p_n = 1
            e_n_p = 1
After gathering enough information, we can draw a Graph for each snippet to be analysed, like so:
    AGENT ----  VERB  ----  OBJECT  (--- VERB --- OBJECT)
for the convenience of the algorithm, we turn the graph into a dictionary that stores information of every nodes in the graph:
graph= {node1:([(neighbor1, edge1),(neighbor2, edge2)...],(es_pos,es_neg)),node2:([(neighbor1, edge1),(neighbor2, edge2)...],(es_pos,es_neg)),....}
the keys are nodes string, and values are a tuple of (neighbors(name_str, sort of edge:gf/bf)), explicit_sentiment(positive,negative))
The message from n_i to its neighbor n_j is computed as: LBP message computation.
and the sentiment after propagation is the the product of the node’s own sentiment and all the messages it gets from it’s neighbours. LBP node sentiment.
For example:
for the sentence: 'Mike hates the good book that helps nice people'
the graph looks like:
MIKE --- HATE --- BOOK --- HELP --- PEOPLE
and the corresponding dictionary:
graph = {'book': ([('Mike','bf'),('people','gf')], (7.5, 2.5)), 'Mike': ([('book', 'bf')], (5, 5)),'people':([('book','gf')],(7,3))}
for node MIKE, the message it gets from the node BOOK (7.5,2.5) is (0.1, 0.9), while node PEOPLE' s sentiment is (7.5,2.5) and MIKE’s final sentiment after propagation is negative.
to see the power of neighbor's neighbor, reverse PEOPLE's positive and negative sentiment to (2.5,7.5), the message MIKE get from BOOK changes to (0.5,0.5), and it's final sentiment is now: neutral.
The situations above happen because MIKE has a positive neighbour BOOK connected with a BadFor verb, which passes it negative sentiment, but the neighbour itself has another neighbour PEOPLE, if PEOPLE is positive, it makes BOOK even more positive and MIKE very negative. The other way around, if PEOPLE is negative, BOOK is less positive and MIKE is less negative, the chain reaction is like the saying: enemy's enemy is a friend.
We can see the algorithm functions as the theory expected.
Evaluation
In the next step we apply the LBP to the corpus we picked out from SST.
A rough first evaluation (combining the neutral and negative classifications for sentiment) yields an accuracy of 67%.
A lot of the errors are caused by complex constructions - even in our corpus of short sentences, most sentences contain idiomatic predicates, ditransitive predicates or embedded sentences, while our argument detector only accounts for NP subjects and objects and a single V as a predicate.
Conclusion and Problems
On one hand, the LBP passes context based sentiment messages pretty well, and provides us with a new way of automatically generating a sentiment corpus.
On the other hand, inferring argument sentiments requires detecting argument structure, stemming the predicate, and disambiguating between the predicate's possible synsets.
The accumulated mistakes from all those automated steps cause a great many errors. Thus our system is in a prototype state and not yet suitable for evaluating the effectiveness of our approach.
As mentioned, the detection of argument structure in the SST's sentences was the biggest problem. The Stanford dependency parser, integrated in NLTK, could've supplied subject, predicate and object, but since dependency parsing destroys a sentence's word order, we couldn't align the parse graphs with the SST trees again.
In conclusion: SST is better at detecting sentiment composition rules based on syntactical structures, like negation. But it performs badly in terms of dependency structure. The better approach for combining the 2 corpora seems to be using the tagging result from +/- effect lexicon as a supplement of SST’s sentiment tags (but firstly we have to analyse the sentences with an independent dependency parser) and retrain SST with these finer grained tags.
Observations
This approach ignores the sentiment of verbs, only treating it as a modifier to its arguments' sentiments. 
A strongly connoted verb with generic, neutral arguments will not be treated correctly, so it turns out explicit verb sentiment can be important:
I loved it !
I: positive     loved: +Effect    it: neutral
By intuition, "it" would be positive since it is loved, but in our model, verbs cannot create a polarity that isn't there. We observe that verbs can have a polarizing effect on their arguments, after all. A more developed system would observe these behaviours and apply the data as a feature.
Another observaton would be: the sentiment propagation highly depends on the type of content. News articles used in Deng's work is more logically organised and sentences have strong connections with each other. However, the author of film reviews- a non-serious type of text are not trained writer and may not focus on the logical connection of the objects mentioned in a comment, which makes it harder to propagate sentiment on the whole review's level.
Citations
Stanford Sentiment Treebank (Socher, Perelygin, Wu, Chuang, Manning, Ng, Potts, 'Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank', 2013)
Deng, Wiebe, 'Sentiment Propagation via Implicature Constraints', 2014
EffectWordNet (Deng/Choi/Wiebe, 'Lexical Acquisition for Opinion Inference: A Sense-Level Lexicon of Benefactive and Malefactive Events', 2014)
Anand, Reschke, 'Verb Classes as Evaluativity Functor Classes', 2010
Natural Language Toolkit 3.2.2: Tree class, stemmer, parser interfaces
Stanford parser (Chen, Manning, 'A Fast and Accurate Dependency Parser using Neural Networks. Proceedings of EMNLP 2014', 2014)
Pang and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL, pages 115–124.
