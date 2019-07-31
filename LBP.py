
'''
    for each event(sentence), draw a Graph like so:

    AGENT ----  VERB  ----  OBJECT  (--- VERB --- OBJECT)

    nodes: agent, object
          !! each node carries two main infos:
                1. it's eighbors(s)
                2. it's own explicit sentiment that need to be required from SST
    edge: verb (+Effect/-Effect)
          !! for each edge, if it's a +Effect or -Effect must be determined by looking up in +/- effect lexicon

'''
class Node:
    def __init__(self,node,graph):
        self.graph=graph
        self.node=node
        self.connections=self.graph[self.node][0]    #neighbor&edge [(n1,v1),(n2,v2)]
        self.es=self.graph[self.node][1]             #explicit sentiment(pos,neg)
        self.m_from_neib=[]

    '''

           the message(pos/neg) one node i gets from it's neighbor j is the product of j's explicit sentiment, the edge's influence(see inference rules below)
           and the message i got from its other neighbors

    '''
    def calc_message(self,neib):
        #neib's explicit sentiment
        neib=Node(neib,self.graph)
        es_pos = neib.es[0]
        es_neg = neib.es[1]

        #get the node and neib's edge
        edge=[]
        for n,v in self.connections:
            if n==neib.node:
                edge.append(v)

        # a +Effect makes both nodes it connects have the same polarity
        if edge[0] == "+Effect":
            e_p_p = 1
            e_n_n = 1
            e_p_n = 0
            e_n_p = 0
        # a -Effect makes both nodes it connects have the opposite polarity
        else:
            e_p_p = 0
            e_n_n = 0
            e_p_n = 1
            e_n_p = 1

        #calc the message the neib gets from it's other neighbors(if there is any)
        m_other_neib=[1,1]
        for other_n,v in neib.connections:
            if other_n != self.node:
                m = neib.calc_message(other_n)

                m_other_neib[0] *= m[0]
                m_other_neib[1] *= m[1]

        m_pos = es_pos * m_other_neib[0] * e_p_p + es_neg *m_other_neib[1] * e_n_p
        m_neg = es_pos * m_other_neib[0] * e_p_n + es_neg * m_other_neib[1] * e_n_n

        return (m_pos,m_neg)


    #the node's sentiment after the propagation process
    def get_senti(self):
        senti_pos=self.es[0]
        senti_neg=self.es[1]


        for neib,v in self.connections:
            m_pos = self.calc_message(neib)[0]
            m_neg = self.calc_message(neib)[1]

            senti_pos*=m_pos
            senti_neg*=m_neg

        if senti_pos > senti_neg:
            return ("positive")
        elif senti_pos < senti_neg:
            return ("negative")
        else:
            return ("neutral")

    #return normalized message
    def get_message(self,neib):
        m=self.calc_message(neib)
        m_pos = m[0]
        m_neg = m[1]
        m_pos_norm = m_pos / (m_pos + m_neg)
        m_neg_norm = m_neg / (m_pos + m_neg)

        return (m_pos_norm, m_neg_norm)










'''
    a simple test:
    for the sentence: 'Mike hates the good book that helps nice people'
    the graph looks like:   MIKE --- HATE --- BOOK --- HELP --- PEOPLE

    for the convenience of the algorithm, we turn the graph into a dictionary that stores information of every nodes in the graph:
    graph= {node1:([(neighbor1, edge1),(neighbor2, edge2)...],(es_pos,es_neg)),node2:([(neighbor1, edge1),(neighbor2, edge2)...],(es_pos,es_neg)),....}
    the keys are nodes string, and values are a tupel of (neighbors, explicit_sentiment)

    for our example:
    graph = {'book': ([('Mike','-Effect'),('people','+Effect')], (7.5, 2.5)), 'Mike': ([('book', '-Effect')], (5, 5)),'people':([('book','+Effect')],(7,3))}

    for node MIKE:
        the message it get's from the node BOOK(7.5,2.5): (0.1, 0.9), while node PEOPLE' s sentiment is (7.5,2.5)
        it's final sentiment after propagation: negative

        to see the power of neighbor's neighbor, change PEOPLE's sentiment to (2.5,7.5)
        the message MIKE get from BOOK changes to (0.5,0.5), and it's final sentiment is now: neutral
        because enemy's enemy is a friend.

'''

if __name__ == "__main__":

    graph = {'book': ([('Mike','-Effect'),('people','+Effect')], (7.5, 2.5)), 'Mike': ([('book', '-Effect')], (5, 5)),'people':([('book','+Effect')],(7.5,2.5))}
    n=Node('Mike',graph)

    print(n.get_message('book'))
    print(n.get_senti())


    graph = {'book': ([('Mike','-Effect'),('people','+Effect')], (7.5, 2.5)), 'Mike': ([('book', '-Effect')], (5, 5)),'people':([('book','+Effect')],(2.5,7.5))}
    n=Node('Mike',graph)

    print(n.get_message('book'))
    print(n.get_senti())
