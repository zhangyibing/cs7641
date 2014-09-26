# This code was pulled from a tutorial on https://triangleinequality.wordpress.com
# 

import tigraphs as tig
import numpy as np
import pandas as pd

class DecisionNode(tig.BasicNode,object):
    def __init__(self, **kwargs):
        super(DecisionNode, self).__init__(**kwargs)
        self.left=None
        self.right=None
        self.children=None
        self.parent=None
        self.prediction=None
        self.tally={}
        self.total=0.0
        self.size=0
        self.depth =0
        self.local_data=None
    def local_filter(self, data): #filters data
        pass first = False)
    def get_next_node_or_predict(self, datapoint):
        pass

class DecisionTree(tig.return_nary_tree_class(directed=True), object):
    def __init__(self, data=None,response='',Vertex=DecisionNode,**kwargs):
        super(DecisionTree, self).__init__(N=2, Vertex=Vertex, **kwargs)
        self.data =data
        self.response=response #data attribute we're trying to predict
    def split_vertex(self, vertex):
        super(DecisionTree, self).split_vertex(vertex)
        vertex.left = vertex.children[0]
        vertex.right = vertex.children[1]
    def fuse_vertex(self, vertex):
        super(DecisionTree, self).fuse_vertex(vertex)
        vertex.left, vertex.right = None, None

class PivotDecisionNode(DecisionNode,object):
    def __init__(self, **kwargs):
        super(PivotDecisionNode, self).__init__(**kwargs)
        self.pivot=None
        self.split_attribute = None
    def local_filter(self, data): #filters the data based on parent's pivot
        if self.parent==None:
            self.size = len(data)
            return data
        attribute = self.parent.split_attribute
        pivot = self.parent.pivot
        if type(pivot)==set:
            ret= data[attribute].isin(pivot)
        else:
            ret = data[attribute] &lt;= pivot
        if self == self.parent.left:
            ret=data[ret]
        else:
            ret=data[~ret]
        self.size=len(ret)
        return ret
    def get_next_node_or_predict(self, datapoint): #tells us where to find a prediction, or returns one
        if self.children == None:
            return self.prediction
        else:
            if type(self.pivot) ==set:
                if datapoint[self.split_attribute] in self.pivot:
                    return self.left
                else:
                    return self.right
            else:
                if datapoint[self.split_attribute] &lt;=self.pivot:
                    return self.left
                else:
                    return self.right



#Test Code Here
t=PivotDecisionTree()
t.create_vertex()
t.set_root(t.vertices[0])
root = t.get_root()
t.leaves.add(root)
t.split_vertex(vertex=t.get_root(), split_attribute='sex', pivot=set(['female']))
import cleantitanic as ct
data = ct.cleaneddf()[0]
t.response='survived'
root = t.get_root()
root.local_data=root.local_filter(data)
for child in root.children:
    child.local_data=child.local_filter(root.local_data)
t.set_predictions()
for leaf in t.leaves:
    print leaf.prediction