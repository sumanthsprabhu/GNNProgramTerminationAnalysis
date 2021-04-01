import ast
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import pydotplus
from astmonkey import visitors, transformers
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import torch.functional as F

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        #self.test = dglnn.GraphConv(in_feats, hid_feats)
        #dglnn.HeteroGraphConv(rel:)

        '''
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        '''

        self.conv1 = dglnn.GraphConv(in_feats, hid_feats)
        self.conv2 = dglnn.GraphConv(in_feats, hid_feats)        
        self.conv3 = dglnn.GraphConv(in_feats, hid_feats)        

    def forward(self, graph, inputs):

        #t = self.test(graph, inputs)

        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        #h = {k: F.relu(v) for k, v in h.items()}
        h = torch.relu(h) #test
        h = self.conv2(graph, h)
        h = torch.relu(h) 
        h = self.conv3(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['x']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)

labels = ['ast.Module()', "ast.FunctionDef(name='policy', returns=None, type_comment=None)", 'ast.arguments(vararg=None, kwarg=None)', "ast.arg(arg='x1', annotation=None, type_comment=None)", "ast.arg(arg='x2', annotation=None, type_comment=None)", "ast.arg(arg='x3', annotation=None, type_comment=None)", "ast.arg(arg='x4', annotation=None, type_comment=None)", 'ast.Expr()', "ast.Constant(value='\\n    simple policy\\n    ', kind=None)", 'ast.Assign(type_comment=None)', "ast.Name(id='a', ctx=ast.Store())", 'ast.Constant(value=0.3, kind=None)', "ast.Name(id='b', ctx=ast.Store())", 'ast.Constant(value=0.6, kind=None)', "ast.Name(id='result', ctx=ast.Store())", 'ast.Constant(value=0, kind=None)', 'ast.If()', 'ast.BoolOp()', 'ast.And()', 'ast.Compare()', "ast.Name(id='x1', ctx=ast.Load())", "ast.Name(id='a', ctx=ast.Load())", "ast.Name(id='x2', ctx=ast.Load())", 'ast.Lt()', "ast.Name(id='b', ctx=ast.Load())", "ast.Name(id='x3', ctx=ast.Load())", "ast.Name(id='x4', ctx=ast.Load())", 'ast.Constant(value=1, kind=None)', 'ast.Return()', "ast.Name(id='result', ctx=ast.Load())"]

def codeToDgl(filename) : 

    with open(filename, 'r') as f :
        program = f.read()

    node = ast.parse(program)
    node = transformers.ParentChildNodeTransformer().visit(node)  

    #exec(program)
    
    visitor = visitors.GraphNodeVisitor()
    visitor.visit(node)
    
    modifiedSource = visitors.to_source(node)    
    
    #with open('result.py', 'w') as f :
    #    f.write(modifiedSource)

    N = nx.nx_pydot.from_pydot(visitor.graph)

    mapping = {}
    features = []

    for i in N._node :         
        lab = N._node[i]['label']

        if lab in labels : 
            num = labels.index(lab)
        else : 
            labels.append(lab)
            num = labels.index(lab)
            print(lab)

        mapping[i] = num        

        #manually defined max of dictinoary !
        features.append(np.eye(40)[num])
        
    #nx.draw_networkx(N, pos=nx.spring_layout(N), labels = mapping, with_labels=True, arrows=True)
    #plt.show()    
    
    g = dgl.from_networkx(N)#, node_attrs = [] )

    x = torch.FloatTensor(features)
    #x = torch.rand(14,3) #give random features instead of actual features
    g.ndata['x'] = x
    test = g.nodes()

    model = HeteroClassifier(40, 40, 2, g.etypes)
    
    for i in range(100) : 
        result = model(g)  #should be batch of graphs
        print(result)


    print('done')

codeToDgl('prog1.py')
