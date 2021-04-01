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
from torch.utils.tensorboard import SummaryWriter
import os

ra = np.random.randint(0,100000)
print(ra)
writer = SummaryWriter('./log/' + str(ra))


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.GraphConv(in_feats, hid_feats)
        self.conv2 = dglnn.GraphConv(in_feats, hid_feats)        
        self.conv3 = dglnn.GraphConv(in_feats, hid_feats)        

    def forward(self, graph, inputs):

        h = self.conv1(graph, inputs)
        h = torch.relu(h) 
        h = self.conv2(graph, h)
        h = torch.relu(h) 
        h = self.conv3(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.sm = nn.Softmax(dim =1)

    def forward(self, g):
        h = g.ndata['x']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h            
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            
            res = self.sm(self.classify(hg))            
            
            return res

labelDict = ['ast.Module()', "ast.FunctionDef(name='policy', returns=None, type_comment=None)", 'ast.arguments(vararg=None, kwarg=None)', "ast.arg(arg='x1', annotation=None, type_comment=None)", "ast.arg(arg='x2', annotation=None, type_comment=None)", "ast.arg(arg='x3', annotation=None, type_comment=None)", "ast.arg(arg='x4', annotation=None, type_comment=None)", 'ast.Expr()', "ast.Constant(value='\\n    simple policy\\n    ', kind=None)", 'ast.Assign(type_comment=None)', "ast.Name(id='a', ctx=ast.Store())", 'ast.Constant(value=0.3, kind=None)', "ast.Name(id='b', ctx=ast.Store())", 'ast.Constant(value=0.6, kind=None)', "ast.Name(id='result', ctx=ast.Store())", 'ast.Constant(value=0, kind=None)', 'ast.If()', 'ast.BoolOp()', 'ast.And()', 'ast.Compare()', "ast.Name(id='x1', ctx=ast.Load())", "ast.Name(id='a', ctx=ast.Load())", "ast.Name(id='x2', ctx=ast.Load())", 'ast.Lt()', "ast.Name(id='b', ctx=ast.Load())", "ast.Name(id='x3', ctx=ast.Load())", "ast.Name(id='x4', ctx=ast.Load())", 'ast.Constant(value=1, kind=None)', 'ast.Return()', "ast.Name(id='result', ctx=ast.Load())"]

def codeToDgl(filename) : 

    with open(filename, 'r') as f :
        program = f.read()

    node = ast.parse(program)
    node = transformers.ParentChildNodeTransformer().visit(node)  

    visitor = visitors.GraphNodeVisitor()
    visitor.visit(node)
    
    N = nx.nx_pydot.from_pydot(visitor.graph)

    mapping = {}
    features = []

    for i in N._node :         
        lab = N._node[i]['label']

        if lab in labelDict : 
            num = labelDict.index(lab)
        else : 
            labelDict.append(lab)
            num = labelDict.index(lab)
            #print(lab)

        mapping[i] = num        

        #manually defined max of dictinoary !
        features.append(np.eye(80)[num])
        
    #nx.draw_networkx(N, pos=nx.spring_layout(N), labels = mapping, with_labels=True, arrows=True)
    #plt.show()    
    
    g = dgl.from_networkx(N)

    return g, features #also return label here


def getBatch(n = 10) :    
    
    path1 = './dataset/0'
    path2 = './dataset/1'
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)

    graphs = []
    labels = []
    filenames = []

    for i in range(n) :        
        if np.random.rand() <0.5 :
            file = os.path.join(path1, np.random.choice(files1))
            label = np.array([0])
        else : 
            file = os.path.join(path2, np.random.choice(files2))
            label = np.array([1])

        g, features = codeToDgl(file)
        filenames.append(file)

        x = torch.FloatTensor(features)
        g.ndata['x'] = x

        graphs.append(g)        
        labels.append(label)
        
    return graphs, labels, filenames

model = HeteroClassifier(80, 80, 2, '_E')
opt = torch.optim.Adam(model.parameters(), lr=0.001)
lossFunc = nn.CrossEntropyLoss()

for ep in range(10000) : 
    
    g, l, f = getBatch()
    
    graphs = dgl.batch(g)
    logits = model(graphs)  
        
    l = np.squeeze(l)
    labels = torch.tensor(l).long()
    loss = lossFunc(logits, labels)

    writer.add_scalar('main/loss', loss, ep)

    re = torch.argmax(logits, 1)
    acc = ((labels.eq(re.float())).sum()).float()/10
    
    writer.add_scalar('main/acc', acc, ep)


    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 50 == 0 :         
    
        
        fig = plt.figure()                
        N = g[0].to_networkx()
        nx.draw_networkx(N, pos=nx.spring_layout(N))#, labels = mapping, with_labels=True, arrows=True)
        
        writer.add_figure("graph", fig, ep, close=True)
        plt.close()
        
        with open(f[0], 'r') as fi :
            program = fi.read()
                
            program = program.replace('\n', '  \n    ')
            writer.add_text('programs', '    ' + program, ep)
        