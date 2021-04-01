import ast
import pydotplus
from astmonkey import visitors, transformers
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

labels = ['ast.Module()', "ast.FunctionDef(name='policy', returns=None, type_comment=None)", 'ast.arguments(vararg=None, kwarg=None)', "ast.arg(arg='x1', annotation=None, type_comment=None)", "ast.arg(arg='x2', annotation=None, type_comment=None)", "ast.arg(arg='x3', annotation=None, type_comment=None)", "ast.arg(arg='x4', annotation=None, type_comment=None)", 'ast.Expr()', "ast.Constant(value='\\n    simple policy\\n    ', kind=None)", 'ast.Assign(type_comment=None)', "ast.Name(id='a', ctx=ast.Store())", 'ast.Constant(value=0.3, kind=None)', "ast.Name(id='b', ctx=ast.Store())", 'ast.Constant(value=0.6, kind=None)', "ast.Name(id='result', ctx=ast.Store())", 'ast.Constant(value=0, kind=None)', 'ast.If()', 'ast.BoolOp()', 'ast.And()', 'ast.Compare()', "ast.Name(id='x1', ctx=ast.Load())", "ast.Name(id='a', ctx=ast.Load())", "ast.Name(id='x2', ctx=ast.Load())", 'ast.Lt()', "ast.Name(id='b', ctx=ast.Load())", "ast.Name(id='x3', ctx=ast.Load())", "ast.Name(id='x4', ctx=ast.Load())", 'ast.Constant(value=1, kind=None)', 'ast.Return()', "ast.Name(id='result', ctx=ast.Load())"]

def codeToGraph(filename) : 

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
    
    for i in N._node :         
        lab = N._node[i]['label']

        if lab in labels : 
            num = labels.index(lab)
        else : 
            labels.append(lab)
            num = labels.index(lab)
            print(lab)

        mapping[i] = num
        
        print(np.eye(len(labels))[num])
        
    nx.draw_networkx(N, pos=nx.spring_layout(N), labels = mapping, with_labels=True, arrows=True)
    plt.show()    


codeToGraph('prog1.py')
