# GNNProgramTerminationAnalysis
Program Termination Analysis using Graph Neural Network

A binary classifier for program termination analysis using graph neural networks. 

## Steps 
create generic programs, feed them into a GNN, and then train it.

* Generation of random python programs and labelling regarding their ability to terminate.
  * Generator that takes the indent into account
  * then exec tests if the program is executable
  * Executing in a thread and measuring of time gives an estimation if they are endless.
  * The result label and the program source are added to the dataset.
* Conversion of program to AST graph 
  * AST is build and converted to networkx
  * from networkx conversion to DGL heterogenous graph.
  * The command at each node is converted to key from command dictionary and each node gets a one-hot encoding like 'if' would be [0,0,0,1,0,0...]
* A batch of graphs is feed into a GNN with a softmax classifier.
* Message forwarding is performed, and the classification is optimized.

Example of AST output with feature key in node:
![Intro](/imgs/Selection_260.png)
Results for a training time of approximatley an hour:
![Intro](/imgs/Selection_265.png)
![Intro](/imgs/Selection_262.png)
Example programs from our dataset:
![Intro](/imgs/Selection_263.png)

