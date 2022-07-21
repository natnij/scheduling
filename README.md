##  Mixed integer linear optimization - production planning and resource allocation

### Problem formulation

There are two types of production planning: static and dynamic.

The static plan is created a long time before actual production. It assumes that all prerequisites are met, and given the optimization objectives (such as minimum delay, minimum inventory cost, etc.), the plan is an optimal solution to the problem. 

The dynamic plan is based on static plan, it is created when there are conflicts to the planning a short time before actual production, such as material shortage, unplanned downtime, strikes, etc.

This project is focused on static production planning. More specifically, heuristics and exact algorithms. Applicable optimization problems include 1).tasks with deadlines, 2).limited resource with expiration date (such as machine utilization: the available slots are only valid for the current time period and cannot be preserved), 3).multi-resource, multi-task production planning, 4).optimization objectives are lowest inventory cost and within deadline.

The planned tasks can have a tree-like hierarchical structure: every upper level task can have multiple child tasks(upstream tasks that are prerequisites of the current task), but every lower level task has only one parent(downstream task that is dependent on the current task). For example, a finished product is the root node of the tree, raw materials are leaf nodes and semifinished products are nodes in between.

A very simple example: a and d are raw materials, b,c,g, semifinished goods, and h the final product. In test data, every task has a unique ID.

![alt text](doc/testProductStructure.JPG)

### Algorithms used:

**Branch and Bound**：

Exact algorithms are guaranteed to find the global optimal solution (provided there is one). It can be time-consuming to search the entire solution space. However, there are exact algorithms that are efficient in solving specific types of problems.

Branch and Bound is very often used to solve mixed integer programming (MIP) problems. It creates a tree-like solution space and by redefining the boundaries, prune the tree and reach the optimal solution. In our project, the algorithm estimates a suboptimal but feasible solution as the upper bound, a relaxed (infeasble) solution as lower bound, and divide the problem into layers of subproblems, until it finds the global optimal. Thanks to the upper and lower bounds, most tree nodes would be pruned in the process, thereby accelerating the process.

The core of Branch and bound is how to find the upper and lower bounds. The method we use in this project is Lagrangian relaxation.

**Lagrangian Relaxation**：

A method often used to reduce the complexity in a linear optimization problem.

Some real-life linear optimization problems have huge amount of constraints and adds to the complexity of the algorithm. Lagrangian relaxation turns constraints into relaxed objectives, whose solution is theoretically infinitely closer to the solutions of the original problem. It is therefore often used in finding bounds.

In our project, we use the method to find the lower bound. Based on the method, we apply Lagrangian heuristic to get a feasible solution as upper bound.

P: minimize $c^Tx$

subject to:  

$Ax \geq b$，$Bx \leq d$, $x \in \mathbb{Z}_{+}^n$

Can be converted into the relaxed problem of:

PL: minimize $c^Tx + \lambda^T(b - Ax)$

s.t. $Bx \leq d $，$x \in \mathbb{Z}_{+}^n$

where $\lambda$ is Lagrangian multiplier. We get the multiplier through subgradient methods (the gradient descent for non-derivable functions).

### the case: production planning

**test data**：

![alt text](doc/testProductStructure.JPG)

In test data, we need to produce four finished products, numbered task+[1-4]. The product  hierarchy is constructed through a tree structure with both upper level and lower level task IDs. Machines are numbered mk+[11-16], a machine that is used in upstream processing has a lower number than the downstream machines. The heuristic will start with downstream machines. Other information such as inventory unit cost, order deadline, etc. are as follows:

task|machine|upstream tasks|downstream task|processing time|product number|deadline|inventory unit cost
----|----|----|----|----|----|----|----
a1|mk16||b1|3|h1|50|1
b1|mk15|a1|c1|2|h1|50|2
c1|mk14|b1|g1|5|h1|50|4
d1|mk13||g1|4|h1|50|1
g1|mk12|c1, d1|h1|6|h1|50|8
h1|mk11|g1||1|h1|50|10
a2|mk16||b2|3|h2|60|1
b2|mk15|a2|c2|2|h2|60|2
c2|mk14|b2|g2|5|h2|60|4
d2|mk13||g2|4|h2|60|1
g2|mk12|c2, d2|h2|6|h2|60|8
h2|mk11|g2||1|h2|60|10
a3|mk16||b3|3|h3|65|1
b3|mk15|a3|c3|2|h3|65|2
c3|mk14|b3|g3|5|h3|65|4
d3|mk13||g3|4|h3|65|1
g3|mk12|c3, d3|h3|6|h3|65|8
h3|mk11|g3||1|h3|65|10
a4|mk16||b4|3|h4|70|1
b4|mk15|a4|c4|2|h4|70|2
c4|mk14|b4|g4|5|h4|70|4
d4|mk13||g4|4|h4|70|1
g4|mk12|c4, d4|h4|6|h4|70|8
h4|mk11|g4||1|h4|70|10

**Modeling the original problem**：

i: current task

k: current machine

$\phi(i)$: set of immediate successors

$\varphi(i)$: corresponding final product

$\Lambda(i)$: set of immediate predecessors

Mk: set of tasks to be processed on the current machine

F: set of final products

$p_i$: processing time of i

$d_i$: order deadline of the final product of i

$h_i$: i's unit time inventory cost per product unit

L: a big enough positive number


We want to get the task sequence and each task's start time:

$s_i$: start time of i

$y_{ij}$: 0/1，if 1, means i is processed before j


The original problem is:

$(\mathrm{P}): \text{minimize } z = \sum_{i \notin F} h_i(s_{\phi(i)} - s_i) + \sum_{i \in F} h_i(d_i - s_i)$

s.t.

$s_i + p_i - s_j \leq  L \cdot (1 - y_{ij}) \text{  for } i, j \in M_k(i < j) \text{ and } \forall k$,  (2)

$s_j + p_j - s_i \leq L \cdot y_{ij} \text{  for } i, j \in M_k(i < j) \text{ and } \forall k$,  (3)

$s_i + p_i \leq d_i, \text{for } i \in F$, (4)

$s_i + p_i \leq s_{\phi(i)}, \text{for } i \notin F$, (5)

$s_i \geq 0, \forall i$, (6)

$y_{ij} = {0,1} \text{ for } i,j \notin M_k(i < j) \text{ and } \forall k$. (7)


We define the echelon inventory and simplify (P) into: 

$(\mathrm{P}): \text{minimize } z = \sum_{i} e_i(d_{\varphi(i)} - s_i)$ (8)

where

$e_i \equiv h_i - \sum_{j \in \Lambda(i)} h_j, \forall i$, 

$d_{\varphi(i)}$ is i's final product's deadline


**Convert the original problem into a relaxed problem**, then decompose the problem into single-machine, multi-task relaxed problems:

By relaxing (4) and (5) we get:

$(\mathrm{LR}_\lambda): \text{ minimize } \sum_{\forall i} ( \lambda_i - e_i - \sum_{j \in \Lambda(i)} \lambda_j ) s_i + \sum_{\forall i} ( e_i d_{\varphi(i)} + \lambda_i p_i ) - \sum_{i \in F} \lambda_i d_i$ （9）

s.t. (2), (3), (6), (7), and $\lambda_i \geq 0 \forall i$.

$L(\lambda)$ denotes the optimal solution to the relaxed problem $(\mathrm{LR}_\lambda)$, or the lower bound to the original problem$(\mathrm{P})$.

Given a random set of $\lambda_n$, we can solve(9) and get a solution $L(\lambda_n)$. Therefore the relaxed problem $(\mathrm{LR}_\lambda)$ is the equivalent of solving the dual problem of $(\mathrm{LR}_\lambda)$:

$(\mathrm{PL}): \text{ maximize } L(\lambda_n)$ s.t. $\lambda \geq 0$

Next we decompose the dual problem $(\mathrm{PL})$, and get $k$ independent single-machine multi-task planning problem:

$(\mathrm{DP}_k): \text{ minimize } \sum_{i \in M_k} ( \lambda_i - e_i - \sum_{j \in \Lambda(i)} \lambda_j ) s_i$ (10)

s.t.

$s_i + p_i - s_j \leq L \cdot (1 - y_{ij}, \forall i, j \in M_k(i < j)$, (2')

$s_j + p_j - s_i \leq L \cdot y_{ij}, \forall i, j \in M_k(i < j)$, (3')

$y_{ij} = {0,1}, \forall i, j \in M_k(i < j)$, (7')

$\lambda_i \geq 0, \forall i \in M_k$ (11)

$s_i \geq l_k, \forall i \in M_k$, (12)

$s_i + p_i \leq u_k, \forall i \in M_k$. (13)

With given $\lambda$, the relaxed problem $(\mathrm{LR}_\lambda)$'s second and third terms are constants and can be omitted in the subproblems $(\mathrm{DP}_k)$. We use $L_k(\lambda)$ to denote solutions to $(\mathrm{DP}_k)$. 

$L(\lambda) = \sum_{k=1}^{K} L_k(\lambda) + \sum_i(e_i d_{\varphi(i)} + \lambda_i p_i) - \sum_{i \in F} \lambda_i d_i$. 

To calculate the approx. solution to the subproblems, we use [GWSPT](https://pdfs.semanticscholar.org/1f93/f3da32b66134b5bc040692c76ca2a888680c.pdf) algorithm. The weights to $(\mathrm{DP}_k)$ are:

$w_i =  ( \lambda_i - e_i - \sum_{j \in \Lambda(i)} \lambda_j )$

It is proven in the Authors' original paper that the task sequence is optimal when it is ordered in descending order of $w/p$, or:

$y_{ij} = 1 \text{ and } y_{ji} = 0 \text{ if } \cfrac{w_i}{p_i} \geq \cfrac{w_j}{p_j}$

According to the sign of $w_i$, machine $k$'s set of tasks $M_k$ can be divided into 3 subsets:

$M_k^{+} = {i: w_i > 0 \text{ and } i \in M_k}$

$M_k^{0} = {i: w_i = 0 \text{ and } i \in M_k}$

$M_k^{-} = {i: w_i < 0 \text{ and } i \in M_k}$

The paper proved that if:

$M_k^{+}$'s task sequence is in $[ l_k, l_k + \sum_{i \in M_k^{+}} p_i ]$

$M_k^{0}$'s task sequence is in $[  l_k + \sum_{i \in M_k^{+}} p_i,  u_k - \sum_{i \in M_k^{-}} p_i ]$, and

$M_k^{-}$'s task sequence is in $[ u_k - \sum_{i \in M_k^{-}} p_i, u_k ]$, 

and ordered according to GWSPT, respectively, then $(\mathrm{DP}_k)$ has an optimal solution. To calculate $l_k, u_k$:

$u_k = max_{i \in M_k}[d_{\varphi(i)} - \sum_{j \in \Phi(i)} p_j ]$, where $\Phi(i)$ is a set that contains task $i$ and all of its downstream tasks.

$l_k = min_{i \in M_k} [min_{j \in \Psi(i)}(\sum_{l \in \Theta(i,j)} p_l - p_i )]$ where $\Psi(i)$ contains all the leaf nodes (raw material) in task $i$'s upstream, $\Theta(i,j)$ contains all other nodes in $i$'s upstream.

### Implementation and sample code

GWSPT sample code:

```python
def doGWSPT(df, lk, uk, plusTotal, minusTotal):
    """ generalized weighted shortest processing time (GWSPT) sequence 
        based on weight/processTime. proposition 1 + corollary 1.       
    """
    mkGroup = df['mkGroup'].tolist()[0]
    df = df.sort_values(by='wOverp', ascending=False)
    sIdx = df.columns.tolist().index('startTime')
    pIdx = df.columns.tolist().index('processTime')
    if mkGroup == 'plus':
        startPoint = lk
    elif mkGroup == 'zero':
        startPoint = lk + plusTotal
    else:
        startPoint = uk - minusTotal
        
    df.iloc[0, sIdx] = startPoint
    p = df.iloc[0, pIdx]
    for i in np.arange(df.shape[0] - 1):
        df.iloc[i+1, sIdx] = startPoint + p
        p += df.iloc[i+1, pIdx]
    return df
```

**Calculate subgradients**：

Given $\lambda_i^0$, the nth iteration is

$\lambda_i^{n+1} = 
      \begin{cases} 
      max\{0, \lambda_i^n + t_n(s_i^n + p_i - d_i)\} & \quad \forall i \in F, \\
      max\{0, \lambda_i^n + t_n(s_i^n + P_i - s_{\phi(i)}^n)\} & \quad \forall i \notin F,
      \end{cases}$
  
where $(s_1^n, s_2^n, \dots s_l^n)$ is an optimal solution to the relaxed problem $(\mathrm{LR}_\lambda)$ given $\lambda^n$.

step $t_n = \cfrac{\mu_n(z^{*} - L(\lambda^n))}{\sum_{i \in F}(s_i^n + p_i - d+i)^2 + \sum_{i \notin F}(s_i^n + p_i - s_{\phi(i)}^n)^2}$

$\mu_n \in (0, 2]$, reduced as we get closer to the optimal solution, to improve convergence property.

$z^{*}$ is an upper bound to the (PL) problem, $L(\lambda^n)$ is the lower bound.

We also set maximum iteration to $\omega$, stop criteria $\epsilon$: when $(z^{*} - L(\lambda^n) / L(\lambda^n) < \epsilon$ the iteration is stopped.


Sample code:

```python
def updateSubgradient(waitingList, zStar, L_lambda):
    """ find best lambda value iteratively.
        input:
            zStar: upper bound on the optimal solution value.
            L(lambda): solution value to LR(lambda) given lambda.
        output:
            updated lambda value stored in waitingList as lambdaIter.
    """
    tn = calculateTn(zStar, L_lambda)
    f = waitingList[waitingList['immSuc']=='finalProduct']
    nf = waitingList[waitingList['immSuc']!='finalProduct']
    
    f['lambdaIter'] = f['lambdaIter'] + tn * (f['startTime'] 
                        + f['processTime'] - f['dueDate'])
    nf['lambdaIter'] = nf['lambdaIter'] + tn * (nf['startTime']
                        + nf['processTime'] - nf['s_phi'])   
    f['lambdaIter'] = np.where(f['lambdaIter'] < 0, 0, f['lambdaIter'])
    nf['lambdaIter'] = np.where(nf['lambdaIter'] < 0, 0, nf['lambdaIter'])
    
    waitingList = pd.concat([f, nf], axis=0)
    return waitingList

def calculateTn(waitingList, zStar, L, mu, ):
    """ learning rate in the subgradient method. """
    numerator = mu * (zStar - L)
    f = waitingList[waitingList.immSuc == 'finalProduct']
    f = np.power(f.startTime + f.processTime - f.dueDate, 2)
    nf = waitingList[waitingList.immSuc != 'finalProduct']
    nf = np.power(nf.startTime + nf.processTime - nf.s_phi, 2)
    
    tn = numerator / (np.sum(f) + np.sum(nf))
    return tn
```

**Calculate upper bound** $z^{*}$ ：

Pseudo code for the Lagrangian heuristic:

- on machine $k$, define $M_k$'s initial task sequence $\rho_k$
- According to the previously calculated lower bound, calculate start times:

    - $\Gamma(i) = \{j: y_{ij} = 1\}, \forall i \in M_k$ is the set of all tasks on $k$ that starts later than task $i$ 
    - set $i$'s finish time to $d_i = s_{\phi(i)}, \forall i \in M_k \text{\F}$ 
    - set $i$'s start time to $s_i = min \{d_i, min_{j \in \Gamma(i)} s_j \} - p_i, \forall i \in M_k$
- beginning from M_k's latest task $l$, search all tasks (denoted $i$) in $M_k$:
    - if i's finish time $d_i > d_l + p_i$: $i$ is the last task on $M_k$
    - else: from $l$, search all tasks $h$ between $l$ and $i$, if $s_h - (s_{h-1} + p_{h-1}) \geq p_i$ and $(s_{h-1} + p_{h-1}) + p_i \leq d_i$, insert $i$ between $h$ and $h-1$

Sample code to calculate $s_{\phi(i)}$:

```python
def updateSphi(waitingList, _tbl):
    """ s_phi(i) is the start time of phi(i).
        phi(i): immediate successor of part i. 
        Be ware - this method will change the table index. 
    """
    tbl = _tbl.copy()
    try: 
        tbl.drop('s_phi', axis=1, inplace=True)
    except ValueError: 
        pass
    s = waitingList.loc[:, ['parts', 'startTime']]
    s.columns = ['immSuc', 's_phi']
    tbl = tbl.merge(s, how='left', on='immSuc')
    tbl.loc[tbl['s_phi'].isnull(), 's_phi'] = (
       tbl.loc[tbl['s_phi'].isnull(), 'dueDate'])
    return tbl
```    

Sample code to calculate $min_{j \in \Gamma(i)} s_j$:

```python
def calculateMinGammaSj(_mkTbl, sortedAsc=False):
    """ calculates minimum start time among tasks j, which 
        belong to set Gamma(i), as basis to the calcultion of start times
        si = min(s_(phi_i), min_(j in Gamma(i)) sj) - pi
        input:
            _mkTbl: table of parts to be sequenced on the same 
                equipment mk
            sortedAsc: if the parts are already sorted in ascending 
                order by start time. If False, then sort. If True, 
                then keep the original to save time.
    """
    if sortedAsc is False:
        mkTbl = _mkTbl.sort_values(by='startTime', ascending=True)
    else:
        mkTbl = _mkTbl.copy()
    sj = mkTbl['startTime'].tolist()
    sj.pop(0)
    sj.append(99999)
    mkTbl['minGammaSj'] = sj
    return mkTbl
```

**Branch and bound**：

The pseudo code for node selection:

- given $m_k$, random tasks on $m_k$ as row, fixed downstream planning set S, still open upstream planning set P', and the upper bound:

createNode:
 - add row to set S
 - delete row from set P'
 - set P' as input to the Lagrangian heuristic, get new upper and lower bounds
    
The pseudo code for branch-out:

- given initial feasible solution from Lagrangian heuristic $z^{*}$
- given upper bound
- given list of machines, ordered in ascending order

branchOut:
 - Add the task with the lastest start time in the most downstream machine as root node to empty set S, according to $z^{*}$
 - delete the root node task from set P'
 - assume the current machine $m_k$ is the first element in the list of machines
 - iterate until upper bound does not improve, or maximum iteration reached:
     - $s_k$ is the set of all unplanned tasks on $m_k$
     - if len($s_k$)==1:
         - add start time to $s_k$
         - add $s_k$ to set S
         - delete $s_k$ from set P'
     - if len($s_k$)==0:
         - go to the next machine in the list
     - else:
         - use every task in $s_k$ as input, execute createNode, get solution and upper/lower bounds
         - if a feasible solution does not exist, or new lower bound is worse than existing upper bound, then delete node
         - else:
           - if new upper bound is improved, use the new upper bound
     - use the set S and set P' of the node with the most optimal lower bound
  
### Test result:

KPI|lagrangian relaxation|lagrangian relaxation and heuristic|lagrangian relaxation and heuristic and BB
-----|-------------|------------------|--------------
inventory holding cost|520|500|460
average throughput|22.5|21.75|20.25
average machine utilization rate|50%|50%|54%

Using branch-and-bound gives better results than using only the Lagrangian relaxation, or the heuristic. 
