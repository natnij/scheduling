# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:20:31 2017

@author: Nat
"""

import pandas as pd
import numpy as np
import sys

def preprocess(filename='testdata.csv'):
    data = pd.read_csv(filename)
    data.fillna(0, inplace=True)
    
    def formatierung(row):
        colnames = row.index.tolist()
        p = colnames.index('immPre')
        s = colnames.index('immSuc')
        row[p] = [x.strip() for x in str(row[p]).split(',')]
        row[s] = str(row[s])  # assuming there is only one immediate successor
        if row[p][0] == '0':
            row[p] = 'rawMaterial'
        if row[s][0] == '0':
            row[s] = 'finalProduct'
        return row
    data = data.apply(lambda row: formatierung(row), axis=1)
    return data

def calculateKPI(solution, nrIter=0,
                 finalProdCol='originalFinalProduct',
                 kpi_invCost=pd.DataFrame(), kpi_makespan=pd.DataFrame(),
                 kpi_utilization=pd.DataFrame()):
    """ calculates KPIs such as maximum makespan, total inventory
        holding cost, equipment utilization rate, etc.
    """
    invCost = solution.loc[:, ['parts','startTime','s_phi','invHolding']]
    invCost['invCost'] = ( (invCost['s_phi'] - invCost['startTime'])
                            * invCost['invHolding'])
    invCost['iteration'] = nrIter
    kpi_invCost = pd.concat([kpi_invCost, invCost], axis=0)
    
    def minMax(df):
        finalProduct = df[finalProdCol].tolist()[0]
        start = np.min(df['startTime'])
        end = np.max(df['finishTime'])
        makespan = end - start
        return pd.DataFrame([[finalProduct, start, end, makespan]],
                        columns=[finalProdCol, 'start', 'end', 'makespan'])
    
    makespan = solution.loc[:, [finalProdCol, 
                                'startTime', 'processTime']]
    makespan['finishTime'] = makespan['startTime']+makespan['processTime']
    makespan = makespan.groupby(finalProdCol,
                    as_index=False).apply(minMax).reset_index(drop=True)
    makespan['iteration'] = nrIter
    kpi_makespan = pd.concat([kpi_makespan, makespan])
    
    def utilRate(df):
        equipment = df['equipment'].tolist()[0]
        start = np.min(df['startTime'])
        end = np.max(df['finishTime'])
        span = end - start
        utilized = np.sum(df['processTime'])
        rate = utilized / span
        return pd.DataFrame([[equipment,start,end,span,utilized,rate]],
                            columns=['equipment','start','end','span',
                                     'utilized','rate'])
    
    utilization = solution.loc[:, ['equipment','startTime','processTime']]
    utilization['finishTime'] = (utilization['startTime'] 
                                    + utilization['processTime'])
    utilization = utilization.groupby('equipment',
                    as_index=False).apply(utilRate).reset_index(drop=True)
    utilization['iteration'] = nrIter
    kpi_utilization = pd.concat([kpi_utilization, utilization])        
    return kpi_invCost, kpi_makespan, kpi_utilization

class Lagrangian():
    def __init__(self, waitingList=pd.DataFrame(), 
                 lambdaZero=0, epsilon=0.0001,
                 muZero=2, zeta=10, omega=1000,
                 useHeuristicOnce=False, zStarRef=None):
        """ reference:
            https://pdfs.semanticscholar.org
                            /1f93/f3da32b66134b5bc040692c76ca2a888680c.pdf
            inputs:
                waitingList: parts list to be scheduled on machines
                lambdaZero: initial values for lagrange vector
                epsilon: relative error threshold as criteria to terminate 
                    subgradient method for finding optimal lambda.
                omega: maximum iteration threshold as criteria to terminate
                    subgradient method for finding optimal lambda.
                mu: scalar in (0, 2] to determine step size in sub-
                    gradient method. 
                muZero: initial value.
                zeta: number of iterations (defines how quickly to reduce mu)
                useHeuristicOnce: if True then execute findUpperBound() method
                    only once at the end with final result from the 
                    subgradient method to save computation time. This is used 
                    in the child nodes of branch-and-bound algorithm. if False
                    then execute to find better upper bound in every iteration.
                zStarRef: if useHeuristicOnce is set to True, then a zStar
                    value as reference must be given. This is usually from 
                    previous runs of the heuristic.
            variables:
                L_lambda: solution value to problem PL. L_lambda values are 
                    iteratively calculated and the real solution value 
                    is the maximum of all L_lambda.
                Lk_lambda: upper bound of solution values to 
                    independent problems DPk, calculated first in method
                    runOptimalSolution() then again in method findUpperBound(). 
                sigmaThetaTbl: a table recording length of every task's 
                    chained predecessors' process time in total. It is 
                    populated with (i, j, sum-of-processTime-of-l)
                    for all j in set Psi(i) and l in set Theta(i, j):
                    set Psi(i) contains ALL predecessors of item i which 
                    do not have any predecessor(raw material). 
                    Theta(i, j) is the set of  all items on the path 
                    connecting items i and j in the product structure 
                    (including both i and j). however the sum
                    of processTime should exclude pi.
                muCounter: if number of iterations in which L(lambda) has 
                    not improved exceeds given threshold zeta, mu will be
                    updated in method updateMu. mCounter counts the continuous
                    number of such iterations. it is reset whenever there is
                    update to self.PL_solutionValue.
                zStar: upper bound to the solution value L(lambda).
                PL_solutionValue: maximum of L(lambda), solution value to 
                    the problem (PL).
                feasibility: True if sum of process time on equipment mk is 
                    smaller than the equipment's uk - lk. If not feasible,
                    then the process is killed with the flag set to False.
            outputs: 
                optimalParameters: result from the lagrangian heuristic, 
                    including the following fields:
                parts: tasks / parts scheduled
                equipment: machine index on which parts are scheduled
                immPre: list of immediate predecessors of the part 
                    in product hierarchy. If no successors, then 'rawMaterial' 
                immSuc: immediate successsor of the part in product hierarchy.
                    If no successor, then 'finalProduct'
                processTime: time units for processing of the part
                finalProduct: index of the final product in product
                    hierarchy.
                dueDate: dueDate of the final product.
                invHolding: inventory holding cost.
                startTime: time of start, the final outcome of the algorithm.
                s_phi: start time of the immediate successor phi(i)
                lambdaIter: lambda value iteratively found
                sigmaLambda_hj: sum of inventory holding cost of j, j being
                    all tasks in the set Lambda(i), which is the set of all 
                    immediate predecessors of i in the product hierarchy. 
                    this is used to calculate echelon. 
                echelon: echelon holding cost ei = hi - sum of hj in Lambda(i).
                stack: storage of process times of all items in Phi(i). This
                    is used in method calculateSigmaThetaPl() to populate 
                    self.sigmaThetaTbl, as utility table to calculate lb in 
                    method findMinPsiPj().
                sigmaPhi_pj: sum of process times of j, j being all tasks in 
                    the set Phi(i), which is the set with all successors of i
                    in the product hierarchy. This is used to calculate upper
                    bound uk.
                sigmaLambda_lambdaj: sum of lambdas in the set Lambda(i), 
                    which is the set of all immediate predecessors of i in 
                    the product hierarchy.
                lb: lower bound calculated with set Psi(i) and set Theta(i, j):
                    Psi(i) contains all i's predecessors which are also 
                    rawMaterial; Theta(i, j) contains all items on the path 
                    connecting i and j, including i and j themselves. the 
                    lb here is the earliest-starting branch of the product
                    chain from i. lk later is the earliest of the lb's among
                    all i tasks. 
                ub: upper bound calculated with set set Phi(i). 
                uk: upper bound of the machine k to schedule a task. 
                lk: lower bound of the machine k to schedule a task. 
                    feasibility is true if sum of the process time of 
                    all tasks on machine k is smaller than uk - lk.
                feasible: see above.
                weight: weight by processing time, 
                    w = lambda - e - sigmaLabmda_lambdaj as in equation 10.
                wOverp: wi / pi, determines scheduling sequence. 
                mkGroup: either mkPlus, mkMinus or mkZero. determines 
                    roughly where to schedule a task. 
                DPkValue: solution value to the problems (DPk), basis to 
                    calculate L(lambda). 
                secondTerm: in equation to calculate L(lambda), the first 
                    term is sum of Lk(lambda), second term and third term
                    are constants when lambda is given. Here it's calculated
                    and stored.
                thirdTerm: third term in the equation to calculate L(lambda).
                minGammaSj: minimum of the start times among j tasks, j being
                    all tasks in the set Gamma(i), which is a set containing 
                    all tasks scheduled on the same machine as i but with 
                    later start times. This term is used to calculate 
                    upper bound to solution value of (PL) in method 
                    findUpperBound(). 
        """
        self.waitingList = waitingList.copy()
        self.numItems = waitingList.shape[0]
        self.lambdaZero = lambdaZero
        self.L_lambda = 0
        self.Lk_lambda = None
        self.L_lambda_tracker = []
        self.epsilon = epsilon
        self.mu = muZero
        self.muCounter = 0
        self.zeta = zeta
        self.omega = omega
        self.sigmaThetaTbl = pd.DataFrame()
        self.PL_solutionValue = 0
        self.PL_solutionValue_tracker = []
        self.zStar = zStarRef
        self.zStar_tracker = []
        self.optimalParameters = pd.DataFrame()
        self.feasibility = True
        self.useHeuristicOnce = useHeuristicOnce   
        
        self.loadWaitingList()

    def loadWaitingList(self):
        """ initialize waiting list. 
        >>> df = pd.DataFrame({'parts':['a1','b1','c1','d1','g1','h1'], 
        ... 'equipment': ['mk6','mk5','mk4','mk3','mk2','mk1'], 
        ... 'immPre': ['rawMaterial','a1','b1','rawMaterial',['c1','d1'],'g1'],
        ... 'immSuc': ['b1','c1','g1','g1','h1','finalProduct'],
        ... 'processTime': [3,2,5,4,6,1],
        ... 'finalProduct': ['h1','h1','h1','h1','h1','h1'],
        ... 'dueDate': [50,50,50,50,50,50],
        ... 'invHolding': [1,2,4,1,8,10],
        ... 'startTime': [0,0,0,0,0,0]})
        >>> la = Lagrangian(waitingList=df)
        >>> np.sum(la.waitingList.sigmaLambda_hj)
        16
        >>> np.sum(la.waitingList.echelon)
        10
        >>> la.waitingList.loc[la.waitingList.parts=='b1', 'stack'].tolist()[0]
        [('b1', 2), ('c1', 5), ('g1', 6), ('h1', 1)]
        >>> np.sum(la.waitingList.sigmaPhi_pj)
        62
        >>> la.waitingList.loc[la.waitingList.parts=='g1', 'lb'].tolist()[0]
        4
        >>> la.waitingList.loc[la.waitingList.parts=='g1', 'ub'].tolist()[0]
        43
        >>> la.waitingList.loc[la.waitingList.parts=='h1', 's_phi'].tolist()[0]
        50.0
        """
        if self.waitingList.shape[0] == 0:
            self.waitingList = preprocess()
        
        self.waitingList['lambdaIter'] = self.lambdaZero
        self.waitingList = self.updateSphi(self.waitingList)
        self.waitingList['sigmaLambda_hj'] = self.waitingList.apply(lambda row: 
                                    self.calculateSigmaLambdaHj(row), axis=1)
        self.waitingList['echelon'] = (self.waitingList['invHolding'] 
                                        - self.waitingList['sigmaLambda_hj'])
        self.waitingList['echelon'] = np.where(self.waitingList['echelon'] < 0,
                                        0, self.waitingList['echelon'])
        self.waitingList['stack'] = self.waitingList.apply(lambda row:
                        self.findPhi(row['parts'], list()), axis=1)
        self.waitingList['sigmaPhi_pj'] = self.waitingList.apply(lambda row:
                                    self.findSigmaPhiPj(row['stack']), axis=1)
        self.calculateSigmaThetaPl()
        self.waitingList['lb'] = self.waitingList.apply(lambda row:
                                    self.findMinPsiPj(row['parts']), axis=1)
        self.waitingList['ub'] = (self.waitingList['dueDate'] 
                                        - self.waitingList['sigmaPhi_pj'])
    
    def updateSphi(self, _tbl):
        """ s_phi(i) is the start time of phi(i).
            phi(i): immediate successor of part i. 
            Be ware - this method will change the table index. 
        """
        tbl = _tbl.copy()
        try: 
            tbl.drop('s_phi', axis=1, inplace=True)
        except ValueError: 
            pass
        s = self.waitingList.loc[:, ['parts', 'startTime']]
        s.columns = ['immSuc', 's_phi']
        tbl = tbl.merge(s, how='left', on='immSuc')
        tbl.loc[tbl['s_phi'].isnull(), 's_phi'] = (
           tbl.loc[tbl['s_phi'].isnull(), 'dueDate'])
        return tbl
    
    def calculateSigmaLambdaHj(self, row):
        """ calculate cumulative holding cost for all of i's 
            immediate predecessors as basis for calculating 
            echelon inventory holding cost:
            echelon_i = h_i - sigma_(j in Lambda_i) h_j
            where Lambda_i (with capital Lambda) is the set of i's
            immediate predecessors in the product hierarchy.
        """
        immPre = row['immPre']
        if immPre == 'rawMaterial':
            return 0
        if not isinstance(immPre, list):
            immPre = [immPre]
        hTotal = 0
        for pre in immPre:
            hTotal += self.waitingList.loc[self.waitingList.parts == pre, 
                                               'invHolding'].tolist()[0]
        return hTotal
    
    def findPhi(self, product, stack=[]):
        """ recursively find the sum of process time of all j in set Phi(i).
            set Phi(i) contains ALL successors of item i in the 
            product hierarchy (regardless of the equipment where
            they are produced), including i itself.
            
            stack: storage of process times of all items in Phi(i). This
                is used in method calculateSigmaThetaPl() to populate 
                self.sigmaThetaTbl, as utility table to calculate lb in 
                method findMinPsiPj().
        """
        row = self.waitingList.loc[self.waitingList.parts == product, :]
        immSuc = row['immSuc'].tolist()[0]
        processTime = row['processTime'].tolist()[0]
        finalProd = row['finalProduct'].tolist()[0]
        stack.append((product, processTime))
        if product != finalProd:
            self.findPhi(immSuc, stack)
        return stack
    
    def findSigmaPhiPj(self, stack):
        Phi_pj = [x[1] for x in stack]
        sigmaPhi_pj = np.sum(Phi_pj)
        return sigmaPhi_pj
    
    def calculateSigmaThetaPl(self):
        """ populates sigmaThetaTbl with tuples 
            (i, j, sum-of-processTime-of-l)
            for all j in set Psi(i) and l in set Theta(i, j):
            set Psi(i) contains ALL predecessors of item i which do not have
            any predecessor(raw material). Theta(i, j) is the set of 
            all items on the path connecting items i and j in 
            the product structure (including both i and j). however the sum
            of processTime should exclude pi.
        """
        try:
            tmpList = self.waitingList.loc[
                                    self.waitingList.immPre=='rawMaterial',
                                    'stack'].tolist()
        except KeyError:
            tmp = self.waitingList.apply(lambda row:
                                    self.findSigmaPhiPj(row['parts']), axis=1)
            tmp.columns = ['sigmaPhi_pj', 'stack']
            self.waitingList = pd.concat(
                    [self.waitingList.reset_index(drop=True), tmp], axis=1)
            tmpList = self.waitingList.loc[
                                    self.waitingList.immPre=='rawMaterial',
                                    'stack'].tolist()
                
        for i in np.arange(len(tmpList)):
            tmp = tmpList[i]
            rm = tmp[0][0]  # raw material index
            sigmaP = 0  # cumulative process time of all tasks between i and j
            for j in np.arange(len(tmp) - 1):
                sigmaP += tmp[j][1]
                tmpTheta = pd.DataFrame([[tmp[j+1][0], rm, sigmaP]])
                tmpTheta.columns = ['i', 'j', 'sigmaThetaPl']
                self.sigmaThetaTbl = pd.concat([self.sigmaThetaTbl, tmpTheta], 
                                               axis=0)
    
    def findMinPsiPj(self, product):
        """ recursively find the min of process time of all j in set Psi(i).
            set Psi(i) contains ALL predecessors of item i that do not have
            any predecessor(raw material). Theta(i, j) is the set of 
            all items on the path connecting items i and j in 
            the product structure.
        """
        if self.sigmaThetaTbl.shape[0] == 0:
            self.calculateSigmaThetaPl()
        Psi_i = self.sigmaThetaTbl[self.sigmaThetaTbl['i'] == product]
        if Psi_i.shape[0] == 0:  # rawMaterial node without further predecessor
            return 0
        else:
            # looser lower bound constraints makes solution to (DPk) less 
            # likely to be a feasible solution to the original problem (P); 
            # however it's quicker in overall solution procedure. 
            # therefore taking the min. - shortest path before i as lower 
            # bound instead of the max. - longest path before i.
            return np.min(Psi_i['sigmaThetaPl'])

    def updateMu(self):
        """ placeholder for updating learning rate of the subgradient. 
        default is by cutting mu by half if LR_lambda for any given lambda^n
        has not increased within self.zeta number of iterations.
        """
        self.mu = self.mu / 2.0
    
    def calculateTn(self, zStar, L):
        """ learning rate in the subgradient method.
        """
        numerator = self.mu * (zStar - L)
        f = self.waitingList[self.waitingList.immSuc == 'finalProduct']
        f = np.power(f.startTime + f.processTime - f.dueDate, 2)
        nf = self.waitingList[self.waitingList.immSuc != 'finalProduct']
        nf = np.power(nf.startTime + nf.processTime - nf.s_phi, 2)
        
        tn = numerator / (np.sum(f) + np.sum(nf))
        return tn
    
    def updateSubgradient(self):
        """ find best lambda value iteratively.
            input:
                zStar: upper bound on the optimal solution value.
                L(lambda): solution value to LR(lambda) given lambda.
            output:
                updated lambda value stored in waitingList as lambdaIter.
        """
        tn = self.calculateTn(self.zStar, self.L_lambda)
        f = self.waitingList[self.waitingList['immSuc']=='finalProduct']
        nf = self.waitingList[self.waitingList['immSuc']!='finalProduct']
        
        f['lambdaIter'] = f['lambdaIter'] + tn * (f['startTime'] 
                            + f['processTime'] - f['dueDate'])
        nf['lambdaIter'] = nf['lambdaIter'] + tn * (nf['startTime']
                            + nf['processTime'] - nf['s_phi'])   
        f['lambdaIter'] = np.where(f['lambdaIter'] < 0, 0, f['lambdaIter'])
        nf['lambdaIter'] = np.where(nf['lambdaIter'] < 0, 0, nf['lambdaIter'])
        
        self.waitingList = pd.concat([f, nf], axis=0)
    
    def calculateSigmaLambda_lambdaj(self, row):
        """ calculate sum of lambda j for all j in Lambda_i, 
            Lambda_i is the set of i's immediate predecessors in 
            the product hierarchy.
        """
        immPre = row['immPre']
        if immPre == 'rawMaterial':
            return 0
        if not isinstance(immPre, list):
            immPre = [immPre]
        lambdaTotal = 0
        for pre in immPre:
            lambdaTotal += self.waitingList.loc[self.waitingList.parts == pre, 
                                                'lambdaIter'].tolist()[0]
        return lambdaTotal
    
    def doGWSPT(self, df, lk, uk, plusTotal, minusTotal):
        """ generalized weighted shortest processing time (GWSPT) sequence 
            based on weight/processTime. proposition 1 + corollary 1.       
        
        >>> df = pd.DataFrame({'parts':['g1','g2','g3','g4'], 
        ... 'equipment': ['mk2','mk2','mk2','mk2'], 
        ... 'immPre': [['c1','d1'],['c1','d1'],['c1','d1'],['c1','d1']],
        ... 'immSuc': ['h1','h2','h3','h4'],
        ... 'processTime': [6,6,6,6],
        ... 'finalProduct': ['h1','h2','h3','h4'],
        ... 'dueDate': [50,60,65,70],
        ... 'invHolding': [8,8,8,8],
        ... 'startTime': [0,0,0,0],
        ... 'lambdaIter': [0,0,0,0],
        ... 's_phi': [40,50,50,40],
        ... 'sigmaLambda_hj': [5,5,5,5],
        ... 'echelon': [3,3,3,3],
        ... 'stack': [[('g1', 6), ('h1', 1)],[('g1', 6), ('h1', 1)],
        ...             [('g1', 6), ('h1', 1)], [('g1', 6), ('h1', 1)]],
        ... 'sigmaPhi_pj': [7,7,7,7], 
        ... 'lb': [4,4,4,4],
        ... 'ub': [43,53,58,63],
        ... 'uk': [63,63,63,63],
        ... 'lk': [4,4,4,4],
        ... 'sigmaLambda_lambdaj': [0,0,0,0],
        ... 'weight': [-3,-3,-3,-3],
        ... 'wOverp': [-0.5,-0.5,-0.5,-0.5],
        ... 'mkGroup': ['minus','minus','minus','minus']})
        >>> la = Lagrangian()
        >>> tbl = la.doGWSPT(df,4,63,0,np.sum(df.processTime))
        >>> tbl.startTime
        0    39
        1    45
        2    51
        3    57
        Name: startTime, dtype: int64
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
    
    def runOptimalSolution(self, mk=[]):
        """ Processes waiting list grouped by machines (mk) based on 
                given lambda values (iterative output from subgradient). 
                mk has to be indexed such that a machine does not have
                a larger index than its upstream predecessor machines. 
                The algorithm will sort machines by index and start 
                scheduling from the smallest index (closest to the final
                stage).
            Calculates sigmaGamma_p_j, which is sum of process time of
                all j tasks in Gamma_i set. 
                Gamma(i): set of items scheduled on the SAME equipment as i 
                with start time later than i; 
                regardless of their product group.
            Calculates weight-over-processTime for each task in each of the
                Mk+, Mk-zero and Mk- groups, and sequence start times of 
                tasks accordingly.
            Calculates min_(j in Gamma(i) sj), sj is the start time of 
                all j tasks in Gamma_i set, Gamma(i): set of items
                scheduled on the SAME equipment as i with start time 
                later than i; regardless of their product group.
            Calculates solution value Lk(lambda) to the problems (DPk).
            Calculates solution value L(lambda) to the problem (PL).
            
        >>> df = pd.DataFrame({'parts':['a1','b1','c1','d1','g1','h1',
        ...                             'a2','b2','c2','d2','g2','h2'], 
        ... 'equipment': ['mk6','mk5','mk4','mk3','mk2','mk1', 
        ...               'mk6','mk5','mk4','mk3','mk2','mk1'], 
        ... 'immPre': ['rawMaterial','a1','b1','rawMaterial',['c1','d1'],'g1',
        ...            'rawMaterial','a2','b2','rawMaterial',['c2','d2'],'g2'],
        ... 'immSuc': ['b1','c1','g1','g1','h1','finalProduct',
        ...            'b2','c2','g2','g2','h2','finalProduct'],
        ... 'processTime': [3,2,5,4,6,1,3,2,5,4,6,1],
        ... 'finalProduct': ['h1','h1','h1','h1','h1','h1',
        ...                  'h2','h2','h2','h2','h2','h2'],
        ... 'dueDate': [50,50,50,50,50,50,60,60,60,60,60,60],
        ... 'invHolding': [1,2,4,1,8,10,1,2,4,1,8,10],
        ... 'startTime': [0,0,0,0,0,0,0,0,0,0,0,0]})
        >>> la = Lagrangian(waitingList=df)
        >>> la.runOptimalSolution()
        >>> la.waitingList.sort_values(['dueDate','parts']).loc[:, 
        ...                               ['parts','processTime','startTime']]
           parts  processTime  startTime
        10    a1            3       31.0
        8     b1            2       34.0
        6     c1            5       36.0
        4     d1            4       37.0
        2     g1            6       41.0
        0     h1            1       49.0
        11    a2            3       43.0
        9     b2            2       46.0
        7     c2            5       48.0
        5     d2            4       49.0
        3     g2            6       53.0
        1     h2            1       59.0
        """
        if len(mk) == 0:
            mk = list(set(self.waitingList.equipment))        
        mk.sort()
#        def calculateSigmaGammaPj(row):
#            sigma_pj = np.sum(tmp.loc[tmp.startTime > row.startTime, 
#                                      'processTime'])
#            return sigma_pj
        
        Lk_lambda = []
        for equipment in mk:
            mkTbl = self.waitingList[self.waitingList.equipment == equipment]
#            tmp = mkTbl.loc[:, ['startTime', 'processTime']]
#            # sigma_(j in Gamma(i)){pj} is the sum of process time of 
#            # all j's in Gamma(i) set
#            mkTbl['sigmaGamma_pj'] = mkTbl.apply(lambda row: 
#                                         calculateSigmaGammaPj(row), axis=1)
            uk = np.max(mkTbl['ub'])
            lk = np.min(mkTbl['lb'])
            mkTbl['uk'] = uk
            mkTbl['lk'] = lk
            mkTbl['feasible'] = (np.sum(mkTbl['processTime']) <= uk-lk)
            if np.sum(mkTbl['feasible'] == 0):
                # solution at equipment mk is not feasible
                # kill the current process
                self.feasibility = False
                return
            mkTbl['sigmaLambda_lambdaj'] = mkTbl.apply(lambda row: 
                                self.calculateSigmaLambda_lambdaj(row), axis=1)
            mkTbl['weight'] = (mkTbl['lambdaIter'] - mkTbl['echelon']
                                - mkTbl['sigmaLambda_lambdaj'])
            # parts with bigger value in 'weight over process time' will
            # precede parts with smaller value (proposition 1: GWSPT sequence)
            mkTbl['wOverp'] = mkTbl['weight'] / mkTbl['processTime']
            # divide i's into Mk+, Mk0, Mk- based on weight
            mkTbl['mkGroup'] = np.where(mkTbl['weight'] > 0, 'plus', 'minus')
            mkTbl.loc[(mkTbl['mkGroup'] == 'minus') 
                                & (mkTbl['weight'] == 0), 'mkGroup'] = 'zero'
            mkGroupPi = mkTbl.loc[:, ['mkGroup', 'processTime']].groupby(
                                                            'mkGroup').sum()
            try:
                plusTotal = mkGroupPi.loc['plus',:].tolist()[0]
            except KeyError:  # no positive weight found
                plusTotal = 0
            try:
                minusTotal = mkGroupPi.loc['minus', :].tolist()[0]
            except KeyError:  # no negative weight found
                minusTotal = 0
            # Corollary 1: 
            mkTbl2 = mkTbl.groupby('mkGroup').apply(self.doGWSPT, lk, uk,
                                                      plusTotal, minusTotal)
            ######################### part added but dont know if correct            
            # calculate minimum start time of j in set Gamma(i), 
            # as basis to finding start times for upper bound
            # (equation 16 in paper)
            mkTbl2 = self.calculateMinGammaSj(mkTbl2)
            # based on minGammaSj, calculate new start time si
            mkTbl2['startTime'] = mkTbl2.apply(
                lambda row: self.adjustStartTimeForUB(row), axis=1)
            ##########################################################            
            # calculate solution value to the independent problems (DPk)
            mkTbl2['DPkValue'] = (mkTbl2['weight'] * mkTbl2['startTime'])
            # calculate second term of L(lambda) = sigma_Lk(lambda)
            # + sigma_(e_i*d_psi(i) + lambda_i * p_i) 
            # - sigma_(i in F)(lambda i * d_i)
            mkTbl2['secondTerm'] = (mkTbl2['echelon'] * mkTbl2['dueDate']
                    + mkTbl2['lambdaIter'] * mkTbl2['processTime'])
            mkTbl2['thirdTerm'] = np.where(
                    mkTbl2['parts'] == mkTbl2['finalProduct'],
                    mkTbl2['lambdaIter'] * mkTbl2['dueDate'], 0)
            Lk_lambda.append(np.sum(mkTbl2['DPkValue']))
            # directly update self.waitingList so the results from 
            # previous iterations can be applied in the following iterations.
            old = self.waitingList[self.waitingList['equipment'] != equipment]
            diff = set(mkTbl.columns) - set(old.columns)
            for missingCol in diff:
                old[missingCol] = None
            old = old.loc[:, mkTbl.columns]
            self.waitingList = pd.concat([old, mkTbl2], axis=0)
            self.waitingList = self.updateSphi(self.waitingList)

        # calculate solution value to the problem (PL): maximum(L_lambda)
        self.L_lambda = (np.sum(Lk_lambda) 
                         + np.sum(self.waitingList['secondTerm'])
                         - np.sum(self.waitingList['thirdTerm']))
        self.L_lambda_tracker.append(self.L_lambda)
        # store current optimal solution value to (PL)
        if self.L_lambda > self.PL_solutionValue:
            self.PL_solutionValue = self.L_lambda
            self.PL_solutionValue_tracker.append(self.PL_solutionValue)
            self.optimalParameters = self.waitingList.copy()
            # reset counter for number of iterations in which L(lambda) has
            # not improved.
            self.muCounter = 0
        else:
            # count the number of iterations in which L(lambda) has not 
            # improved.
            self.muCounter += 1
        # store current optimal solution value to (DPk)
        self.Lk_lambda = dict(zip(mk, Lk_lambda))
        
        # if findUpperBound() will only be run once after updateSubgradient()
        # returns final result, then a zStar value has to be set. 
        # if it is somehow not given, default will be to always add on top 
        # of the best solution value L(lambda) a small epsilon, to facilitate
        # improvements in subgradient.
        if self.useHeuristicOnce and self.zStar is None:
            self.zStar = self.PL_solutionValue + 2 * self.epsilon
    
    def runLagrangianRelaxation(self):
        """ find lower bound to the original problem (P) by obtaining
                optimal solution value to problem (PL): maximize L(lambda),
                L(lambda) being the solution value of problem LR(lambda) in 
                equation 9 of paper.
            find upper bound to (PL) as z*, which is both a feasible
                solution and a reference to the subgradient method. 
            methods:
                runOptimalSolution(): calculates Lk(lambda) and L(lambda)
                    given lambda. Together with updateSubgradient() determines 
                    lower bound to the problem (PL).
                findUpperBound(): calculates zStar. Determines upper
                    bound (feasible solution) to the problem (PL).
                updateMu(): if no improvement for zeta iterations, 
                    then reduce the value of mu.
                updateSubgradient(): finds optimal lambda iteratively.
        """
        for nrIter in np.arange(self.omega):
            self.runOptimalSolution()  # find lower bound to problem (PL)
            if self.feasibility is False:
                break

            if not self.useHeuristicOnce:
                # find upper bound (z*) to problem (PL)
                self.findUpperBound()

            relativeError = ( (self.zStar - self.PL_solutionValue) 
                               / self.PL_solutionValue )
            if relativeError <= self.epsilon:
                break
            if self.muCounter > self.zeta:
                self.updateMu()
            self.updateSubgradient()
            
        if self.useHeuristicOnce:
            # if to save computation time, the heuristic for finding upper
            # bound (feasible solution) is set to only run once after 
            # subgradient is concluded: 
            self.findUpperBound()
        
    def calculateMinGammaSj(self, _mkTbl, sortedAsc=False):
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

    def adjustStartTimeForUB(self, row):
        """ determine starting time of items for given sequence. 
            equation 16 in paper.
        """
        if row['parts'] == row['finalProduct']:
            si = (np.min([row['dueDate'], row['minGammaSj']]) 
                    - row['processTime'])
        else:
            si = (np.min([row['s_phi'], row['minGammaSj']])
                    - row['processTime'])
        return si
       
    def findUpperBound(self, mk=[]):
        """ find upper bound (feasible solution) to the problem (PL).
                mk's are indexed such that a machine does not have a larger 
                index than its upstream predecessor machines.
        """
        if len(mk) == 0:
            mk = list(set(self.waitingList.equipment))
        mk.sort()  # start from smallest index (latest down-stream machine)
        newTbl = pd.DataFrame()
        Lk_lambda = []
        for equipment in mk:
            mkTbl = self.waitingList[self.waitingList.equipment == equipment]
            mkTblIdx = mkTbl.index
            mkTbl = self.updateSphi(mkTbl)  # will change index. be ware.
            mkTbl.index = mkTblIdx
            # step 0
            mkTbl = mkTbl.sort_values(by='startTime', ascending=True)
            rho = mkTbl.index.tolist()
            # step 1
            # calculate minimum start time of j in set Gamma(i), 
            # as basis to finding start times for upper bound
            # (equation 16 in paper)
            mkTbl = self.calculateMinGammaSj(mkTbl, sortedAsc=True)
            # based on minGammaSj, calculate new start time si
            mkTbl['startTime'] = mkTbl.apply(
                lambda row: self.adjustStartTimeForUB(row), axis=1)
            mkTbl['dueDate_dummy'] = mkTbl['s_phi']
            mkTbl.loc[mkTbl.parts==mkTbl.finalProduct, 
                'dueDate_dummy'] = mkTbl.loc[mkTbl.parts==mkTbl.finalProduct,
                                             'dueDate']
            L = mkTbl.shape[0] - 1
            # step 2
            if L > 0: 
                # if there are at least two items in mkTbl:
                for l in np.arange(L-1, -1, step=-1, dtype=int):
                    # step 3:
                    dIdx = mkTbl.columns.tolist().index('dueDate_dummy')
                    pIdx = mkTbl.columns.tolist().index('processTime')
                    sIdx = mkTbl.columns.tolist().index('startTime')
                    d_l = mkTbl.iloc[l, dIdx]
                    d_mk = mkTbl.iloc[-1, dIdx]
                    p_l = mkTbl.iloc[l, pIdx]
                    if d_l < d_mk + p_l:
                        # step 4:
                        for h in np.arange(L, l, step=-1, dtype=int):
                            s_h = mkTbl.iloc[h, sIdx]
                            s_h_1 = mkTbl.iloc[h-1, sIdx]
                            p_h_1 = mkTbl.iloc[h-1, pIdx]
                            p_l = mkTbl.iloc[l, pIdx]
                            if (s_h - (s_h_1 + p_h_1) >= p_l
                                and (s_h_1 + p_h_1) + p_l <= d_l):
                                tmp = rho.pop(l)
                                rho.insert(h, tmp)
                                mkTbl = mkTbl.loc[rho, :]
                                # update task start time
                                mkTbl.iloc[-1, sIdx] = (mkTbl.iloc[-2, sIdx]
                                                        + mkTbl.iloc[-2, pIdx])
                                # update minGammaSj for all
                                mkTbl = self.calculateMinGammaSj(mkTbl, 
                                                                sortedAsc=True)
                                # adjust start time based on equation 16
                                mkTbl['startTime'] = mkTbl.apply(
                                    lambda row: self.adjustStartTimeForUB(row),
                                    axis=1)     
                    else:
                        # move l-th task to the end
                        tmp = rho.pop(l)
                        rho.append(tmp)
                        mkTbl = mkTbl.loc[rho, :]
                        # update task start time
                        mkTbl.iloc[-1, sIdx] = (mkTbl.iloc[-2, sIdx] 
                                                    + mkTbl.iloc[-2, pIdx])
                        # update minGammaSj for all
                        mkTbl = self.calculateMinGammaSj(mkTbl, sortedAsc=True)
                        # adjust start time based on equation 16
                        mkTbl['startTime'] = mkTbl.apply(
                            lambda row: self.adjustStartTimeForUB(row), axis=1)                    
            
            # update self.waitingList with new down-stream start times,
            # so the s_phi of upstream machines can be updated in the 
            # following iterations.
            self.waitingList.loc[mkTbl.index, 'startTime'] = mkTbl['startTime']
            
            mkTbl.drop('dueDate_dummy', axis=1, inplace=True)
            # calculate solution value to the independent problems (DPk)
            mkTbl['DPkValue'] = (mkTbl['weight'] * mkTbl['startTime'])            
            Lk_lambda.append(np.sum(mkTbl['DPkValue']))
            newTbl = pd.concat([newTbl, mkTbl], axis=0)

        self.waitingList = newTbl.copy()   
        
        # calculate upper bound of solution value to the 
        # problem (PL): maximum(L_lambda)
        self.zStar = (np.sum(Lk_lambda) + np.sum(newTbl['secondTerm'])
                                             - np.sum(newTbl['thirdTerm']))
        self.zStar_tracker.append(self.zStar)
        # store upper bound of solution values to (DPk)
        self.Lk_lambda = dict(zip(mk, Lk_lambda))
        self.optimalParameters = self.waitingList.copy()
        
class BranchBound():
    def __init__(self, lagrangianClass=None, runLagrangian=False):
        self.runLagrangian = runLagrangian
        self.UB = None  # upper bound of solution value to (PL)
        self.setS = pd.DataFrame()  # fixed set of rho_k and sigma_k
        self.setPprime = pd.DataFrame()  # set P': all items not in set S
        self.sigma_k = pd.DataFrame()  # partial sequence of machine k
        self.s_k = pd.DataFrame()  # partial, not scheduled tasks on machine k
        self.machineList = []  # all machines to be scheduled
        self.mk = None  # index of machine associated with current node
        self.solution = None  # current best soluton
        self.UBtrack = []  # tracker for system UB
        self.LBtrack = []  # tracker for system LB
        self.finished = False  # indicator for end of algorithm
        self.kpi_invCost = pd.DataFrame()  # tracker for system inventory cost
        self.kpi_makespan = pd.DataFrame()  # tracker for avg order make span
        self.kpi_utilization = pd.DataFrame()  # tracker for utilization %
        
        self.loadOriginal(lagrangianClass)
        
    def loadOriginal(self, lagrangianClass):
        """ create lagrangian class if not given.
            run the heuristic once to initialize, if not already done.
        """
        if lagrangianClass is None:
            self.lagrangian = Lagrangian()
        else:
            self.lagrangian = lagrangianClass
        if self.runLagrangian:
            self.lagrangian.runLagrangianRelaxation()
        
        # set P': set of anything not in S
        self.setPprime = self.lagrangian.optimalParameters
        # back up the original values
        self.setPprime['originalDueDate'] = self.setPprime['dueDate']
        self.setPprime['originalFinalProduct'] = self.setPprime['finalProduct']
        self.setPprime['originalImmSuc'] = self.setPprime['immSuc']
    
        self.solution = self.setPprime.copy()
        self.UB=self.lagrangian.zStar
        self.UBtrack.append(self.UB)
        self.LBtrack.append(self.lagrangian.L_lambda)
        self.machineList = list(set(self.setPprime.equipment))
        self.machineList.sort()
        
    def createRootNode(self):
        """ take the last node scheduled in the lagrangian heuristic
            during initailization. 
            update sets S, P', sigma_k, s_k.
            S and P' being global trackers of fixed items and items to be 
                scheduled.
            sigma_k, s_k and mk are dynamic variables to keep track of 
                the current machine associated with the current node in focus.
        """
        self.mk = self.machineList[0]
        self.s_k = self.setPprime[self.setPprime.equipment == self.mk]
        rootNode = self.s_k.loc[
                        self.s_k.startTime == np.max(self.s_k.startTime), :]
        self.sigma_k = rootNode
        self.setS = rootNode
        self.s_k = self.s_k[~self.s_k.index.isin(rootNode.index)]
        self.setPprime = self.setPprime[
                                ~self.setPprime.index.isin(rootNode.index)]
        # update final product, due dates and immediate successors in the 
        # set P', when a task is fixed and moved into set S. Its immediate
        # predecessors will become the final products in the next node.
        self.setPprime = self.updateFinalProduct(self.setS, self.setPprime)
    
    def updateFinalProduct(self, setS, _setP, smin=None):
        """ A part becomes a final product if its 
            immediate successor is already scheduled and fixed in set S.
            in set P', immediate successor of the new final product, as well
            as final product/due date of its predecessors will have to be 
            updated.
        """
        setP = _setP.copy()
        if smin is None:
            try:
                smin = np.min(self.sigma_k['startTime'])
            except:  # beginning of scheduling on a new machine, sigma_k = []
                smin = sys.maxsize
        mk = self.mk

        def updateDueDate(row):
            if row['equipment'] == mk:
                newDueDate = np.min([smin, row['s_phi'], row['dueDate']])
            else:
                newDueDate = np.min([row['s_phi'], row['dueDate']])
            return newDueDate
        
        newFinalProduct = setP.loc[setP['immSuc'].isin(setS['parts']),
                ['parts','equipment', 'immPre', 'immSuc', 'dueDate', 's_phi']]
        if newFinalProduct.shape[0] == 0:  # no additional tasks added to set S
            return setP
        
        newFinalProduct['newDueDate'] = newFinalProduct.apply(lambda row: 
                                                updateDueDate(row), axis=1)
        for idx in newFinalProduct.index:
            newFinal = newFinalProduct.loc[idx, 'parts']
            newDue = newFinalProduct.loc[idx, 'newDueDate']
            pre = newFinalProduct.loc[idx, 'immPre']
            # update all predecessors
            for item in pre:
                setP.loc[setP['parts'] == item, 'finalProduct'] = newFinal
                setP.loc[setP['parts'] == item, 'dueDate'] = newDue
            # update new final product itself
            setP.loc[setP['parts'] == newFinal, 
                ['immSuc', 'finalProduct', 'dueDate']] = ['finalProduct', 
                newFinal, newDue]
        return setP

    def createNode(self, row):
        """ create new node with random row selected from self.s_k
            return new setS, setP, deleteFlag, LB and UB
        """
        deleteFlag = False
        df = pd.DataFrame([row])
        df['startTime'] = df['dueDate'] - df['processTime']
        setS_tmp = self.setS.copy()
        setS_tmp = pd.concat([setS_tmp, df], axis=0)
        setP_tmp = self.setPprime.loc[self.setPprime.parts != row.parts, :]
        smin = np.min(setS_tmp.loc[setS_tmp.equipment == self.mk, 'startTime'])
        setP_tmp = self.updateFinalProduct(setS_tmp, setP_tmp, smin)
        
        lagrangianNode = Lagrangian(waitingList=setP_tmp, 
                                    useHeuristicOnce=True,
                                    zStarRef=self.UB)
        lagrangianNode.runLagrangianRelaxation()
        
        if not lagrangianNode.feasibility:
            deleteFlag = True

        Q = (  np.sum( setS_tmp['echelon'] 
                * (setS_tmp['originalDueDate'] - setS_tmp['startTime']) )
             + np.sum( setP_tmp['echelon']
                * (setP_tmp['originalDueDate'] - setP_tmp['dueDate']) ) )
        
        nodeLB = lagrangianNode.PL_solutionValue + Q
        nodeUB = lagrangianNode.zStar + Q
        
        # why is my upper bound greater than my lower bound? 
        # is it because I applied stricter constraints to make LB solution
        # feasible in Lagrangian.runOptimalSolution()? so here it should be 
        # "if min(nodeUB, nodeLB)>=self.UB" instead of "if nodeLB>=self.UB"?
        if np.min([nodeLB, nodeUB]) >= self.UB:
            deleteFlag = True
        
        return pd.Series([lagrangianNode.optimalParameters,
                setS_tmp, setP_tmp, nodeLB, nodeUB, deleteFlag])
        
    def branchOut(self):
        """ assuming root node is already created.
        """
        if self.setS.shape[0] == 0:
            self.createRootNode()
        
        if self.setPprime.shape[0] == 0:
            self.finished = True
            return

        if self.s_k.shape[0] == 1: 
            # current machine has only one task left to be scheduled
            self.s_k['startTime'] = (self.s_k['dueDate'] 
                                        - self.s_k['processTime'])
            self.setS = pd.concat([self.setS, self.s_k], axis=0)
            self.setPprime = self.setPprime[
                                self.setPprime.index != self.s_k.index]
            self.s_k = pd.DataFrame()
            self.updateFinalProduct(self.setS, self.setPprime)
            
        if self.s_k.shape[0] == 0:  # current machine is fully sequenced:
            self.mk = self.machineList[self.machineList.index(self.mk) + 1]
            self.s_k = self.setPprime[self.setPprime.equipment == self.mk]
            self.sigma_k = []
        
        nodes = self.s_k.apply(lambda row: self.createNode(row), axis=1)
        nodes.columns = ['solution', 
                         'setS', 'setP', 'nodeLB', 'nodeUB', 'deleteFlag']
        nodes = nodes[~nodes['deleteFlag']]
        if nodes.shape[0] == 0:  # no better solutions found:
            self.finished = True
            return
        newUB = np.min(nodes['nodeUB'])
        if newUB < self.UB:
            self.UB = newUB
            self.UBtrack.append(newUB)
            s = nodes.loc[nodes['nodeUB'] == newUB, 'setS'].tolist()[0]
            newSolution = nodes.loc[nodes['nodeUB'] == newUB, 
                                                  'solution'].tolist()[0]
            self.solution = pd.concat([s, newSolution], axis=0)
            self.solution.reset_index(drop=True, inplace=True)
        
        nodes = nodes[nodes['nodeLB'] < newUB]
        if nodes.shape[0] == 0:  # no better solutions found:
            self.finished = True
            return
        minLB = np.min(nodes['nodeLB'])
        self.LBtrack.append(minLB)        
        self.setS = nodes.loc[nodes['nodeLB'] == minLB, 'setS'].tolist()[0]
        self.setPprime = nodes.loc[nodes['nodeLB'] == minLB, 
                                                       'setP'].tolist()[0]
        self.sigma_k = self.setS[self.setS['equipment'] == self.mk]
        self.s_k = self.setPprime[self.setPprime['equipment'] == self.mk]
            
    def runBB(self):
        self.createRootNode()
        counter = 0
        self.kpi_invCost, self.kpi_makespan, self.kpi_utilization = (
                calculateKPI(solution=self.solution, nrIter=counter,
                kpi_invCost=self.kpi_invCost, kpi_makespan=self.kpi_makespan,
                kpi_utilization=self.kpi_utilization ) )
        while self.finished is not True:
            counter += 1
            self.branchOut()
            self.kpi_invCost, self.kpi_makespan, self.kpi_utilization = (
                calculateKPI(solution=self.solution, nrIter=counter,
                kpi_invCost=self.kpi_invCost, kpi_makespan=self.kpi_makespan,
                kpi_utilization=self.kpi_utilization ) )
            
#%%        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
    la = Lagrangian()
    la.runLagrangianRelaxation()
    bb = BranchBound(lagrangianClass=la)
    bb.runBB()
    bb.solution.to_csv('testBBresult.csv')
        
    
    