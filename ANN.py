#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:59:24 2017

@author: Nat
"""
from branchBound import calculateKPI
from importlib import reload
import os
from keras import backend as K

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
set_keras_backend("theano")
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop

import numpy as np
import pandas as pd
import itertools

class Brain():
    def __init__(self, nrState, nrAction, nrFeature=1,
                 learnRate=0.00025, nrUnit=[64],
                 batchSize=64, epoch=1, verbose=0):
        """ dynamic scheduling reference: 
                http://www.redalyc.org/pdf/3783/378349711004.pdf
            RL modeling reference:
                https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
            DQN reference:
                https://web.stanford.edu/class/psych209/Readings/
                                MnihEtAlHassibis15NatureControlDeepRL.pdf
            Inputs:
            test data: n=4/m=6/flowshop permutation(P)/max.flowtime(Fmax) 
                problem, with only one agent defining one sequence 
                for all machines
            nrState: each state is an array of n jobs, representing one 
                possible sequence(permutation) of the n jobs. nrState = n
            nrAction: number of actions = (C(n,r=2) + 1), 
                each action represents the choice of a job to 
                reschedule (C(n,1)), and an inserting point in the 
                existing sequence (C(n-1,1), choose one position other than
                the current one); plus the choice of 'do-nothing'.
            nrUnit: number of units in hidden layer
            learnRate: learning rate of optimizer
            batch size: batch size to feed into the ANN
            epoch: number of epochs to go through
            verbose: if to print out all or part of the messages during 
                training
        """
        
        self.nrUnit = nrUnit.copy()
        self.nrState = nrState
        self.nrAction = nrAction
        self.nrFeature = nrFeature
        self.learnRate = learnRate
        self.model = self.createModel()
        self.batchSize = batchSize
        self.epoch = epoch
        self.verbose = verbose
    
    def createModel(self, lossFunc='mse', dropoutRate=0.25):
        """ DQN. """
        model = Sequential()
        model.add( Dense(units=self.nrUnit[0],
                            activation='relu', 
                            input_shape=(self.nrState, self.nrFeature)) )
        model.add(Dropout(rate=dropoutRate))
        for i in np.arange(start=1, stop=len(self.nrUnit)):
            model.add(Dense(units=self.nrUnit[i], activation='relu'))
            model.add(Dropout(rate=dropoutRate))
        model.add(Flatten())
        model.add(Dense(units=self.nrAction, activation='linear'))
        # about RMSprop: 
        # www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        opt = RMSprop(lr=self.learnRate)
        model.compile(loss=lossFunc, optimizer=opt)
        return model
    
    def train(self, x, y):
        """ x: an array of n-vectors of states (different permutations 
                of the n jobs)
            y: an array of n=vectors of actions 
                (different inserting points)
        """
        self.model.fit(x, y, batch_size=self.batchSize, 
                       epochs=self.epoch, verbose=self.verbose)
    
    def predict(self, s):
        return self.model.predict(s)  
    
class Memory():
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.experience = []

    def add(self, newExperience):
        """ add a tuple (s,a,r,s') into the memory.
            s: current sequence(permutation) in a n-vector
            a: chosen action and its next inserting point in a tuple(pos1,pos2)
            r: scalar reward returned by environment
            s': next sequence(permutation) in a n-vector
        """
        self.experience.append(newExperience)        

        if len(self.experience) > self.capacity:
            self.experience.pop(0)
    
    def sample(self, sampleSize):
        """ sample from memory for replay """
        num = np.random.choice(len(self.experience), size=sampleSize)
        samples = [self.experience[i] for i in num]
        return samples
        
class Agent():
    """ represents a machine with n jobs to schedule. 
        In a n/m/P/. problem, same job permutation is used across all 
            machines. therefore only one agent is neccessary. Otherwise
            in a normal n/m/F/. problem, each machine needs to be 
            represented by an agent.
    """
    def __init__(self, nrState, nrAction, nrFeature=1,
                 learnRate=0.00025, nrUnit=[64],
                 batchSize=64, epoch=1, verbose=0, memoryCapacity=100000):
        self.nrState = nrState
        self.nrAction = nrAction
        self.nrFeature = nrFeature
        self.batchSize = batchSize
        self.brain = Brain(nrState=nrState, nrAction=nrAction, 
            nrFeature=nrFeature, learnRate=learnRate, 
            nrUnit=nrUnit, batchSize=batchSize, 
            epoch=epoch, verbose=verbose)
        self.memory = Memory(capacity=memoryCapacity) 
        # exploration vs. exploitation threshold in epsilon greedy
        self.epsilon = 1
        self.t = 0  # time step
        self.lmbda = 0.001  # deterioration rate of epsilon
        self.epsilonMax = 0.99
        self.epsilonMin = 0.01
        self.gamma = 0.99  # deterioration rate in reward
    
    def act(self, s):
        """ epsilon-greedy: if random number is less than given epsilon,
                then explore. else exploit.
            input:
                s: current state, whose action is to be chosen.
                    Only one row, as opposed to the prediction of 
                    the entire batch. Therefore a reshape of s is required.
            output:
                a: index of the tuple (action and next inserting point), 
                    the list of true action tuples defined in 
                    environment.initializeStates
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(low=0, high=self.nrAction)
        else:
            pred = self.brain.predict(s.reshape(1,self.nrState,self.nrFeature))
            a = np.argmax(pred.flatten())
        return a
    
    def observe(self, newExperience):
        """ record return (newExperience) from the environment.
            update epsilon (make it less explorative as more experience 
                becomes available).
        """
        self.memory.add(newExperience)
        self.updateEpsilon()
        
    def updateEpsilon(self): 
        """
        \epsilon=\epsilon_{min}+(\epsilon_{max}-\epsilon_{min}) e^{-\lambda t}
        """
        self.t += 1
        if self.epsilon > 0.1:
            self.epsilon = (self.epsilonMin  + (self.epsilonMax 
                            - self.epsilonMin) * np.exp(-self.lmbda * self.t))
    
    def replay(self):
        """ variables:
            s: current state in the format of n-vector, representing the 
                current sequence (permutation) of the n jobs
            sPrime: s' - next state (as in the tuple (s,a,r,s'))
            q: value of actions from s
            qPrime: value of actions from s'
        """
        samples = self.memory.sample(self.batchSize)
        s = np.array([s[0] for s in samples])
        sPrime = np.array([s[3] for s in samples])
        
        # all action values of the current state.
        # this is to keep the record of the actions which are not involved in 
        # this tuple of (s,a,r,s').
        q = self.brain.predict(s)
        # all action values of the next state.
        # this is to update the action value for the involved action a in 
        # this tuple of (s,a,r,s').
        qPrime = self.brain.predict(sPrime)
        
        x = np.zeros([self.batchSize, self.nrState, self.nrFeature])
        y = np.zeros([self.batchSize, self.nrAction])
        for i in np.arange(self.batchSize):
            # update current action values and create new x, y for training
            # Q(s, a) = r + \gamma max_a Q(s', a)
            sample = samples[i]
            # reward of the current state
            r = sample[2]
            # action index of the involved action in (s,a,r,s')
            a = int(sample[1])
            
            newQ = q[i]
            # only update the target action value 
            # which is in the tuple (s,a,r,s')
            newQ[a] = r + self.gamma * np.argmax(qPrime[i])
            
            x[i,] = s[i]
            y[i,] = newQ
        
        self.brain.train(x, y)

class Environment():
    def __init__(self, refSolution, features=['finalProduct'],
                 setupTimeDict=None, defaultSetupTime=0):
        self.waitingList = refSolution
        if 'originalFinalProduct' in self.waitingList.columns:
            self.waitingList['finalProduct'] = self.waitingList[
                                                    'originalFinalProduct']
            self.waitingList['immSuc'] = self.waitingList['originalImmSuc']
            self.waitingList['dueDate'] = self.waitingList['originalDueDate']
        else:
            # create column for storing original due dates. 
            # the column 'duedate' will be used for randomization of due dates
            self.waitingList['originalDueDate'] = self.waitingList['dueDate']            
        self.waitingList.index = np.arange(0,self.waitingList.shape[0])
        # number of jobs is equal number of final products
        self.jobs = self.waitingList.loc[
                self.waitingList['immSuc'] == 'finalProduct', 'finalProduct']
        self.jobs = list(set(self.jobs))
        self.nrState = len(self.jobs)
        self.actions = [(0,0)]  # do-nothing
        self.actions = ( self.actions 
            + list(itertools.combinations(np.arange(1, len(self.jobs)+1), 2)) )
        self.nrAction=len(self.actions)
        self.features = features
        self.nrFeature = len(self.features)
        self.setupTimeDict = setupTimeDict
        self.defaultSetupTime = defaultSetupTime
        self.relativeDueDate = self.setRelativeDueDate()
        self.goal = 'invCost'  # 1 of 3:'invCost', 'makeSpan', 'utilization'
        self.rewardTracker = []

        self.state = self.setState(jobs=self.jobs)
        self.codedJobs = self.codeJob()  # dictionary to map jobs to numerics
        
    def setState(self, jobs, state=None):
        """ set initial state in each run. """
        if state is not None:
            # if initial states are given (e.g. from a static schedule)
            s = np.array(state)
        else:
            s = np.array(np.random.choice(jobs, size=self.nrState, 
                                                          replace=False))
        return s

    def setRelativeDueDate(self):
        """ placeholder for setting relative due date as adjustment to the 
            due date feature as input to the NN.
        """
        return np.min(self.waitingList['originalDueDate'])

    def codeJob(self):
        """ utility function for encoding and decoding jobs (only numerics 
            are used in the brain)
        """
        original = list(set(self.waitingList['parts'].apply(str)))
        original.sort()
        idx = np.arange(0, len(original)).tolist()
        new = dict(zip(original + idx, idx + original))
        return new
        
    def runScheduler(self, s, a):
        """ return next state and reward given current state and action.
            s: current state (sequence)
            a: intended action (index to the tuple (pos0, pos1),
                the two positions being the indexes of the pair to be 
                swapped in sequence)
            _waitingList: list of jobs with detailed info such as 
                process time, due date, routing, etc.
        """
        sPrime = s.tolist()
        pos0, pos1 = self.actions[int(a)]
        if pos0 != 0:
            # if NOT the do-nothing option of s' = s,
            # recover position as index of inserting point
            pos0 -= 1
            pos1 -= 1
            job = s[pos0]
            sPrime.remove(job)
            sPrime.insert(pos1, job)
        
        self.allMachineSameSequenceBackward(_sPrime=sPrime)
        
        kpi_invCost, kpi_makespan, kpi_utilization = calculateKPI(
                            self.waitingList, finalProdCol='finalProduct')
        
        if self.goal == 'invCost':
            r = -np.sum(kpi_invCost['invCost'])
        if self.goal == 'makeSpan':
            r = -np.max(kpi_makespan['makespan'])
        if self.goal == 'utilization':
            r = np.average(kpi_utilization['utilization'])
              
        return r, sPrime

    def updateSphi(self, _tbl):
        """ utility function to update s_phi.
            s_phi(i) is the start time of phi(i).
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
    
    def lookupTbl(self, key, tbl, keyID, targetID):
        """ utility function for table lookup.
            input:
                key: keys list to be looked up
                tbl: table of both keys and corresponding values
                keyID: column name of the keys
                targetID: column name of the values
            output:
                target: values list corresponding to the key.
        """
        key = pd.DataFrame(key, columns=[keyID])
        tbl = key.merge(tbl.loc[:,[keyID, targetID]], on=keyID, how='left')
        tbl.drop_duplicates(inplace=True)
        target = tbl[targetID].tolist()
        return target
    
    def allMachineSameSequenceBackward(self, _sPrime):
        # always mark out the sequence with final product index
        sPrime = _sPrime.copy()
        sPrime = self.lookupTbl(key=sPrime, tbl=self.waitingList, 
                                keyID='parts', targetID='finalProduct')
        
        sPrime.reverse()  # backward: later jobs get to be scheduled first
        mk = list(set(self.waitingList.equipment))
        mk.sort()  # start from smallest index (latest down-stream machine)
       
        for equipment in mk:
            mkTbl = self.waitingList[self.waitingList.equipment == equipment]
            partsSequence = self.lookupTbl(key=sPrime, tbl=mkTbl, 
                                keyID='finalProduct', targetID='parts')
            newTbl = self.singleMachineSequenceBackward(mkTbl, partsSequence)
            
            # update waitingList with new down-stream start times,
            # so the s_phi of upstream machines can be updated in the 
            # following iterations.
            self.waitingList.loc[newTbl.index, 
                                        'startTime'] = newTbl['startTime']       
            idx = self.waitingList.index
            self.waitingList = self.updateSphi(_tbl=self.waitingList)
            self.waitingList.index = idx

    def singleMachineSequenceBackward(self, mkTbl, partsSequence):
        latestStartTime = 99999
        newTbl = pd.DataFrame()
        # partsSequence: reversed, latest job in front
        for job in partsSequence:
            row = mkTbl.loc[mkTbl['parts'] == job,:]
            if job == partsSequence[0]:
                # for all jobs which are not the last job scheduled on 
                # the machine: add setup time between itself and the 
                # next job on the same machine                
                nextJob = job
                setupTime = 0
            else:
                try: 
                    setupTime = self.setupTimeDict[(job, nextJob)]
                except KeyError:
                    setupTime = self.defaultSetupTime
                
            if row['immSuc'].tolist()[0] == 'finalProduct':
                latestStartTime = (min([row['dueDate'].tolist()[0], 
                                        latestStartTime]) 
                                 - setupTime - row['processTime'].tolist()[0])
            else:
                latestStartTime = (min([row['s_phi'].tolist()[0], 
                                        latestStartTime]) 
                                - setupTime - row['processTime'].tolist()[0])
            nextJob = job
            row['startTime'] = latestStartTime
            newTbl = pd.concat([newTbl, row], axis=0)
        return newTbl

    def reformatAndAddFeatures(self, state, randomizeDueDate=True, 
                               relativeDueDate=0):
        """ utility function for creating s and s' to feed the NN """
        # convert to numerics
        s = np.array([self.codedJobs[key] for key in state 
                                             if key in self.codedJobs])
        # if to add other dimensions for learning:
        # s \in \mathbb{R}^{batchSize x nrState x nrFeature}
        tmpFeatures = self.features.copy()
        tmpFeatures.remove('finalProduct')
        for f in tmpFeatures:
            feature = np.array(self.lookupTbl(
                state, self.waitingList, 'finalProduct', f))
            if f == 'dueDate':
                # use relative due date:
                feature = feature - relativeDueDate
                if randomizeDueDate:
                    feature = np.random.randint(low=0, 
                                    high=np.max(feature), size=len(feature))
                    
            s = np.column_stack([s, feature])
        return s
        
    def run(self, agent, observationLoop=10, replayLoop=10):
        """ observationLoop: number of times to stimulate the environment
                and only observe the resulting state / reward.
            replayLoop: number of times to retrain the brain with new
                observations
        """
        totalReward = 0
        # randomly chosen initial state for each run.
        self.state = self.setState(jobs=self.jobs)
        
        def updateDueDate(row):
            tmpState = row['finalProduct']
            tmpRidx = list(self.state).index(tmpState)
            tmpCidx = self.features.index('dueDate')
            return s[tmpRidx, tmpCidx] + self.relativeDueDate
        
        for i in np.arange(replayLoop):
            for j in np.arange(observationLoop):             
                s = self.reformatAndAddFeatures(self.state, 
                                        randomizeDueDate=True, 
                                        relativeDueDate=self.relativeDueDate)
                self.waitingList['dueDate'] = self.waitingList.apply(
                                    lambda row: updateDueDate(row), axis=1)
                a = agent.act(s)
                
                r, _sPrime = self.runScheduler(self.state, a)               
                sPrime = self.reformatAndAddFeatures(_sPrime,
                                         randomizeDueDate=False,
                                         relativeDueDate=self.relativeDueDate)
                
                agent.observe((s, a, r, sPrime))
                self.state = np.array(_sPrime)
                totalReward += r
                self.waitingList['dueDate']=self.waitingList['originalDueDate']
            agent.replay()
        
        self.rewardTracker.append(totalReward)
    
    def sortByDueDate(self):
        refState = self.waitingList.loc[:, 
                               ['finalProduct','dueDate']].drop_duplicates()
        refState.sort_values('dueDate', ascending=True, inplace=True)
        refState = np.array(refState['finalProduct'])
        return refState
    
    def returnResult(self, refState=None):
        agent.epsilon = 0
        if refState is None:
            self.state = self.sortByDueDate()
        else:
            self.state = refState
        s = self.reformatAndAddFeatures(self.state, 
                                        randomizeDueDate=False,
                                        relativeDueDate=self.relativeDueDate)
        a = agent.act(s)
        
        _, _sPrime = self.runScheduler(self.state, a)
        kpi_invCost, kpi_makespan, kpi_utilization = calculateKPI(
                                self.waitingList, finalProdCol='finalProduct')
        return _sPrime, kpi_invCost, kpi_makespan, kpi_utilization
        
def createDict(setupTimeDict):
    def createTuple(row):
        return (str(row[0]), str(row[1]))
    setupTimeDict['key'] = setupTimeDict.apply(lambda row: createTuple(row), 
                                                                        axis=1)
    keys = setupTimeDict['key'].tolist()
    values = setupTimeDict['setupTime'].tolist()
    newDict = dict(zip(keys, values))
    return newDict

#%%
if __name__ == '__Main__':

    refSolution = pd.read_csv('testBBresult.csv')
    setupTimeDict = pd.read_csv('testSetupTime.csv')
    
    setupTimeDict = createDict(setupTimeDict)
    
    env = Environment(refSolution=refSolution, setupTimeDict=setupTimeDict, 
                      features=['finalProduct', 'dueDate'])
    agent = Agent(env.nrState, env.nrAction, env.nrFeature)
#    agent.brain.model.summary()
    
    env.run(agent, 10, 1)
    for i in np.arange(20):
        env.run(agent, 10, 5)

    solution, kpi_invCost, kpi_makespan, kpi_utilization = env.returnResult()

