import pandas as pd
import numpy as np
import pickle
import os

#### Data preparation #######
# Convert to transactional rows
def convert2RowFormatTransactions(data, toNumericalStep=True):
  step = 1
  lastDT = 0
  transactions = []
  transaction = []
  minIdx = data.index[0]
  for j in range(data.shape[0]):
    i = j + minIdx

    if data.loc[i, 'datetime'] != lastDT:
      # update lastDT
      lastDT = data.loc[i, 'datetime']
      
      # get uniqueness
      transaction = np.array(transaction)
      transaction = np.unique(transaction)
      transaction = list(transaction)
      transaction = [step] + sorted(transaction)

      # append to the main transactions
      transactions.append(transaction)
      #print(transaction)

      # re-init current transaction
      transaction = []
      step += 1
    
    # add mesh to current trans
    transaction.append(data.loc[i, 'id'])
  
  return transactions

def getTimestampPattern(transactions, pattern):
  timestamps = []
  for transaction in transactions:
    transactionItems = set(transaction[1:])
    pattern = set(pattern)
    if pattern <= transactionItems:
      timestamps.append(transaction[0])
  
  return timestamps

class Node(object):
    def __init__(self, item, children):
        self.item = item
        self.children = children
        self.parent = None
        self.tids = []

    def addChild(self, node):
        self.children[node.item] = node
        node.parent = self


class Tree(object):
    def __init__(self):
        self.root = Node(None, {})
        self.summaries = {}
        self.info={}
    def add_transaction(self,transaction,tid):
        curr_node=self.root
        for i in range(len(transaction)):
            if transaction[i] not in curr_node.children:
                new_node=Node(transaction[i],{})
                curr_node.addChild(new_node)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(new_node)
                else:
                    self.summaries[transaction[i]]=[new_node]                    
                curr_node=new_node                
            else:
                curr_node=curr_node.children[transaction[i]]            
        curr_node.tids= curr_node.tids + tid
        
    def get_condition_pattern(self,alpha):
        final_patterns=[]
        final_sets=[]
        for i in self.summaries[alpha]:
            q= self.genrate_tids(i)
            set1=i.tids 
            set2=[]
            while(i.parent.item!=None):
                set2.append(i.parent.item)
                i=i.parent
            if(len(set2)>0):
                set2.reverse()
                final_patterns.append(set2)
                final_sets.append(set1)
        final_patterns,final_sets,info=cond_trans(final_patterns,final_sets)
        return final_patterns,final_sets,info
    
    def genrate_tids(self,node):
        final_tids=node.tids
        return final_tids
    def remove_node(self,node_val):
        for i in self.summaries[node_val]:
            i.parent.tids = i.parent.tids + i.tids
            del i.parent.children[node_val]
            i=None
    def get_ts(self,alpha):
        temp_ids=[]
        for i in self.summaries[alpha]:
            temp_ids+=i.tids
        return temp_ids
                
    def generate_patterns(self,prefix):
        global maximalTree,maximalItemsets
        for i in sorted(self.summaries,key= lambda x:(self.info.get(x)[0],-x)):
            pattern=prefix.copy()
            pattern.append(i)
            patterns,tids,info=self.get_condition_pattern(i)
            conditional_tree=Tree()
            conditional_tree.info=info.copy()
            head=pattern.copy()
            tail=[]
            for l in info:
                tail.append(l)
            sub=head+tail
            if(maximalTree.checkerSub(sub)==1):
                for pat in range(len(patterns)):
                    conditional_tree.add_transaction(patterns[pat],tids[pat])
                if(len(patterns)>1):
                    conditional_tree.generate_patterns(pattern)
                else:
                    #print(prefix,pattern)
                    if len(patterns)==0:
                        mappedP=[]
                        for lm in pattern:
                            mappedP.append(pfList[lm])
                        maximalTree.add_transaction(pattern)
                        maximalItemsets.append([mappedP,self.info[i]])
                    else:
                        inf=getPer_Sup(tids[0])
                        patterns[0].reverse()
                        pattern=pattern+patterns[0]
                        mappedP=[]
                        for lm in pattern:
                            mappedP.append(pfList[lm])
                        maximalTree.add_transaction(pattern)
                        maximalItemsets.append([mappedP,inf])
            self.remove_node(i)


class MNode(object):
    def __init__(self, item, children):
        self.item = item
        self.children = children

    def addChild(self, node):
        self.children[node.item] = node
        node.parent=self

class MPTree(object):
    def __init__(self):
        self.root = Node(None, {})
        self.summaries = {}
    def add_transaction(self,transaction):
        curr_node=self.root
        # #print(transaction)
        transaction.sort()
        # #print("adding",transaction)
        for i in range(len(transaction)):
            if transaction[i] not in curr_node.children:
                new_node=MNode(transaction[i],{})
                curr_node.addChild(new_node)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].insert(0,new_node)
                else:
                    self.summaries[transaction[i]]=[new_node]                    
                curr_node=new_node                
            else:
                curr_node=curr_node.children[transaction[i]]

    def checkerSub(self,items):
        items.sort(reverse=True)
        sitem=items[0]
        if sitem not in self.summaries:
            return 1
        else:
            if len(items)==1:
                return 0
        for t in self.summaries[sitem]:
            cur=t.parent
            i=1
            e=0
            while cur.item != None:
                if items[i]==cur.item:
                    i+=1
                    if i==len(items):
                        return 0
                cur=cur.parent
        return 1


                
maximalTree=MPTree()
maximalItemsets=[]
def getPer_Sup(tids):
    tids.sort()
    cur=0
    per=0
    sup=0
    for j in range(len(tids)):
        per=max(per,tids[j]-cur)
        if(per>periodicity):
            return [0,0]
        cur=tids[j]
        sup+=1
    per=max(per,lno-cur)
    return [sup,per]

def cond_trans(cond_pat,cond_tids):
    pat=[]
    tids=[]
    data1={}
    for i in range(len(cond_pat)):
        for j in cond_pat[i]:
            if j in data1:
                data1[j]= data1[j] + cond_tids[i]
            else:
                data1[j]=cond_tids[i]

    up_dict={}
    for m in data1:
        up_dict[m]=getPer_Sup(data1[m])
    up_dict={k: v for k,v in up_dict.items() if v[0]>=minSup and v[1]<=periodicity}
    count=0
    for p in cond_pat:
        p1=[v for v in p if v in up_dict]
        trans=sorted(p1, key= lambda x: (up_dict.get(x)[0],-x), reverse=True)
        if(len(trans)>0):
            pat.append(trans)
            tids.append(cond_tids[count])
        count+=1
    return pat,tids,up_dict

def generate_dict(transactions):
    global rank
    data={}
    for tr in transactions:
        for i in range(1,len(tr)):
            if tr[i] not in data:
                data[tr[i]]=[int(tr[0]),int(tr[0]),1]
            else:
                data[tr[i]][0]=max(data[tr[i]][0],(int(tr[0])-data[tr[i]][1]))
                data[tr[i]][1]=int(tr[0])
                data[tr[i]][2]+=1
    for key in data:
        data[key][0]=max(data[key][0],lno-data[key][1])

    data={k: [v[2],v[0]] for k,v in data.items() if v[0]<=periodicity and v[2]>=minSup}
    genList=[k for k,v in sorted(data.items(),key=lambda x: (x[1][0],x[0]),reverse=True)]
    rank = dict([(index,item) for (item,index) in enumerate(genList)])
    # genList=[k for k,v in sorted(data.items(),key=lambda x: (x[1][0],x[0]),reverse=True)]
    return data,genList


def update_transactions1(list_of_transactions,dict1,rank):
    list1=[]
    for tr in list_of_transactions:
        list2=[int(tr[0])]
        for i in range(1,len(tr)):
            if tr[i] in dict1:
                list2.append(rank[tr[i]])                       
        if(len(list2)>=2):
            basket=list2[1:]
            basket.sort()
            list2[1:]=basket[0:]
            list1.append(list2)
    return list1
def build_tree(data,info):
    root_node=Tree()
    root_node.info=info.copy()
    for i in range(len(data)):
        set1=[]
        set1.append(data[i][0])
        root_node.add_transaction(data[i][1:],set1)
    return root_node 
lno=0
rank={}
rank2={}

def main(data):
    global pfList,lno,rank2
    list_of_transactions = data

    generated_dict,pfList=generate_dict(list_of_transactions)
    # print("No. of single items:",len(pfList))
    #print(pfList)   
    updated_transactions1=update_transactions1(list_of_transactions,generated_dict,rank)
    # print(updated_transactions1)
    info={rank[k]: v for k,v in generated_dict.items()}
    list_of_transactions=[]
    Tree = build_tree(updated_transactions1,info)
    q = Tree.generate_patterns([])
    return q, info

######## Execution CONFIGURATION ###############
print('INPUT')
datasetFile = 'Journal_DATA.csv'

# dataset
print('LOAD DATA')
dataset = pd.read_csv(datasetFile, header=None, index_col=False)
dataset.columns = ['datetime', 'step', 'x', 'y', 'id', 'length']

# define time milestones
start_time = end_time = 201505010000
cnter_month = 0
days_of_months = {5:31, 6:30, 7:31, 8:31, 9:30, 10:31}
offset = 100 # 2 hours
end_date = 2400
one_date = 10000
num_milestones = 30000
milestone_length = 200 # 6 hours
 

milestones = []
for i in range(num_milestones):
  # check new date
  if (start_time % one_date) >= end_date:
    offset_date = start_time % one_date
    offset_date -= end_date
    start_time -= end_date
    start_time += one_date
  
  # check new month
  dd = (start_time // one_date) % 100
  mm = (start_time // one_date // 100) % 100
  if mm > 10:
    break
  if dd == days_of_months[mm] + 1:
    start_time -= ((days_of_months[mm]+1) * one_date)
    start_time += (one_date * 100) 
    start_time += one_date
    
  # calculate end_time
  end_time = start_time + milestone_length
  
  # check new date
  if (end_time % one_date) >= end_date:
    offset_date = end_time % one_date
    offset_date -= end_date
    end_time -= end_date
    end_time += one_date
    
  # check new month
  dd = (end_time // one_date) % 100
  mm = (end_time // one_date // 100) % 100
  if mm > 10:
    break
  if dd == days_of_months[mm] + 1:
    end_time -= ((days_of_months[mm]+1) * one_date)
    end_time += (one_date * 100) 
    end_time += one_date

  milestone = [start_time, end_time]
  milestones.append(milestone)
  start_time += offset
milestones = milestones[:-3]
print(milestones)
######## INPUT ###############
# filepath = 'results/patterns_Length_minSup_periodicity_periodicSup.csv'
FILEPATH = './results_maxPFP/patterns_{0}_{1}_{2}_{3}_BigData_app.csv'

MAX_PATTERNS = 0

# PARAMETERS
lengths = [350]#,250] #[200,250, 300, 350, 400, 450]
minSups = [.3] #[.55, .65, .75, .85, .95]
periodicitys = [12]#[2, 6, 12, 24, 48]
periodicSups = [.5]#[.55, .65, .75, .85, .95]
minSup = None
periodicity = None

import itertools
import time
runtime = open('runtime_maxpfp_app.csv', 'w')
for combination in list(itertools.product(*[lengths, minSups, periodicitys, periodicSups])):
  start = time.time()
  # print([lengths, minSups, periodicitys, periodicSups])
  # print(combination)
  filepath = FILEPATH.format(combination[0], combination[1], combination[2], combination[3])
  print(filepath)
  outputFile = open(filepath, 'w')
  for stage in range(len(milestones)):
    num_pattern = 0
    # extract data based on milestones
    # print('EXTRACT')
    milestone = milestones[stage]
    data = dataset.loc[(dataset['datetime'] >= milestone[0]) & (dataset['datetime'] <= milestone[1])]  
    data = data.loc[data['length'] > combination[0]]
    data = data.loc[data['step'] == 0]
    data = data.reset_index(drop=True)
    if data.shape[0] == 0:
      continue

    maximalTree=MPTree()
    maximalItemsets=[]

    ##### load and clean data #####  
    # transactions
    transactions = convert2RowFormatTransactions(data)
    totalTransactions = len(transactions)

    # all are in percentage of the number of transactions (x100)
    minSup = combination[1] # threshold to consider a pattern is frequent
    periodicity = combination[2]/totalTransactions # 12 * 5-minute interval = 1 hour
    # minNumberOfPeriodicOccurrences = combination[3] # threshold to consider a pattern is periodic-frequent

    ###### extract patterns #######
    # print(periodicity, minSup)
    q, gene_list = main(transactions)

    if len(maximalItemsets) > 0:
        print('{0},{1},{2}'.format(stage, milestone[0], milestone[1]))
        outputFile.write('{0},{1},{2}'.format(stage, milestone[0], milestone[1]) + '\n')
    
    for x in maximalItemsets:
      #print(x[0], len(x[0]), ";".join(str(x) for x in sorted(x[0]))    )
      # we are only interested in finding itemset (having >= 2 items per pattern)
      num_pattern += 1
      line = ";".join(str(x) for x in sorted(x[0]))        
      outputFile.write(str(len(x[0])) + ',' + line + '\n')
        # print(line)
      # get the most 500k common patterns
      # if len(occurred_patterns) == 500000:
        # break
          
    del transactions
    del data

  outputFile.close()
  runtime.write('{0},{1}\n'.format(filepath, time.time()-start))
runtime.close()