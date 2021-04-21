import pandas as pd
import numpy as np
import pickle
import os

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
        self.children = children  # dictionary of children
        self.parent = None
        self.tids = set()
        self.freq = 0

    def addChild(self, node):
        self.children[node.item] = node
        node.parent = self

class Tree(object):
    def __init__(self):
        self.root = Node(None, {})
        self.summaries = {}

    def add_transaction(self, transaction, tid, freq):
        curr_node = self.root
        for i in range(len(transaction)):
            # if a co-occurrent item is not included to i'Node, add it as a child
            if transaction[i] not in curr_node.children:
                new_node = Node(transaction[i], {})
                curr_node.addChild(new_node)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(new_node) # append an item to a null list
                else:
                    self.summaries[transaction[i]] = [new_node] # extend an item to list
                curr_node = new_node
            # move node to the new node
            else:
                curr_node = curr_node.children[transaction[i]]
        curr_node.tids |= tid # add a new element to set
        curr_node.freq += freq

    def get_condition_pattern(self, alpha):
        final_patterns = []
        final_sets = []
        final_freq = []
        for i in self.summaries[alpha]:
            set1 = i.tids
            loc_f = i.freq
            set2 = []
            while (i.parent.item != None):
                set2.insert(0, i.parent.item)
                i = i.parent
            if (len(set2) > 0):
                final_patterns.append(set2)
                final_freq.append(loc_f)
                final_sets.append(set1)
        return final_patterns, final_sets, final_freq

    def remove_node(self, node_val):
        for i in self.summaries[node_val]:
            i.parent.tids |= i.tids
            i.parent.freq += i.freq
            del i.parent.children[node_val]
            i = None

    def get_ts(self, alpha, per, min_pf, min_sup):
        tid_s = set()
        freq = 0
        per_fre = 0
        valid = 0
        for i in self.summaries[alpha]:
            tid_s |= i.tids
            freq += i.freq
        # TSβ > min_sup
        if freq > min_sup:
            per_fre = get_per_fre(tid_s, per) # CalculateLocalPeriodicity
            if (per_fre < min_pf):
                valid = 0
            else:
                valid=1
        return per_fre, freq, valid

    def generate_patterns(self, prefix, per, min_pf, min_sup, genelist):
        for i in sorted(self.summaries, reverse=True):
            #Generate pattern β = ai ∪ α. Collect all of the a i's ts-lists into a temporary array, TSβ .
            per_fre, freq, valid = self.get_ts(i, per,min_pf, min_sup)
            if (valid == 1):
                pattern = prefix.copy()
                pattern.append(genelist[i])
                yield pattern, per_fre
                # print(pattern,per_fre)
                patterns, tid_summ, tid_pf = self.get_condition_pattern(i)
                conditional_tree = Tree()
                for pat in range(len(patterns)):
                    conditional_tree.add_transaction(patterns[pat], tid_summ[pat], tid_pf[pat])
                if (len(patterns) >= 1):
                    for li in conditional_tree.generate_patterns(pattern, per, min_pf, min_sup,genelist):
                        yield (li)
            self.remove_node(i)

def get_per_fre(tids, per):
    tids = list(tids)
    tids.sort()
    cur = tids[0]
    pf = 0
    for j in range(1, len(tids)):
        if (tids[j] - cur <= per):
            pf += 1
        cur = tids[j]
    return pf


def generatePFList(transactions, per_freq, min_sup, periodicity):
    data = {}
    for tr in transactions:
        for i in range(1, len(tr)):
            if type(tr[0]) != int:
                continue

            if tr[i] not in data:
                data[tr[i]] = [int(tr[0]), 1, 0] # step, frequency, prediodic
            else:                
                if ((int(tr[0]) - data[tr[i]][0]) <= periodicity): # (curStep - lastStep) <= periodicity
                    data[tr[i]][2] += 1 # increase periodic
                data[tr[i]][0] = int(tr[0]) # update last occurrence step
                data[tr[i]][1] += 1 # frequency ++

    # remove all items which are not frequent-periodic
    data = {k: v for k, v in data.items() if v[2] >= per_freq and v[1] >= min_sup } 
    return data

def createSortedCandidateListPerp(list_of_transactions, dict1, gene_li):
    rank = dict([(index, item) for (item, index) in enumerate(gene_li)])
    #print('update_transactions1:',rank) # periodic-frequent items with descending ranking

    list1 = []
    k = len(list_of_transactions)
    avg_tran_len = 0
    for tr in list_of_transactions:
        if type(tr[0]) != int:
            continue
            
        list2 = [int(tr[0])]
        for i in range(1, len(tr)):
            if tr[i] in dict1:
                list2.append(rank[tr[i]]) # appending the ranking of items

        # Create the sorted candidate item list in t be [p|P], where p is the first item and P is the remaining list.
        # it will be used to build PF-tree++
        if (len(list2) >= 2):
            # create P
            basket = list2[1:]
            avg_tran_len += len(basket)            
            basket.sort()

            # create [p|P]
            list2[1:] = basket[0:] 
            list1.append(list2)
    return list1, avg_tran_len / k

def build_tree(data):
    root_node = Tree()
    # print('build_tree:',len(data))
    for i in range(len(data)):
        set1 = set()
        set1.add(data[i][0])
        root_node.add_transaction(data[i][1:], set1, 1)
    return root_node

def get_segments(transactions, length):
    t_start = 1
    seg_id = 1
    segments = []
    sub_segment = []
    sub_segment.append(str(seg_id))
    incr = 0
    for i in transactions:
        incr += 1
        if ((int(i[0]) - t_start) > length):
            segments.append(sub_segment)
            seg_id += int((int(i[0]) - t_start) / (length + 1))
            t_start = int(i[0])
            sub_segment = []
            sub_segment.append(str(seg_id))
            incr = 1

        for j in range(1, len(i)):
            item = i[j]  # +str(incr)
            sub_segment.append(item)
    if len(sub_segment) > 1:
        segments.append(sub_segment)
    return segments

def main(data, per_freq, periodicity, min_sup):
    list_of_transactions = data

    # threshold in according to number of transactions
    total_transactions = len(list_of_transactions)
    per_freq = int(per_freq * total_transactions)
    min_sup = int(min_sup * total_transactions)
    per = int(periodicity * total_transactions)
    print('noTransactions: {0}; minSup: {1}; periodic: {2}; frequent_periodic: {3}'.format(total_transactions, min_sup, per, per_freq))
    
    # build PF-list++
    generated_dict = generatePFList(list_of_transactions, per_freq, min_sup, per)

    # Consider the remaining items in PF-list++ as periodic-frequent items and sort them with respect to their support/frequency (PI)
    gene_list = [key for key, value in sorted(generated_dict.items(), key=lambda x: x[1][1], reverse=True)] 
    
    sortedCandidates, k = createSortedCandidateListPerp(list_of_transactions, generated_dict, gene_list)
    Tree = build_tree(sortedCandidates)
    q = Tree.generate_patterns([], per, per_freq, min_sup, gene_list) #PF-growth++
    return q, gene_list

######## INPUT ###############
print('INPUT')
datasetFile = '../Fusion_3DCNN/BigData2020_JulyAugust.csv'

# dataset
print('LOAD DATA')
dataset = pd.read_csv(datasetFile, header=None, index_col=False)
dataset.columns = ['datetime', 'step', 'x', 'y', 'id', 'length']

# define time milestones
start_time = 201507010000
days_of_months = 31 # July and August have 31 days
offset = 400
end_date = 2400
num_milestones = 372
milestone_length = 10000 # 1 day

milestones = []
for i in range(num_milestones):
  # new day
  if (start_time % 10000) == end_date:
    start_time -= end_date
    start_time += 10000
  
  # new month
  if ((start_time // milestone_length) % 100) == days_of_months+1:
    start_time -=  320000
    start_time += 1010000    
    
  end_time = start_time + milestone_length
  if ((end_time // milestone_length) % 100) == days_of_months+1:
    end_time -=  320000
    end_time += 1010000
    
  milestone = [start_time, end_time]
  milestones.append(milestone)
  start_time += offset
print(milestones)
######## INPUT ###############
# filepath = 'results/patterns_Length_minSup_periodicity_periodicSup.csv'
FILEPATH = 'results/patterns_{0}_{1}_{2}_{3}_BigData.csv'

MAX_PATTERNS = 0

# PARAMETERS
lengths = [200]#,250] #[200,250, 300, 350, 400, 450]
minSups = [.75] #[.55, .65, .75, .85, .95]
periodicitys = [12]#[2, 6, 12, 24, 48]
periodicSups = [.70, .80, .90]#[.55, .65, .75, .85, .95]

import itertools
for combination in list(itertools.product(*[lengths, minSups, periodicitys, periodicSups])):
  # print([lengths, minSups, periodicitys, periodicSups])
  print(combination)
  filepath = FILEPATH.format(combination[0], combination[1], combination[2], combination[3])
  print(filepath)
  outputFile = open(filepath, 'w')
  for stage in range(len(milestones)):
    num_pattern = 0
    # extract data based on milestones
    print('EXTRACT')
    milestone = milestones[stage]
    data = dataset.loc[(dataset['datetime'] >= milestone[0]) & (dataset['datetime'] <= milestone[1])]  
    data = data.loc[data['length'] > combination[0]]
    data = data.loc[data['step'] == 0]
    data = data.reset_index(drop=True)
    print(data.head())
    
    print('{0},{1},{2}'.format(stage, milestone[0], milestone[1]))
    outputFile.write('{0},{1},{2}'.format(stage, milestone[0], milestone[1]) + '\n')

    ##### load and clean data #####  
    # transactions
    transactions = convert2RowFormatTransactions(data)
    totalTransactions = len(transactions)

    # all are in percentage of the number of transactions (x100)
    minSup = combination[1] # threshold to consider a pattern is frequent
    maxPeriodicity = combination[2]/totalTransactions # 12 * 5-minute interval = 1 hour
    minNumberOfPeriodicOccurrences = combination[3] # threshold to consider a pattern is periodic-frequent

    ###### extract patterns #######
    print(minNumberOfPeriodicOccurrences, maxPeriodicity, minSup)
    q, gene_list = main(transactions, minNumberOfPeriodicOccurrences, maxPeriodicity, minSup)
    
    for x in q:
      # we are only interested in finding itemset (having >= 2 items per pattern)
      if len(x[0]) > 1:
        num_pattern += 1
        line = ";".join(str(x) for x in sorted(x[0]))        
        outputFile.write(str(len(x[0])) + ',' + line + ',' + str(x[1]) + '\n')
        if num_pattern % 5000 == 0:
          print(num_pattern, combination)
        # print(line)
      # get the most 500k common patterns
      # if len(occurred_patterns) == 500000:
        # break
          
    del transactions
    del data

  outputFile.close()

