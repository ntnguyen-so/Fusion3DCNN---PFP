"""
  This file is used to generate sets of alert codes for long-/medium-/short-term respectively.
  The output of this process is the lists of locations that will have high traffic demand.
  The lists can be plotted by using the function of understand_road_network.py.
"""

import pickle
from map_utils import *

def read_pattern(result_patterns_file):
  result_patterns_file = open(result_patterns_file, 'r')
  lines = result_patterns_file.readlines()
  patterns_days = []
  patterns_day = []
  days_id = []
  for line in lines:
    if ';' not in line:
      days_id.append(line.split(',')[1])
      print(line.split(',')[1])
      if len(patterns_day) > 0:
        patterns_days.append(patterns_day)
        patterns_day = []
    else:
      line = line.split(',')[1]
      patterns_day.append(line)
  patterns_days.append(patterns_day)
  result_patterns_file.close()

  print('num milestones:', len(patterns_days))
  for i in range(len(patterns_days)):
    print(i, 'num patterns:', len(patterns_days[i]))
  
  return patterns_days
  
def get_topk_patterns(collected_patterns, k):
  order = {k: v for k, v in sorted(collected_patterns.items(), key=lambda item: item[1], reverse=True)}
  return list(order.keys())[:min(k, len(order))]
  
# build alert codes for long-/medium-/short-term patterns respectively
# patterns_days: patterns extracted
# k: number of alert codes
def build_longterm_alert_code(patterns_days, k=10):
  collected_patterns = {}
  steps = 6
  for start in range(len(patterns_days)-6):
    patterns_day0 = patterns_days[start]
    for pattern in patterns_day0:
      appear = True
      for patterns_day in patterns_days[start:start+steps]:
        if pattern not in patterns_day:
          appear = False
          break
          
      if appear == True:
        if pattern in collected_patterns:
          collected_patterns[pattern] += 1
        else:
          collected_patterns[pattern] = 1
          
  return get_topk_patterns(collected_patterns, k)
  
def build_mediumterm_alert_code(patterns_days, k=10):
  collected_patterns = {}
  stepss = [3,4,5]
  for steps in stepss:
    for start in range(len(patterns_days)-steps):
      patterns_day0 = patterns_days[start+1]
      for pattern in patterns_day0:
        appear = True
        for patterns_day in patterns_days[start+1:start+steps+1]:
          if pattern not in patterns_day:
            appear = False
            break
          
      if appear == True:
        if (pattern not in patterns_days[start]) and \
           (pattern not in patterns_days[start+steps+1]):
            if pattern in collected_patterns:
              collected_patterns[pattern] += 1
            else:
              collected_patterns[pattern] = 1
  
  return get_topk_patterns(collected_patterns, k)
  
def build_shortterm_alert_code(patterns_days, k=10):
  collected_patterns = {}
  stepss = [1,2]
  for steps in stepss:
    for start in range(len(patterns_days)-steps):
      patterns_day0 = patterns_days[start+1]
      for pattern in patterns_day0:
        appear = True
        for patterns_day in patterns_days[start+1:start+steps+1]:
          if pattern not in patterns_day:
            appear = False
            break
          
      if appear == True:
        if (pattern not in patterns_days[start]) and \
           (pattern not in patterns_days[start+steps+1]):
            if pattern in collected_patterns:
              collected_patterns[pattern] += 1
            else:
              collected_patterns[pattern] = 1
  
  return get_topk_patterns(collected_patterns, k)
  
# This is to generate alert codes presented in the paper
def generate_alertcode_examples():
  ## UDML 2020 ##
  long_patterns = ['30083;30084', '81163;82163', '82163;83163']
  medium_patterns = ['29083;30083;30084', '97145;98144;98145']
  short_patterns = ['97146;98144', '29083;30083']

  ## BigData 2020 ##
  long_patterns = ['81163;82163;83163', '29083;30084']
  medium_patterns = ['98142;98143', '63174;64175']
  short_patterns = ['97146;98144', '29083;30083']
  
  # Take the interested pattern defined in the lists above, and put it to "patterns" variable
  patterns = '97145;98142'
  inattentive = get_inattentive(patterns)
  locations = patterns.split(';')
  m = createBaseMap()
  for i in range(len(locations)):
    location = locations[i]
    path = './base_figures/heavy.png'
    img = Image.open(path)
    img.save(path)
    location = loc2list(location)
    folium.raster_layers.ImageOverlay(
      image=path, 
      bounds=[relativeloc2Coordinate(location), relativeloc2Coordinate([x + 1 for x in location])],
      opacity=.5
    ).add_to(m)

  for i in range(len(inattentive)):
    location = inattentive[i] 
    path = './base_figures/light.png'
    img = Image.open(path)
    folium.raster_layers.ImageOverlay(
      image=path, 
      bounds=[relativeloc2Coordinate(location), relativeloc2Coordinate([x + 1 for x in location])],
      opacity=.5
    ).add_to(m)
        
  m.save('results/alertcodes.html')
  
if __name__ == "__main__":
  ##########################################################
  ##                   Generate patterns                  ##
  ##########################################################
  K = 500
  patterns_days = read_pattern('results/patterns_200_0.95_12_0.9_BigData.csv')
  # longterm_alertcode = build_longterm_alert_code(patterns_days, K)
  # medium_alertcode = build_mediumterm_alert_code(patterns_days, K)
  shortterm_alertcode = build_shortterm_alert_code(patterns_days, K)
  
  # print(longterm_alertcode)
  # print(medium_alertcode)
  print(shortterm_alertcode)
  
  ##########################################################
  ##             Draw examples of patterns                ##
  ##########################################################
  generate_alertcode_examples()