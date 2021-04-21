"""
 This file is used to plot traffic-congested lengths of the specified locations on the map
"""
from map_utils import *
  
if __name__ == "__main__":
  #UDML 2020
  patterns ='29083;74100;76101;76102;77102;79104;89141'
  start_timestamp = 201507052000
  timestamps = [201507052000]
  excluded = [0,1,2,3, 23,22,21,20, 19,18,17,16]

  datasetFile = '../Fusion_3DCNN/BigData2020_JulyAugust.csv'
  dataset = pd.read_csv(datasetFile, header=None, index_col=False)
  dataset.columns = ['datetime', 'step', 'x', 'y', 'id', 'length']
  
  # example 1: This is to draw traffic-congested situations for a medium-term pattern
  print('start example 1')  
  patterns ='89143;98142'
  start_timestamp = 201507071200
  timestamps = [201507071200]#[201507052000, 201507060000, 201507060400, 201507060800, 201507061200, 201507061600]
  for i in range(23):
    start_timestamp += 100
    if (start_timestamp % 10000) // 2400 == 1:
        start_timestamp -=  2400
        start_timestamp += 10000
    timestamps.append(start_timestamp)
  
  excluded = [0,1,2,3, 23,22,21,20]
  locations = patterns.split(';')
  m = createBaseMap()
  for i in range(len(locations)):
    location = locations[i]
    lengths = extract_length(location, timestamps, dataset)
    path = congested_plot_image(lengths, location, 400, excluded)
    print(path)
    img = Image.open(path)
    img = ImageOps.expand(img,border=2,fill='black')
    img.save(path)
    location = loc2list(location)
    folium.raster_layers.ImageOverlay(
      image=path, 
      bounds=[relativeloc2Coordinate(location), relativeloc2Coordinate([x + 1 for x in location])],
      opacity=.7
    ).add_to(m)
    
  m.save('results/plan_resources_1.html')
  return
  # example 2: This is to draw traffic-congested situations for a long-term pattern
  print('start example 2')  
  patterns = '97146;98145;97145'  
  start_timestamp = 201508220400
  timestamps = [start_timestamp]#[201508220400, 201508220800, 201508221200, 201508221600, 201508222000, 201508230000]
  for i in range(23):
    start_timestamp += 100
    if (start_timestamp % 10000) // 2400 == 1:
        start_timestamp -=  2400
        start_timestamp += 10000
    timestamps.append(start_timestamp)
    
  excluded = []
  locations = patterns.split(';')
  m = createBaseMap()
  for i in range(len(locations)):
    location = locations[i]
    lengths = extract_length(location, timestamps, dataset)
    path = congested_plot_image(lengths, location, 550, excluded)
    print(path)
    img = Image.open(path)
    img = ImageOps.expand(img,border=2,fill='black')
    img.save(path)
    location = loc2list(location)
    folium.raster_layers.ImageOverlay(
      image=path, 
      bounds=[relativeloc2Coordinate(location), relativeloc2Coordinate([x + 1 for x in location])],
      opacity=.7
    ).add_to(m)
    
  m.save('results/plan_resources_2.html')
  
  
  
  
  
  
  

