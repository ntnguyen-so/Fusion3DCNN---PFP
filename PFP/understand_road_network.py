"""
 This file is used to generate examples of underlying road networks in high traffic demand areas.
"""
from map_utils import *

if __name__ == "__main__":
  long_patterns = ['30083;30084', '81163;82163', '82163;83163']
  medium_patterns = ['89139;89140']
  short_patterns = ['76102;77102']
  
  # Take the interested pattern defined in the lists above, and put it to "patterns" variable
  patterns = '98144;98145;98143'
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
      opacity=.3
    ).add_to(m)
        
  m.save('results/road_networks.html')
  
  
  
  
  
  
  
  
  
  
  

