import urllib.request
import os

#download data files if necessary
if not (os.path.isfile('./data/USA_Detailed_Water_Bodies.zip')):
    print('file not found, downloading')
    urllib.request.urlretrieve('http://geodata.utc.edu/datasets/48c77cbde9a0470fb371f8c8a8a7421a_0.zip', './data/USA_Detailed_Water_Bodies.zip')
    
else:
    print("found './data/USA_Detailed_Water_Bodies.zip'")