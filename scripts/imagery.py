import urllib.request
from PIL import Image
import numpy as np

def downloadGoogleImage(lat, lon,pixel_dim=224, zoom_factor =18 , full_path = None):
    if full_path == None:
        full_path = f'./google_image_lat_{lat}_lon_{lon}_zoom_{zoom_factor}.png'
    key = 'AIzaSyAig-Wkbj1Rw-6ElabUWK_JlWvOpn57YJs'
    url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' + \
    str(lon) + '&zoom='+str(zoom_factor)+'&size='+str(pixel_dim)+'x'+str(pixel_dim+50)+'&maptype=satellite&key=' + key

    urllib.request.urlretrieve(url, full_path )
    
    #open image as numpy array
    image = Image.open(full_path)
    image.load()
    image_np = np.asarray(image.convert('RGB'))
    image_np = image_np[25:-25,:,:]
    return(image_np)

#from matplotlib import pyplot as plt
#plt.imshow(downloadGoogleImage(46.929701,-94.359045), interpolation='nearest')
#plt.show()
#Image.fromarray(downloadGoogleImage(46.929701,-94.359045), 'RGB')