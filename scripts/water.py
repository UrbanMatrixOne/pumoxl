import geopandas as gpd, numpy as np
from scripts import geofeatures,imagery,create_earth_image_raster
import matplotlib.pyplot as plt
import rasterio
water_path = 'data/USA_Detailed_Water_Bodies.zip'
water_data = gpd.read_file(f'zip://{water_path}')

Latitude_in,Longitude_in,buffer_dist = 33.5, -117.74, .05 
#Latitude_in,Longitude_in,buffer_dist = 33.671622, -117.0307, .05 
#Latitude_in,Longitude_in,buffer_dist = 46.929701, -94.359045, .05 

def getWaterDataForPoint(Latitude_in,Longitude_in,buffer_dist = .05):
    user_point = geofeatures.makeGeoDataFrame(Latitude_in,Longitude_in ,water_data.crs)
    user_point['buffer'] = user_point.buffer(buffer_dist)
    user_buffer = gpd.GeoDataFrame(user_point, geometry = 'buffer', crs = user_point.crs)

    #merge water data with buffer
    water_geodata_buffer = gpd.sjoin(water_data,user_buffer)
    return water_geodata_buffer
    #make plot
    
#    satellite_img = imagery.downloadGoogleImage(Latitude_in,Longitude_in)
"""   imagery_dict = create_earth_image_raster.generate_for_coord(Latitude_in,Longitude_in,18)
    with source as rasterio.open(imagery_dict['tif'],'r'):
        red = source.read(1)
        green = source.read(2)
        blue = source.read(3)
        pix = np.dstack((red, green, blue))
        bounds = (source.bounds.left, source.bounds.right, \
                source.bounds.bottom, source.bounds.top)

        f = plt.figure(figsize=(6, 6))
        ax = plt.imshow(pix, extent=bounds)
        water_geodata_buffer.plot(ax = ax)
        
        filename_out = f'data/lon_{Longitude_in}_lat{Latitude_in}_water_{buffer_dist}.png'
        plt.savefig(filename_out )"""


    


""" 
#
#
#
#

import geopandas as gpd, numpy as np
from scripts import geofeatures,imagery,create_earth_image_raster
import matplotlib.pyplot as plt
import rasterio
water_path = 'data/USA_Detailed_Water_Bodies.zip'
water_data = gpd.read_file(f'zip://{water_path}')

Latitude_in,Longitude_in,buffer_dist = 33.5, -117.74, .05 

user_point = geofeatures.makeGeoDataFrame(Latitude_in,Longitude_in ,water_data.crs)
user_point['buffer'] = user_point.buffer(buffer_dist)
user_buffer = gpd.GeoDataFrame(user_point, geometry = 'buffer', crs = user_point.crs)

#merge water data with buffer
water_geodata_buffer = gpd.sjoin(water_data,user_buffer)
print(water_geodata_buffer.shape)
#make plot

#    satellite_img = imagery.downloadGoogleImage(Latitude_in,Longitude_in)
imagery_dict = create_earth_image_raster.generate_for_coord(Latitude_in,Longitude_in,18)
source = rasterio.open(imagery_dict['tif'],'r')
red = source.read(1)
green = source.read(2)
blue = source.read(3)
pix = np.dstack((red, green, blue))
bounds = (source.bounds.left, source.bounds.right, \
        source.bounds.bottom, source.bounds.top)

f = plt.figure(figsize=(6, 6))
ax = plt.imshow(pix, extent=bounds)
water_geodata_buffer.plot(ax = ax)

filename_out = f'data/lon_{Longitude_in}_lat{Latitude_in}_water_{buffer_dist}.png'
plt.savefig(filename_out) """