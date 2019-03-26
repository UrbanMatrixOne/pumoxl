import geopandas as gpd
import geofeatures
import matplotlib.pyplot as plt
import imagery
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
    print(water_geodata_buffer.shape)
    #make plot
    
    satellite_img = imagery.downloadGoogleImage(Latitude_in,Longitude_in)
    satellite_ax = plt.imshow(satellite_img, interpolation='nearest')

    water_geodata_buffer.plot(ax = satellite_ax)
    
    user_point.plot(ax = base, marker = '+',color = 'red',markersize = 100)
    filename_out = f'data/lon_{Longitude_in}_lat{Latitude_in}_water_{buffer_dist}.png'
    plt.savefig(filename_out)
    

getWaterDataForPoint(  Latitude_in,Longitude_in )