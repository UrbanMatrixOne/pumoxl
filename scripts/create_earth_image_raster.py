import click 
import rasterio
import os,pathlib
import glob
from PIL import Image
import numpy as np
import math
import urllib.request
import pandas as pd

#functions that specify the size of one pixel
def degrees_longitude_per_pixel(zoom_level):
    num_pixels_per_osm_tile =256
    return 360* 2**(-int(zoom_level))/num_pixels_per_osm_tile

def degrees_latitude_per_pixel(zoom_level, latitude):
    lat_lon_ratio = math.cos(latitude *math.pi / 180)
    num_pixels_per_osm_tile =256
    return lat_lon_ratio* 360* 2**(-int(zoom_level))/num_pixels_per_osm_tile

def meters_per_pixel(zoom_level,latitude):
    return((40075016.686/256) * math.cos(math.radians(latitude)) / (2**zoom_level))

def meters_per_degree(zoom_level,latitude):
    return(meters_per_pixel(zoom_level,latitude) /degrees_longitude_per_pixel(zoom_level) )

  
#download a single image    
def downloadGoogleImage(lat , lon, zoom_level = 18, pixel_dim = 992, img_dir = './data/images/png/' , buffer =25):
    img_filename = os.path.join(img_dir, f'google_image_lat_{lat}_lon_{lon}_zoom_{zoom_level}.png')
    key = 'AIzaSyAig-Wkbj1Rw-6ElabUWK_JlWvOpn57YJs'
    lat_lon_ratio =degrees_longitude_per_pixel(zoom_level)/ degrees_latitude_per_pixel(zoom_level, lat)
    x_dim = pixel_dim
    y_dim = int(pixel_dim*(1+(lat_lon_ratio-1)*2)+buffer*2)
    
    url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' + \
    str(lon) + '&zoom='+str(zoom_level)+'&size='+str(x_dim)+'x'+str(y_dim)+'&maptype=satellite&key=' + key
    
    urllib.request.urlretrieve(url, img_filename )
    return(img_filename)
    

#convert a PNG image to TIF
def savePNGasTIF(lat,lon,zoom_level =18, img_dir ='./', tif_dir =  './data/images/png/',buffer =25):
    png_img_filename = os.path.join(img_dir, f'google_image_lat_{lat}_lon_{lon}_zoom_{zoom_level}.png')
    tif_filename =  os.path.join(tif_dir, f'google_image_lat_{lat}_lon_{lon}_zoom_{zoom_level}.tif')
    
    image = Image.open(png_img_filename)
    image.load()
    image_np = np.asarray(image.convert('RGB'))
    image_np = image_np[buffer:image_np.shape[0]-buffer,:,:]
    
    image_height = image_np.shape[0]
    image_width  = image_np.shape[1]
    
    lat_lon_ratio =degrees_longitude_per_pixel(zoom_level)/ degrees_latitude_per_pixel(zoom_level, lat)
    
    west_border = lon - degrees_longitude_per_pixel(zoom_level) * image_width /2 
    north_border= lat + degrees_longitude_per_pixel(zoom_level) * image_height /(2*lat_lon_ratio) #  *lat_lon_ratio
    res =    degrees_longitude_per_pixel(zoom_level) 
    res_lat = degrees_latitude_per_pixel(zoom_level,lat )
    #transform = rasterio.transform.from_origin(west_border - res/2 , north_border + res/2, res,res)#*lat_lon_ratio
    transform = rasterio.transform.from_origin(west_border  , north_border , res,res_lat)#*lat_lon_ratio
    
    #open a raster in writing mode 
    newfile = rasterio.open(tif_filename, 'w', driver = 'GTIFF',
              height = image_height,
              width  = image_np.shape[1],
              count  = 3,
              dtype = np.uint8,
              crs    = '+proj=latlong',#rasterio.crs.CRS.from_dict(init = 'epsg:4326'),
              transform = transform, 
              nodata = None, 
             )
    newfile.write(np.moveaxis(image_np,-1,0).astype(np.uint8))
    newfile.close()
    return(tif_filename)

#combine rasters
def combineRasters(raster_paths, filename_out ):
    datasets = []
    for file in raster_paths:
        datasets.append(rasterio.open(file,'r'))
    merged_dataset,transform =rasterio.merge.merge(datasets)
    
    newfile  = rasterio.open(filename_out, 'w', driver = 'GTIFF',
                  height = merged_dataset.shape[1],
                  width  = merged_dataset.shape[2],
                  count  = 3,
                  dtype = np.uint8,
                  crs    = '+proj=latlong',#rasterio.crs.CRS.from_dict(init = 'epsg:4326'),
                  transform = transform, 
                  nodata = None, 
                 )
    newfile.write(merged_dataset)
    newfile.close()

def generate_for_coord(latitude,longitude, zoom_level, img_dir ='./data/images'):
    width = 224
    
    png_path =  os.path.join(img_dir,'png/')
    tif_path =  os.path.join(img_dir,'tif/')

    png_out = downloadGoogleImage(latitude,longitude,pixel_dim=width, img_dir= png_path ,buffer=25)
    tif_out = savePNGasTIF(latitude,longitude,img_dir= png_path, tif_dir= tif_path,buffer=25)
    return {'png':png_out,'tif':tif_out}

def generate_for_bbox(bbox, zoom_level = 18):
#def createImagesAndRastersForBoundingBox(west,south,east,north, zoom_level = 18):
    #bbox = [west,south,east,north]
    (west,south,east,north) = bbox
    #width = 400
    width = 535

    degrees_per_pixel = degrees_longitude_per_pixel(zoom_level) 
    degrees_per_image_wide = degrees_per_pixel*width
    lat_lon_ratio =lat_lon_ratio =degrees_longitude_per_pixel(zoom_level)/ degrees_latitude_per_pixel(zoom_level, (north+south)/2)

    height = int(width *(1+(lat_lon_ratio-1)*2))
    degrees_per_image_tall = height *degrees_per_pixel * lat_lon_ratio
    rasters= []
    print(f'will generate {math.ceil((bbox[2]-bbox[0])/degrees_per_image_wide)*math.ceil((bbox[3]-bbox[1])/degrees_per_image_tall)} images')
    for longitude in np.arange(bbox[0],bbox[2], degrees_per_image_wide):
        for latitude in np.arange(bbox[1],bbox[3], degrees_per_image_wide):#
            downloadGoogleImage(latitude,longitude,pixel_dim=width, img_dir= './png',buffer=25)
            rasters.append(savePNGasTIF(latitude,longitude,img_dir= './png', tif_dir='./tif',buffer=25))

    print(f'generated {len(rasters)} images')

#@click.group()
#def raster_command():
#    pass

#@raster_command.command()
#@click.option('--file', type=click.Path())
def csv(file):
    coords = pd.read_csv(file)
    for index, row in coords.iterrows():
        downloadGoogleImage(row.lat, row.lon, row.zoom_level, row.pixel_dim, row.img_dir  )
        savePNGasTIF(row.lat,row.lon,  row.zoom_level,    img_dir= row.img_dir ,  tif_dir=row.tif_dir)
#iterate over bounding box
#@raster_command.command()
#@click.option('--bbox', nargs=4, type=float)
def box(bbox, zoom_level = 18):
#def createImagesAndRastersForBoundingBox(west,south,east,north, zoom_level = 18):
    #bbox = [west,south,east,north]
    (west,south,east,north) = bbox
    #width = 400
    width = 535

    degrees_per_pixel = degrees_longitude_per_pixel(zoom_level) 
    degrees_per_image_wide = degrees_per_pixel*width
    lat_lon_ratio =lat_lon_ratio =degrees_longitude_per_pixel(zoom_level)/ degrees_latitude_per_pixel(zoom_level, (north+south)/2)

    height = int(width *(1+(lat_lon_ratio-1)*2))
    degrees_per_image_tall = height *degrees_per_pixel * lat_lon_ratio
    rasters= []
    print(f'will generate {math.ceil((bbox[2]-bbox[0])/degrees_per_image_wide)*math.ceil((bbox[3]-bbox[1])/degrees_per_image_tall)} images')
    for longitude in np.arange(bbox[0],bbox[2], degrees_per_image_wide):
        for latitude in np.arange(bbox[1],bbox[3], degrees_per_image_wide):#
            downloadGoogleImage(latitude,longitude,pixel_dim=width, buffer=25)
            rasters.append(savePNGasTIF(latitude,longitude,buffer=25))

    print(f'generated {len(rasters)} images')



if __name__ =='__main__':
    raster_command()
    #createImagesAndRastersForBoundingBox()
    #createImagesAndRastersForBoundingBox(28.35,-15.4 ,28.4,-15.35)

