import pandas as pd, geopandas as gpd
from shapely.geometry import Point
def makeGeoDataFrame(Latitude_in,Longitude_in, crs = {'init': 'epsg:4269'}):
    df = pd.DataFrame({'Latitude' : Latitude_in, 'Longitude' : Longitude_in}, index=[0] )
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    gdf =  gpd.GeoDataFrame(df, geometry= 'Coordinates')
    gdf.crs =crs
    return(gdf)
    
