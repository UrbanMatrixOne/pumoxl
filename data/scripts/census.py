
import pandas as pd, geopandas as gpd
from shapely.geometry import Point
import cenpy as cen
import us
import pathlib
import urllib.request
import matplotlib.pyplot as plt

TIGER_URL_2018 = "https://www2.census.gov/geo/tiger/TIGER2018"
TIGER_DATA_PATH = "./data/TIGER"
con = cen.base.Connection('ACSDT5Y2017')
cols = con.varslike('B01001_001')
cols += sorted(con.varslike('B17026_'))

SHAPEFILE_CACHE = {}
def downloadFile(data_file, url):
    if not data_file.is_file():
        with urllib.request.urlopen(url) as resp, \
                open(data_file, "wb") as f:
            f.write(resp.read())

def analyzeDataFrame(df):
    pass


def getStates():
    states_filename = "tl_2018_us_state.zip"
    states_url = f"{TIGER_URL_2018}/STATE/{states_filename}"
    states_file = pathlib.Path(TIGER_DATA_PATH, states_filename)
    downloadFile(states_file, states_url)

    return(gpd.read_file(f'zip://{states_file}'))

def getCounties(state_id):
    county_filename = "tl_2018_us_county.zip"
    county_url = f"{TIGER_URL_2018}/COUNTY/{county_filename}"
    county_file = pathlib.Path(TIGER_DATA_PATH,  county_filename)
    downloadFile(county_file, county_url)

    county_geo = gpd.read_file(f"zip://{county_file}")
    return(county_geo[county_geo.STATEFP == state_id])

def getCensusTracts(state_id):
    #download imagery if needed
    tract_filename = f"tl_2018_{state_id}_tract.zip"
    tract_url = f"{TIGER_URL_2018}/TRACT/{tract_filename}"
    tract_file = pathlib.Path(TIGER_DATA_PATH,tract_filename)
    downloadFile(tract_file, tract_url)
    
    #check cache for census tract data
    data = pd.DataFrame()
    if state_id in SHAPEFILE_CACHE.keys():
        data = SHAPEFILE_CACHE[state_id]
    else:
    #pull tract data for entire state from census api 
        g_unit = 'tract:*'
        #state_filter ='state:*'
        state_filter = {'state':state_id} 
        data = con.query(cols, geo_unit= g_unit, geo_filter=state_filter)
        #store data in cache
        SHAPEFILE_CACHE[state_id] =data

    #perform some analysis
    data['pct_under_poverty_line'] = (data.B17026_002E +data.B17026_003E +data.B17026_004E)/data.B17026_001E
    
    #merge the census data and the geo data.
    tract_geo = gpd.read_file(f"zip://{tract_file}").merge(data, left_on = ('STATEFP','COUNTYFP','TRACTCE') , right_on = ('state','county','tract'))
    return(tract_geo)

def makeGeoDataFrame(Latitude_in,Longitude_in):
    df = pd.DataFrame({'Latitude' : Latitude_in, 'Longitude' : Longitude_in}, index=[0] )
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    gdf =  gpd.GeoDataFrame(df, geometry= 'Coordinates')
    gdf.crs = {'init': 'epsg:4269'}
    return(gdf)

def saveMapImage(gdp,metric, img_filename):
    gdp.plot(column = metric, legend = True)
    plt.savefig(f'{img_filename}.png')

def getStateFIPSForPoint(user_point):
    return(gpd.sjoin(getStates(), user_point).STATEFP.to_string(index= False))

def getDataAndImageForPoint(Latitude_in,Longitude_in,metric):
    user_point = makeGeoDataFrame(Latitude_in,Longitude_in)
    user_point['buffer'] = user_point.buffer(.05)
    user_buffer = gpd.GeoDataFrame(user_point, geometry = 'buffer', crs = user_point.crs)
    
    #get tract level data fror state
    state_fips = getStateFIPSForPoint(user_point)
    tract_geodata = getCensusTracts(state_fips)
    tract_geodata_buffer = gpd.sjoin(tract_geodata,user_buffer)
    base = tract_geodata_buffer.plot(metric,legend = True)     
    user_point.plot(ax = base,marker = '+',color = 'red',markersize = 100)
    plt.savefig(f'lon_{Longitude_in}_lat{Latitude_in}_{metric}.png')
    return (tract_geodata_buffer)
    
    #generateImgForPoint(34.0522,-118.2436,metric= 'pct_under_poverty_line')
