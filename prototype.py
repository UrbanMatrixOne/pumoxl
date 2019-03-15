import xlwings as xw
import matplotlib.pyplot as plt 
import pandas as pd, numpy as np
import geopandas
from shapely.geometry import Point
import os, sys, re 
import urllib.request


from keras.applications.mobilenetv2 import preprocess_input
from keras.preprocessing import image
import keras.applications 
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers

from absl import flags
from scipy import misc
import data.scripts.census

WORLD_LOWRES = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

SHOW_IMAGES = True

print('finished imports ') 

def makeGeoDataFrame(Latitude_in,Longitude_in):
    df = pd.DataFrame({'Latitude' : Latitude_in, 'Longitude' : Longitude_in}, index=[0] )
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'] .apply(Point)
    gdf =  geopandas.GeoDataFrame(df, geometry= 'Coordinates')
    return(gdf)

@xw.func
def getPythonVersion():
    return ( str(sys.version_info))

@xw.func
def getCondaEnv():
    return os.environ['CONDA_DEFAULT_ENV']
@xw.func
def getPythonExe():
    return(sys.executable)

@xw.func
def getPythonWD():
    return(os.getcwd())

@xw.func
def coord2country(Latitude_in,Longitude_in):
    gdf = makeGeoDataFrame(Latitude_in,Longitude_in)
    gdf_country = geopandas.sjoin(gdf, WORLD_LOWRES, how = 'inner', op = 'within')

    return (gdf_country.name[0])  

@xw.func
def coord2NERC(Latitude_in,Longitude_in):
    NERC_REGIONS = geopandas.read_file(os.path.abspath('./data/NERC_Regions/NERC_Regions.shp'))
    gdf = makeGeoDataFrame(Latitude_in,Longitude_in)
    gdf_country = geopandas.sjoin(gdf, NERC_REGIONS, how = 'inner', op = 'within')
    NERC_Acronym = re.search('\(([^)]+)', gdf_country.SUBNAME[0]).group(1) 
    return (NERC_Acronym)  

#capital recovery factor
@xw.func
def CRF(i, n):
    return (float( i*(1+i)**n /  (((1+i)**n)-1)))

@xw.func
@xw.arg('overnight_capital_cost', doc = 'This is the construction cost per MW capacity')
@xw.arg('fixed_om_cost', doc = 'This is the fixed annual O&M cost per MW capacity')
@xw.arg('variable_om_cost', doc = 'This is the variable annual O&M cost (including fuel) per MWh')
@xw.arg('capacity_factor', doc = 'This is the plant capacity factor')
@xw.arg('int_rate', doc = 'This is the applicaple discount rate')
def LCOE(overnight_capital_cost, fixed_om_cost, variable_om_cost, capacity_factor, int_rate ):
    return((overnight_capital_cost * CRF(int_rate,20.) + fixed_om_cost) /(8760 * capacity_factor) + variable_om_cost    )

@xw.func
def projectIRR(cost, annual_revenue, n):
    cashflows = [-abs(float(cost))]+ [float(annual_revenue)]* int(n)
    return (np.irr(cashflows ))

@xw.func
def projectDPBP(cost,  annual_revenue, n, rate):
    
    cashflows = [-abs(float(cost))]+ [float(annual_revenue)]* int(n)
    cf_df = pd.DataFrame(cashflows, columns = ['cashflows'])
    cf_df['discounted_cashflows'] = np.pv(rate= rate, nper = cf_df.index,pmt = 0,fv= -cf_df['cashflows'] )
    cf_df['cumulative_discounted_cashflows'] =  np.cumsum(cf_df['discounted_cashflows'])
    try:
        payback_period =cf_df[cf_df.cumulative_discounted_cashflows < 0].index.values.max()
        payback_period -= cf_df.cumulative_discounted_cashflows[payback_period ]/cf_df.discounted_cashflows[payback_period + 1]
        return (payback_period)
    except KeyError:
        return("DPBP > {} yrs".format(int(n)))

@xw.func
def projectDBCR(cost,  annual_revenue, n, rate):
    cashflows = [0] + [float(annual_revenue)]* int(n)
    benefit = np.npv(rate,cashflows)
    return (benefit / abs(float(cost)))

def plot_projects(ProjectID_in, Latitude_in,Longitude_in):
    df = pd.DataFrame({'ProjectID' :ProjectID_in,      'Latitude' : Latitude_in,      'Longitude' : Longitude_in})


    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'] .apply(Point)
    gdf =  geopandas.GeoDataFrame(df, geometry= 'Coordinates')

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    sht = xw.Book.caller().sheets.active
    fig = plt.figure()
    ax = world.plot(color = 'white', edgecolor = 'black',)
    ax_full = gdf.plot(ax=ax, color = 'red')
    fig.add(ax_full)
    sht.pictures.add(fig,name = 'Plot', update =     True)
    return ('Plotted a thing')


@xw.func
#@xw.func(async_mode='threading')
@xw.arg('xl_app', vba='Application')
def save_img(lat,lon, file_path, file_name,download_images, xl_app):
    
    
    #path
    
    full_path = os.path.abspath(os.path.join(file_path,file_name))
    sht = xw.Book.caller().sheets.active
    try:
        sht.pictures[full_path].delete()
    except KeyError:
#        return("doesn't exist")
        pass
    if(not download_images):
        return False
    

    zoom_factor = 18
    pixel_dim = 400
    key = 'AIzaSyAig-Wkbj1Rw-6ElabUWK_JlWvOpn57YJs'
    url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' + \
    str(lon) + '&zoom='+str(zoom_factor)+'&size='+str(pixel_dim)+'x'+str(pixel_dim+50)+'&maptype=satellite&key=' + key

    urllib.request.urlretrieve(url, full_path )
    
    
    row_offset = 7 
    row_number = int(xl_app.caller.row)-row_offset
    img_top = 190 + row_number *200
    img_left = int(xl_app.caller.left)
    sht.pictures.add(os.path.abspath(os.path.join(file_path,file_name) ), name = os.path.abspath(os.path.join(file_path,file_name) ), height = 200, left = img_left , top = img_top, width = 200)
    
    return(full_path ) #os.path.abspath(os.path.join(file_path,file_name) ))
#    b = cStringIO.StringIO(a)
#    image = ndimage.imread(b, mode='RGB')
    # when no image exists, api will return an image with the same color. 
    # and in the center of the image, it said'Sorry. We have no imagery here'.
    # we should drop these images if large area of the image has the same color.
 #   if np.array_equal(image[:,:10,:],image[:,10:20,:]):
 #       pass
 #   else:
 #       misc.imsave(file_path + file_name, image[50:450, :, :])

@xw.func
def delete_all_images():
    #SHOW_IMAGES = False
    sht = xw.Book.caller().sheets.active
    for picture in sht.pictures:
        picture.delete()
    return(True)

@xw.func
@xw.arg('xl_app', vba='Application')
def getPovertyLinePct(lat,lon, buffer_dist ,download_images, xl_app):
    geodata,filename = data.scripts.census.getDataAndImageForPoint(lat,lon,'pct_under_poverty_line', buffer_dist)

    #display image
    #remove previous image
    sht = xw.Book.caller().sheets.active

    try:
        sht.pictures[os.path.abspath(filename )].delete()
    except KeyError:
        pass

    if download_images:
        row_offset = 7 
        row_number = int(xl_app.caller.row)-row_offset
        img_top = 190 + row_number *200
        img_left = int(xl_app.caller.offset(0,2).left)
        sht.pictures.add(os.path.abspath(filename ), name = os.path.abspath(filename ), height = 200, left = img_left , top = img_top, width = 200)
    
    return (sum(geodata.pct_under_poverty_line *geodata.B01001_001E) / sum(geodata.B01001_001E))

@xw.func
@xw.arg('xl_app', vba='Application')
def getPopulation(lat,lon, buffer_dist ,download_images, xl_app):
    geodata,filename = data.scripts.census.getDataAndImageForPoint(lat,lon,'B01001_001E', buffer_dist)

    #display image
    #remove previous image
    sht = xw.Book.caller().sheets.active
    try:
        sht.pictures[os.path.abspath(filename )].delete()
    except KeyError:
        pass

    if download_images:
        row_offset = 7 
        row_number = int(xl_app.caller.row)-row_offset
        img_top = 190 + row_number *200
        img_left = int(xl_app.caller.offset(0,2).left)
        sht.pictures.add(os.path.abspath(filename ), name = os.path.abspath(filename ), height = 200, left = img_left , top = img_top, width = 200)
    
    return ( sum(geodata.B01001_001E)/(3.8621020e-7*sum(geodata.ALAND)))

print('end UDF definitions')

#@xw.func
#def MacaulayDuration(rate, price, ytm, maturity):
#    rates = np.full(maturity, rate)
#    cf = rates * 100
#    cf[maturity] += 100

#    mac_dur = np.sum([cfs[i]*（i+1）/np.power(1+rates[i],i+1) for i in range(len(cfs))])/price
#    mod_dur = mac_dur/(1+ytm/no_coupons)
#    return mac_dur, mod_dur

@xw.sub
def sample_sub():
    #wb = xw.Workbook.caller()
    xw.App.calculation= 'manual'
    xw.App.screen_updating = False
    count = 100 
    for i,image_path in enumerate(xw.Range("image_paths")):
        if image_path.value != None :
            xw.Range('urban_rural_classifications')[i].value = image_path.value

 

    xw.App.calculation= 'automatic'
    xw.App.screen_updating = True


@xw.sub
def classifyUrbanRural():
    #make excel stop updating for performance
    xw.App.calculation= 'manual'
    xw.App.screen_updating = False

    
    #define labels for 2 classes to predict
    LABELS = {0: 'Rural', 1: 'Urban'}
    
    #load mobilenet architecture
    base_model = keras.applications.mobilenetv2.MobileNetV2(weights=None, include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1280, activation='relu')(x)
    # and a logistic layer for 2 classes
    predictions_layer = Dense(2, activation='softmax')(x)
    #assemble model
    model = Model(inputs=base_model.input, outputs=predictions_layer)

    #load pre-trained weights
    model.load_weights('./weights/urban/20180924-mobilenet-best_weights.hdf5')
    model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])


    image_tensor = np.zeros(shape = (xw.Range("image_paths").count,224,224,3))
    max_i = 0
    for i,image_path in enumerate(xw.Range("image_paths")):
        if image_path.value != None :
            max_i = i
            img = image.load_img(image_path.value, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            image_tensor[i,:,:,:]=x

    pred = model.predict(image_tensor[0:(max_i+1),])

    for i in range(max_i+1):
        xw.Range('urban_rural_classifications')[i].value = "{:.2%} ".format(np.max(pred[i,])) +LABELS[np.argmax(pred[i,])]
     
    xw.App.calculation= 'automatic'
    xw.App.screen_updating = True

@xw.sub
def clearImages():
    sht = xw.Book.caller().sheets.active
    for picture in sht.pictures:
        picture.delete()
    