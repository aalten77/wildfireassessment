from wildfireassessment.ops import * #my package
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import morphology
from skimage.transform import resize
import pandas as pd
import geopandas as gpd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from sklearn.externals import joblib
from rasterstats import zonal_stats
import fiona
from joblib import Parallel, delayed
import multiprocessing
import time

def labelPburns():
    print("Reading filepaths...")
    filepath_mask = Path("./results")
    fps_masks = sorted(list(filepath_mask.glob("predict*.tif")))

    print("Going through all shapefiles...")
    pathShapeFiles = Path("./data/shapefiles/trainingData")
    for i in range(len(fps_masks)-2):
        nameID = fps_masks[i].name.split('_')[3].replace('.tif', '')
        fileShapes = sorted(list(pathShapeFiles.glob("*"+ nameID +"*.shp")))

        print("reading raster")
        raster_src_mask, mask = readOneImg(fps_masks[i])

        print("reading shapefiles for", nameID)
        for fileShape in fileShapes:
            gdf = gpd.read_file(fileShape)
            gdfcrs = gdf.crs

            prediction_segs = zonal_stats(gdf['geometry'], mask, affine=raster_src_mask.meta['transform'], stats='majority', nodata=-999, all_touched=True)
            
            df = pd.DataFrame(prediction_segs)
            gdf['pred_burn'] = df.to_numpy()
            gdf = gdf.astype({"pred_burn": int})
            gdf = gpd.GeoDataFrame(gdf, crs=gdfcrs)

            newFileName = fileShape.parent / Path(fileShape.name.split('.shp')[0] + "_pburn.shp")
            print("writing to path...", newFileName)
            gdf.to_file(newFileName)

        
        
"""
def writeRasters():
    #read in filepaths for data
    print("Reading filepaths...")
    filepath_post = Path("./data/Paradise/post")
    filepath_pre = Path("./data/Paradise/pre")

    #WorldView Post/Pre
    fps_wv_post = sorted(list(filepath_post.glob("2*_clip.tif")))
    fps_wv_pre = sorted(list(filepath_pre.glob("2*_clip.tif")))

    #WorldView Post/Pre
    fps_sent2_post = sorted(list((filepath_post / "clippedB08s").glob("B08_*.tif")))
    fps_sent2_pre = sorted(list((filepath_pre / "clippedB08s").glob("B08_*.tif")))

    print("Loading Model")
    #LOAD model
    rf_model = joblib.load(open("models/rf_grid_bin_precision.pkl", 'rb'))

    print("Start reading images")
    for i in range(len(fps_wv_post)):
        print("Reading RGB...")
        raster_src_post, rgb_post = readRGBImg(fps_wv_post[i])
        raster_src_pre, rgb_pre = readRGBImg(fps_wv_pre[i])

        print("Reading S2 B8...")
        raster_src_post_b08, b08_post = readOneImg(fps_sent2_post[i])
        raster_src_pre_b08, b08_pre = readOneImg(fps_sent2_pre[i])

        print("Resizing B8 images")
        b08_upscaled_post = resize(b08_post, raster_src_post.shape, anti_aliasing=True)
        b08_upscaled_post = b08_upscaled_post * 255
        b08_upscaled_post = b08_upscaled_post.astype(rasterio.uint8)

        b08_upscaled_pre = resize(b08_pre, raster_src_pre.shape, anti_aliasing=True)
        b08_upscaled_pre = b08_upscaled_pre * 255
        b08_upscaled_pre = b08_upscaled_pre.astype(rasterio.uint8)

        print("unravel rgb, b08")
        #unravel
        rgb_rav_post = {0 : rgb_post[:,:,0].ravel().astype(float),
                        1 : rgb_post[:,:,1].ravel().astype(float),
                        2 : rgb_post[:,:,2].ravel().astype(float)}
        rgb_rav_pre = {0 : rgb_pre[:,:,0].ravel().astype(float),
                       1 : rgb_pre[:,:,1].ravel().astype(float),
                       2 : rgb_pre[:,:,2].ravel().astype(float)}

        b08_rav_post = b08_upscaled_post.ravel().astype(float)
        b08_rav_pre = b08_upscaled_pre.ravel().astype(float)

        #release mem
        b08_upscaled_post = None
        b08_upscaled_pre = None
        b08_post = None
        b08_pre = None
        rgb_pre = None
        rgb_post = None

        print("starting predictions with model")
       
        def processInParallel(i):
            X_chunk = makeChunkX(rgb_rav_post[2][i:i+100], rgb_rav_post[1][i:i+100], rgb_rav_post[0][i:i+100], b08_rav_post[i:i+100],
                   rgb_rav_pre[2][i:i+100], rgb_rav_pre[1][i:i+100], rgb_rav_pre[0][i:i+100], b08_rav_pre[i:i+100])
            
            #impute by mean for missing values    
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X_chunk)
            X_chunk_imp = imp.transform(X_chunk)

            return rf_model.predict(X_chunk_imp)

        start_time = time.time() 
        num_cores = multiprocessing.cpu_count()
        pred_y = Parallel(n_jobs=num_cores, backend="multiprocessing")(delayed(processInParallel)(i) for i in range(0, len(b08_rav_post), 100))
        print("--- %s seconds ---" % (time.time() - start_time))

        print("Create mask")
        #create mask
        pred_y_rf = np.hstack(pred_y).reshape(raster_src_post.shape)

        #clean mask
        pred_y_rf_clean = morphology.remove_small_holes(pred_y_rf==1, 500) 
        pred_y_rf_clean = morphology.remove_small_objects(pred_y_rf_clean, 500)
            
        fileNameMask = "../results/predict_mask_rf_" + fps_wv_post[i].name.split('_')[0] + ".tif"
        print("Writing image mask to path:", fileNameMask)

        metadata = {
            'driver': 'GTiff', 
            'dtype': 'uint8',
            'width': raster_src_post.meta['width'],
            'height': raster_src_post.meta['height'],
            'count': 1, 
            'crs': raster_src_post.meta['crs'],
            'transform': raster_src_post.meta['transform']
        }
        
        with rasterio.open(fileNameMask, 'w', **metadata) as dst:
            dst.write(pred_y_rf_clean.astype(np.uint8), 1)
        


def computeSI(b1, b2):
    return (b1-b2)/(b1+b2)

def changedSI(SI_pre, SI_post):
    return SI_pre - SI_post

def makeChunkX(b, g, r, n, b_p, g_p, r_p, n_p):
    SI_gb = (computeSI(g, b), computeSI(g_p, b_p)) #(post, pre)
    SI_rb = (computeSI(r, b), computeSI(r_p, b_p))
    SI_rg = (computeSI(r, g), computeSI(r_p, g_p))
    SI_nb = (computeSI(n, b), computeSI(n_p, b_p))
    SI_ng = (computeSI(n, g), computeSI(n_p, g_p))
    SI_nr = (computeSI(n, r), computeSI(n_p, r_p))
    
    dSI_gb = changedSI(SI_gb[1], SI_gb[0])
    dSI_rb = changedSI(SI_rb[1], SI_rb[0])
    dSI_rg = changedSI(SI_rg[1], SI_rg[0])
    dSI_nb = changedSI(SI_nb[1], SI_nb[0])
    dSI_ng = changedSI(SI_ng[1], SI_ng[0])
    dSI_nr = changedSI(SI_nr[1], SI_nr[0])
    
    return np.dstack((b, b_p, g, g_p, r, r_p, n, n_p,
                      SI_gb[0], SI_rb[0], SI_rg[0], SI_nb[0], SI_ng[0], SI_nr[0],
                      SI_gb[1], SI_rb[1], SI_rg[1], SI_nb[1], SI_ng[1], SI_nr[1],
                      dSI_nb, dSI_rg, dSI_rb, dSI_gb, dSI_nr, dSI_ng))[0]
"""
