"""
    Author: Ai-Linh Alten
    Date created: 9/23/2019
    Date last modified: 10/2/2019
    Python Version: 3.6.5

    This is a series of operations for Wildfire Damage Assessment project.
    Project for Master's in Computer Science at SJSU for the CS 298 class.
"""
import rasterio
from rasterio.transform import xy
from rasterio.mask import mask
import numpy as np
from fiona.crs import from_epsg
from pyproj import transform
from functools import partial
import pyproj
from shapely.ops import transform
from rasterstats import zonal_stats
import itertools
from skimage.segmentation import slic, felzenszwalb
from skimage.measure import regionprops
from shapely.wkt import loads
from shapely.geometry import shape, box
from rasterio import features
import geopandas as gpd
import pandas as pd

def writeDatasets(fps_post, fps_pre, fp_s2_post, fp_s2_pre):
    print("Writing to data...")
    for i, fp in enumerate(fps_post):
        print(fp)
        filename = str(fp).split('\\')[3].split('_')[0]

        print("reading clipped rgb images for {}".format(filename))
        raster_src_post, rgb_post = readRGBImg(fp)
        raster_src_pre, rgb_pre = readRGBImg(fps_pre[i])

        # bounds of the WorldView images
        bbox = box(raster_src_post.bounds.left, raster_src_post.bounds.bottom, raster_src_post.bounds.right,
                        raster_src_post.bounds.top)

        print("reading and clipping sent2 images")
        raster_src_post_b08, b08_post = readOneImg(fp_s2_post)
        out_img_post_b08, out_img_transform_post_b08, out_post_meta= clipImg(bbox, raster_src_post_b08, proj=4326)
        b08_post = None

        raster_src_pre_b08, b08_pre = readOneImg(fp_s2_pre)
        out_img_pre_b08, out_img_transform_pre_b08, out_pre_meta = clipImg(bbox, raster_src_pre_b08, proj=4326)
        b08_pre = None

        print("chunk image")
        # 2- Segment, Vectorize, save to file
        chunkindices = chunkImageIndices(rgb_post)
        for i, chunkindextup in enumerate(chunkindices):
            print("starting segmentation for chunk {}".format(i))
            img_chunk_post = rgb_post[chunkindextup[0]:chunkindextup[1], chunkindextup[2]:chunkindextup[3], :]
            img_chunk_pre = rgb_pre[chunkindextup[0]:chunkindextup[1], chunkindextup[2]:chunkindextup[3], :]
            segments = segmentToLabels(img_chunk_pre, n_segments=5000, compactness=10)

            print("vectorizing...")
            # Vectorize
            gdf = vectorizeSegments(segments, img_chunk_post, img_chunk_pre, raster_src_post.transform, out_img_post_b08, out_img_pre_b08, out_img_transform_post_b08)
            img_chunk_post = None
            img_chunk_pre = None
            segments = None

            print("adding SIs to gdf")
            #add SIs
            gdf = addSIs2DF(gdf)

            gdf_filename = "./data/segments_{}_{}.geojson".format(filename, i)
            print("writing to destination: {}".format(gdf_filename))
            #write to file
            gdf.to_file(gdf_filename, driver="GeoJSON")

def writeClippedHelper(img, img_transform, img_meta, filename):
    trueColor = rasterio.open(filename, 'w', **img_meta)
    trueColor.write(img[2], 3) #blue
    trueColor.write(img[1], 2) #green
    trueColor.write(img[0], 1) #red
    trueColor.close()


def writeClipped(fps, fps_pre):

    for i, fp in enumerate(fps):
        filepath_post = fp.parent
        filename_post = fp.name.replace("clipped", "clip")
        filepath_pre = fps_pre[i].parent
        filename_pre = fps_pre[i].name.replace("clipped", "clip")

        raster_src_post, rgb_post = readRGBImg(fp)
        indices_post = indicesDG(rgb_post) #calculate indices from nonzero
        rgb_post = None
        bbox_post = indicesToBBOX(indices_post, raster_src_post) # make bbox
        out_img_post, out_img_transform, out_meta = clipImg(bbox_post, raster_src_post, proj=4326)
        writeClippedHelper(out_img_post, out_img_transform, out_meta, filepath_post / filename_post)
        out_img_post, out_img_transform, out_meta = None, None, None

        raster_src_pre, rgb_pre = readRGBImg(fps_pre[i])
        rgb_pre = None
        out_img_pre, out_img_transform, out_meta = clipImg(bbox_post, raster_src_pre, proj=4326)
        writeClippedHelper(out_img_pre, out_img_transform, out_meta, filepath_pre / filename_pre)
        out_img_post, out_img_transform, out_meta = None, None, None


def addChangedSIToDFhelper(gdf, SI_combos):
    # now, add dSIs to dataframe
    for i, tup in enumerate(SI_combos):
        SI_post = tup[0]
        SI_pre = tup[1]
        col_name = "dSI_" + SI_post.split('_')[1]
        gdf[col_name] = changedSI(gdf[SI_pre], gdf[SI_post])
    return gdf

def addSIToDFhelper(gdf, perm, tag="_post"):
    SI_keys = []
    for i, tup in enumerate(perm):
        b1 = tup[0]
        b2 = tup[1]
        col_name = "SI_" + b1[0] + b2[0] + tag
        SI_keys.append(col_name)
        gdf[col_name] = computeSI(gdf[b1], gdf[b2])
    return gdf, SI_keys

def addSIs2DF(gdf):
    # convert the keys to list
    post_keys = ['blue_value', 'green_value', 'red_value', 'nir_value']
    pre_keys = ['blue_value_pre', 'green_value_pre', 'red_value_pre', 'nir_value_pre']

    perm = itertools.permutations(post_keys, 2)
    perm_pre = itertools.permutations(pre_keys, 2)

    gdf, SI_keys_post = addSIToDFhelper(gdf, perm, tag="_post")
    gdf, SI_keys_pre = addSIToDFhelper(gdf, perm_pre, tag="_pre")

    SI_keys = SI_keys_pre + SI_keys_post
    band_combos = list(set([key.split('_')[1] for key in SI_keys_post]))
    SI_combos = [tuple([s for s in SI_keys if bcombostr in s]) for bcombostr in band_combos]

    gdf = addChangedSIToDFhelper(gdf, SI_combos)

    gdf['land_class'] = np.nan
    gdf['burn_class'] = np.nan

    return gdf

def computeSI(b1, b2):
    return (b1-b2)/(b1+b2)

def changedSI(SI_pre, SI_post):
    return SI_pre - SI_post

def projectPolygons(gdf, label, inproj='epsg:4326', outproj='epsg:32610'):
    project = partial(
        pyproj.transform,
        pyproj.Proj(init=inproj),  # source coordinate system
        pyproj.Proj(init=outproj))  # destination coordinate system

    g1 = gdf.loc[gdf['seg_index'] == label, 'geometry'].iloc[0]
    g2 = transform(project, g1)  # apply projection

    return g2

def vectorizeSegments(segment_labels, img_chunk_post, img_chunk_pre, transform, b8_post, b8_pre, b8_transform):
    shapes_list = [{'seg_index': int(v), 'geometry': loads(shape(g).wkt)} for g, v in
                   features.shapes(segment_labels.astype(np.uint16), mask=None, transform=transform)]
    gdf = gpd.GeoDataFrame(shapes_list)
    gdf.crs = {'init': 'EPSG:4326'}
    gdf = gdf[gdf['seg_index'] != 0] #remove the 0 label

    ## use regionprops onn segments to extract properties for dataframe
    # post
    regions_red = regionprops(segment_labels, img_chunk_post[:,:,0])
    regions_blue = regionprops(segment_labels, img_chunk_post[:,:,1])
    regions_green = regionprops(segment_labels, img_chunk_post[:,:,2])

    # pre
    regions_red_pre = regionprops(segment_labels, img_chunk_pre[:,:,0])
    regions_blue_pre = regionprops(segment_labels, img_chunk_pre[:,:,1])
    regions_green_pre = regionprops(segment_labels, img_chunk_pre[:,:,2])

    region_spectrals = []
    for i in range(len(regions_red)):
        seg_label = regions_red[i].label
        g2 = projectPolygons(gdf, seg_label)

        nir_zone_post = zonal_stats(g2, b8_post[0], affine=b8_transform, stats='mean',
                                    nodata=-999)
        nir_zone_pre = zonal_stats(g2, b8_pre[0], affine=b8_transform, stats='mean', nodata=-999)

        dict_seg = {'seg_index': regions_red[i].label,
                    'red_value': regions_red[i].mean_intensity,
                    'blue_value': regions_blue[i].mean_intensity,
                    'green_value': regions_green[i].mean_intensity,
                    'nir_value': nir_zone_post[0]['mean'],
                    'red_value_pre': regions_red_pre[i].mean_intensity,
                    'blue_value_pre': regions_blue_pre[i].mean_intensity,
                    'green_value_pre': regions_green_pre[i].mean_intensity,
                    'nir_value_pre': nir_zone_pre[0]['mean'],
                    'area_m': regions_red[i].area * 0.31}  # area in meters
        region_spectrals.append(dict_seg)

    df = pd.DataFrame(region_spectrals)
    gdf2 = pd.merge(gdf, df, on='seg_index')

    return gdf2

def segmentToLabels(img_chunk, n_segments=5000, compactness=10):
    """ Convert img to segments and return image chunk with it."""
    segments_slic = slic(img_chunk, n_segments=n_segments, compactness=compactness)
    print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

    return segments_slic


def chunkImageIndices(img):
    """return list of tuples with indices for image chunk"""
    chunksize = (img.shape[0]//2, img.shape[0]//2)

    #list of tuples (begx, endx, begy, endy)
    indices = [(0, chunksize[0], 0, chunksize[1]),
               (chunksize[0], img.shape[0], 0, chunksize[1]),
               (0, chunksize[0], chunksize[1], img.shape[1]),
               (chunksize[0], img.shape[0], chunksize[1], img.shape[1])
               ]
    return indices

# https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def clipImg(bbox, src_img, proj=4326):
    """ Clip image to bbox and output new numpy img matrix. Return transform as well."""

    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(proj))
    geo = geo.to_crs(crs=src_img.crs.data)
    coords = getFeatures(geo)
    out_img, out_transform = mask(src_img, shapes=coords, crop=True)

    out_meta = src_img.meta.copy()
    out_meta.update({'driver': 'GTiff',
                         'width': out_img.shape[2],
                         'height': out_img.shape[1],
                         #  'crs': pycrs.parser.from_epsg_code(epsg_code).to_proj4(),
                         'transform': out_transform})

    return out_img, out_transform, out_meta


def indicesToBBOX(indices, src_img):
    """ Takes indices from DG image and returns the BBOX in coordinates of transform.
    :param indices: list
    :param src_img: raster data
    :return: Shapely POLYGON
    """
    min_coords = xy(src_img.transform, indices[1], indices[2])
    max_coords = xy(src_img.transform, indices[0], indices[3])

    return box(min_coords[0], min_coords[1], max_coords[0], max_coords[1])

def indicesDG(img):
    """
    Read in WorldView-3 image and get indices of non-zeros.
    :param img: Ndarray
        Numpy 3D matrix RGB.
    :return: list
        Return list of indices: begining x, end x, begining y, end y
    """
    indices = np.nonzero(img)
    beg_x = indices[0][0]
    end_x = indices[0][indices[0].shape[0]-1]
    beg_y = indices[1][0]
    end_y = indices[1][indices[1].shape[0]-1]
    indices = None

    return [beg_x, end_x, beg_y, end_y]

def readOneImg(datapath, driver="GTiff"):
    """
    Reads image path and returns 2-D numpy matrix of single band image.
    :param datapath: Path/String
        Image path that is opened by rasterio.
    :param driver: String
        GDAL driver for opening the image. Default: 'GTiff'.
    :return: rasterio data, Ndarray
        2-D numpy matrix of image.
    """
    src_img = rasterio.open(datapath, driver=driver)

    band = src_img.read(1)

    return src_img, band

def readRGBImg(datapath, driver="GTiff"):
    """
    Reads image path and returns 3-D numpy matrix of RGB image.
    :param datapath: Path/String
        Image path that is opened by rasterio.
    :param driver: String
        GDAL driver for opening the image. Default: 'GTiff'.
    :return: rasterio data, Ndarray
        Tuple with opened rasterio data and 3-D numpy matrix of image.
    """
    src_img = rasterio.open(datapath, driver=driver)

    r = src_img.read(1)
    g = src_img.read(2)
    b = src_img.read(3)

    return src_img, np.dstack((r, g, b))
