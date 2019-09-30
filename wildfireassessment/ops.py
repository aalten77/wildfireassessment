"""
    Author: Ai-Linh Alten
    Date created: 9/23/2019
    Date last modified: 9/24/2019
    Python Version: 3.6.5

    This is a series of operations for Wildfire Damage Assessment project.
    Project for Master's in Computer Science at SJSU for the CS 298 class.
"""
import rasterio
from rasterio.transform import xy
from rasterio.plot import show, show_hist
from rasterio.mask import mask
import matplotlib.pyplot as plt
from pathlib import Path
import os
import earthpy as et
import earthpy.plot as ep
import numpy as np
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
from pyproj import Proj, transform

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

    out_img = np.dstack((out_img[0], out_img[1], out_img[2]))

    return out_img, out_transform


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

# def imshowBBOX(im, bbox=[], title=""):
#     """ Shows image to a zoomed spatial extent.
#     Parameters:
#         im : ndarray
#             The image read and loaded by rasterio.
#         bbox : list
#             [minx, miny, maxx, maxy] in coordinates of the image CRS.
#         title : String
#             title for the figure
#     """
#
#     bbox_gdf = gpd.GeoDataFrame()
#     bbox_gdf.loc[0, 'geometry'] = box(*bbox)
#
#     fig, ax = plt.subplots(figsize=(8,3))
#     ep.plot_rgb(im, extent=bbox, title=title, ax=ax)
#     ax.set_xlim(bbox[0], bbox[2])
#     ax.set_ylim(bbox[1], bbox[3])
#     plt.show()
