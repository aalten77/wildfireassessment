import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def numpy2KVfile(X, y, filename="paradiseXydata.txt"):
    n = X.shape[0]
    m = X.shape[1]

    col_nums = [i for i in range(1, m + 1)]

    f = open(Path("./data/Spark_data") / filename, "w+")

    for i in range(n):
        X_i = list(map(str, X[i]))
        y_i = str(y[i])
        X_i_str = ["{}:{}".format(j, x) for j, x in zip(col_nums, X_i)]
        X_i_str.insert(0, y_i)
        Xy_str = "\t".join(X_i_str)

        f.write(Xy_str+"\n")

    f.close()

def convertDF2Numpy(df):

    categories = list(df['land_class'].values)
    burn_cats = list(df['burn_class'].values)

    #combine the labels
    both_cats = ["{}_{}".format(a, b) for a, b in zip(categories, burn_cats)]

    # build X and y
    X = df.drop(columns=['land_class', 'burn_class']).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(both_cats)
    print(le.classes_)

    return X, y

def readPaths(filepaths):
    gdfs = []
    for filepath in filepaths:
        gdfs.append(gpd.read_file(filepath))

    #combine and remove null values
    gdf_all = pd.concat(gdfs)
    gdf_all.dropna(inplace=True)
    gdf_all = gdf_all.astype({'burn_class': 'uint8'})

    #drop columns not necessary for train/test set
    df = gdf_all.drop(columns=['seg_index', 'area_m', 'geometry'])

    df = df.drop(columns=['SI_bg_post', 'SI_br_post', 'SI_bn_post', 'SI_bg_pre', 'SI_br_pre', 'SI_bn_pre',
                             'SI_gr_post', 'SI_gn_post', 'SI_gr_pre', 'SI_gn_pre', 'SI_rn_post', 'SI_rn_pre',
                             'dSI_bg', 'dSI_gr', 'dSI_gn', 'dSI_br', 'dSI_rn', 'dSI_bn'])

    # there was a row with water labelled as burnt, have to fix it!
    df.at[2044, 'burn_class'] = 0

    return df


def main():
    filepaths = [Path("./data/shapefiles/segments_2011023_0.shp"), Path("./data/shapefiles/segments_2011201_0.shp"),
                 Path("./data/shapefiles/segments_2011203_0.shp"), Path("./data/shapefiles/segments_2010133_0.shp"),
                 Path("./data/shapefiles/segments_2011200_0.shp")]

    # 1 read files as dataframe
    df = readPaths(filepaths)
    print(df.head())

    # 2 convert dataframe into X and y numpy matrices
    X, y = convertDF2Numpy(df)

    # 3 split by stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True, stratify=y)

    # 4 write X y to libsvm file format for Spark
    numpy2KVfile(X_train, y_train, filename="paradiseXy_train.txt")
    numpy2KVfile(X_test, y_test, filename="paradiseXy_test.txt")

if __name__ == "__main__":
    main()