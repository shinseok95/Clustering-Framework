# 알고리즘에 필요한 library

import base64
import datetime
import time
import io
import os
import keras
import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as plt
import pandas as pd
import glob
import math
import cv2
import pickle
import umap.umap_ as umap
import warnings

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cityblock
from math import cos, pi

from matplotlib import cm
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestCentroid

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from tslearn.clustering import KShape

from math import sqrt
from scipy import stats
from scipy.cluster import vq
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

from tensorflow.python.keras.backend import eager_learning_phase_scope
from keras.engine.topology import Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Reshape, LeakyReLU, \
    Conv1D, UpSampling1D, MaxPooling1D, Conv1DTranspose, ELU, Dropout, MaxPooling2D, UpSampling2D, concatenate, \
    Activation
from keras.layers import Input
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
from keras import optimizers

# Dash에 필요한 library

from dash.dependencies import Input, Output, State
import dash_table
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc

from plotly.subplots import make_subplots

import plotly.graph_objects as go
import pickle
import copy
import pathlib
import urllib.request
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html

# Multi-dropdown options
from controls import COUNTIES, WELL_STATUSES, WELL_TYPES, WELL_COLORS

# Download
from flask import Flask, send_file
from dash_extensions import Download

# 전역 변수


np.set_printoptions(threshold=np.inf)  # ...없이 출력하기

PATH = None
df = None

dataset = None
embedding_data = None
predict = None
dataset_pure = None
preprocessing_csv = None
process_label = None
csv_file_name = None
split_name = None

autoencoder_hist = None
dataset_pure_list = None
cutting_dataset = None
cutting_dataset_pure = None
embedding_csv = None

centroid_idx = None
centroid_value = None

Kmean = KMeans(n_clusters=5)

# Function

"""

Data Proprocessing(0) : CSV로부터 DATA 입력받기
"""

# 데이터 입력 함수

"""
path : csv 파일의 경로
column : 데이터를 나타내는 칼럼명
"""


def align_timeseries_dataset(path, value_col, process_col=None):
    if path == None or value_col == None:
        return

    input_csv = pd.read_csv(path, engine='python', encoding='euc-kr')
    input_csv = input_csv.astype({value_col: 'float32'})

    # 결측치 제거
    input_csv = input_csv.dropna(subset=[value_col])

    # z_score
    zscore_dataset = z_score_normalize(input_csv[value_col])
    input_csv['z_score'] = zscore_dataset

    # min-max
    minmax_dataset = min_max_normalize(input_csv[value_col])
    input_csv['min_max'] = minmax_dataset

    # 전처리 csv 저장용 DataFrame
    preprocessing_csv = pd.DataFrame()

    # Process가 존재하지 않는 경우
    if process_col == None:
        process_set = None
        dataset_preprocessing = input_csv['min_max']
        dataset = input_csv[value_col]

    # Process가 존재하는 경우 process 별로 데이터 분리
    else:
        dataset_preprocessing = []
        dataset = []

        process_list = input_csv[process_col]
        process_set = list(set(process_list))

        for process in process_set:
            data = input_csv[(input_csv[process_col] == process)]
            data_preprocessing = data['min_max']
            data_pure = data[value_col]

            dataset_preprocessing.append(data_preprocessing)
            dataset.append(data_pure)

        preprocessing_csv['Process'] = process_list

    preprocessing_csv['Value'] = input_csv[value_col]
    preprocessing_csv['z_score'] = zscore_dataset
    preprocessing_csv['min_max'] = minmax_dataset

    return np.array(dataset_preprocessing), preprocessing_csv, process_set, np.array(dataset)


"""Data preprocessing function(1) - 시계열 데이터 길이 조절(truncation, padding, sliding window, DTW 등)"""


# 데이터 자르기 함수

def data_truncation(dataset):
    return_dataset = []
    max_size = 0
    min_size = 999999

    for i in range(len(dataset)):
        if dataset[i].size < min_size:
            min_size = dataset[i].size

    for i in range(len(dataset)):
        if dataset[i].size > min_size:
            return_dataset.append(dataset[i][:min_size])
        else:
            return_dataset.append(dataset[i])

    return np.array(return_dataset)


# 데이터 패딩 함수

def data_padding(dataset):
    return_dataset = []
    max_size = 0

    for i in range(len(dataset)):
        if dataset[i].size > max_size:
            max_size = dataset[i].size

    for i in range(len(dataset)):
        if dataset[i].size < max_size:
            return_dataset.append(np.pad(dataset[i], (0, max_size - dataset[i].size), 'constant', constant_values=0))
        else:
            return_dataset.append(dataset[i])

    return np.array(return_dataset)


# Sliding window 함수

def sliding_window(dataset, window_size=10, shift_size=1):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(window_size, shift=shift_size, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    return_dataset = list()

    for window in dataset:
        return_dataset.append(window.numpy())
    return_dataset = np.array(return_dataset)

    return return_dataset


# DTW 유사도를 통한 시계열 데이터 확장 함수 (작은 길이를 큰 길이로 맞춤)

def data_dtw(dataset):
    return_dataset = []
    max_size = 0
    max_index = 0

    for i in range(len(dataset)):
        if dataset[i].size > max_size:
            max_size = dataset[i].size
            max_index = i

    long_ts_data = dataset[max_index]

    for i in range(len(dataset)):
        if dataset[i].size < max_size:
            return_dataset.append(DTW_resize_algorithm(long_ts_data, dataset[i])[0])
        else:
            return_dataset.append(dataset[i])

    return np.array(return_dataset)


def DTW_resize_algorithm(long_ts_data, short_ts_data):
    if len(long_ts_data) == len(short_ts_data):
        return np.array(short_ts_data), np.array([0] * len(short_ts_data))

    step = 0
    similarity_degree_path = [0] * len(long_ts_data)
    long_ts_data = np.array(long_ts_data)
    short_ts_data = np.array(short_ts_data)

    path_coordinates = fastdtw(short_ts_data, long_ts_data)[1]

    for i in range(len(similarity_degree_path)):

        similarity_degree_path[i] = (long_ts_data[path_coordinates[step][1]] - short_ts_data[path_coordinates[step][0]])

        for j in range(step + 1, len(path_coordinates)):

            if path_coordinates[step][1] == path_coordinates[j][1]:
                similarity_degree_path[i] = similarity_degree_path[i] + (
                        long_ts_data[path_coordinates[j][1]] - short_ts_data[path_coordinates[j][0]])
                step = j
                continue

            else:
                step += 1
                break

    resize_ts_data = (long_ts_data - similarity_degree_path)

    return np.array(resize_ts_data), np.array(similarity_degree_path)


"""Data preprocessing function(2) - 데이터 정규화 or 일반화"""


# Min-Max Normalization: 모든 feature들의 스케일이 동일하지만, 이상치(outlier)를 잘 처리하지 못한다.

def min_max_normalize(lst):
    normalized = []

    min_value = min(lst)
    max_value = max(lst)

    for value in lst:
        normalized_num = (value - min_value) / (max_value - min_value)
        normalized.append(normalized_num)

    return np.array(normalized)


# Z-Score Normalization : 이상치(outlier)를 잘 처리하지만, 정확히 동일한 척도로 정규화 된 데이터를 생성하지는 않는다.

def z_score_normalize(lst):
    normalized = []

    mean_value = np.mean(lst)
    std_value = np.std(lst)

    for value in lst:
        normalized_num = (value - mean_value) / std_value
        normalized.append(normalized_num)
    return np.array(normalized)


"""Data preprocessing function(3) - 잠재 벡터 추출(UMAP, 이미지화 등)"""

"""
RP 알고리즘

serialize_vector : 시계열 데이터 value vector
"""


def RP_algorithm(serialize_vector):
    N = serialize_vector.size
    S = np.repeat(serialize_vector[None, :], N, axis=0)
    Z = np.abs(S - S.T)
    Z /= Z.max()
    Z *= 255
    Z = Z.astype('uint8')
    Z = np.array(Z)
    return Z


"""

GAF 알고리즘 

"""


def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def cos_sum(a, b):
    """To work with tabulate."""
    return (math.cos(a + b))


class GAF_algorithm:

    def __init__(self):
        pass

    def __call__(self, serie):
        """Compute the Gramian Angular Field of an image"""
        # Min-Max scaling
        min_ = np.amin(serie)
        max_ = np.amax(serie)
        scaled_serie = (2 * serie - max_ - min_) / (max_ - min_)

        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        # Polar encoding
        phi = np.arccos(scaled_serie)
        # Note! The computation of r is not necessary
        r = np.linspace(0, 1, len(scaled_serie))

        # GAF Computation (every term of the matrix)
        gaf = tabulate(phi, phi, cos_sum)

        gaf = (1 + gaf) * 255 / 2.0

        return gaf


"""
image_vector : 변경하고자 하는 이미지 리스트
img_size : 변경하고자 하는 이미지 크기
"""


def resize_img(image_vector_list, img_size):
    image_list = []

    for i in range(len(image_vector_list)):

        if len(image_vector_list[i]) > img_size:
            img = cv2.resize(image_vector_list[i], (img_size, img_size), interpolation=cv2.INTER_AREA)

        else:
            img = cv2.resize(image_vector_list[i], (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        img = img.astype('uint8')
        image_list.append(img)

    image_list = np.array(image_list)

    return image_list


# Auto_Encoder 함수

"""

dataset : (n,) shape의 데이터
LEARNING_LATE : 러닝레이트
BATCH_SIZE : 배치사이즈
EPOCHS : 에폭
TEST_SIZE : traing에서 뽑아낼 test 사이즈(default = 100)
IMG_SIZE : 이미지사이즈(default = 64)

"""


def embedding_AE(dataset, LEARNING_LATE, BATCH_SIZE, EPOCHS, IMAGING_FLAG='1', IMG_SIZE_FLAG='1', TEST_SIZE=500):
    global autoencoder_hist

    np.random.seed(1)
    tf.random.set_seed(1)

    latent_dim = 2
    dataset_img = []
    autoencoder_hist = []

    if IMAGING_FLAG == '1':
        for i in range(len(dataset)):
            dataset_img.append(RP_algorithm(dataset[i]))
    else:
        gaf = GAF_algorithm()
        for i in range(len(dataset)):
            dataset_img.append(gaf(dataset[i]))

    if IMG_SIZE_FLAG == '1':
        IMG_SIZE = 64
    elif IMG_SIZE_FLAG == '2':
        IMG_SIZE = 256
    else:
        IMG_SIZE = 512

    dataset_class_img = resize_img(dataset_img, IMG_SIZE)
    dataset_class_img = np.array(dataset_class_img)

    dataset_image = dataset_class_img.reshape(len(dataset), IMG_SIZE, IMG_SIZE)

    # Image -> train / test로 나누기

    train = dataset_image
    np.random.shuffle(train)

    train = train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    test = train[len(train) - TEST_SIZE:]
    train = train[:len(train) - TEST_SIZE]

    # 데이터 정규화

    train = train / dataset_image.max()
    test = test / dataset_image.max()

    # 체크포인트 설정

    ae_checkpoint_path = 'AE.ckpt'
    ae_checkpoint_dir = os.path.dirname(ae_checkpoint_path)

    ae_callback_early = keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=50,
        verbose=0,
        mode='auto'
    )

    ae_callback_best = keras.callbacks.ModelCheckpoint(
        filepath=ae_checkpoint_path,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq=1
    )

    count = factor(IMG_SIZE)
    # AE Layer 설정

    encoder_input = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input')

    x = Conv2D(16, 3, strides=2, padding='same', activation='relu')(encoder_input)
    x = BatchNormalization()(x)

    lenl = 32

    for i in range(count - 4):
        x = Conv2D(lenl, 3, strides=2, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        lenl *= 2

    x = Flatten()(x)
    units = x.shape[1]

    # 2D 좌표로 표기하기 위하여 2를 출력값으로 지정
    embed = Dense(latent_dim, name='embedded')(x)

    x = Dense(units)(embed)
    x = Reshape((8, 8, IMG_SIZE))(x)

    lenl = IMG_SIZE / ((count % 2) + 1)

    for i in range(count - 4):
        x = Conv2DTranspose(lenl, 3, strides=2, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        lenl /= 2

    decoder_outputs = Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='sigmoid', name='output')(x)

    # 오토인코더 실행

    autoencoder = Model(encoder_input, decoder_outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_LATE), loss=tf.keras.losses.MeanSquaredError())

    # hist 저장
    for i in range(1, EPOCHS + 1):
        start = time.perf_counter()
        hist = autoencoder.fit(train, train, batch_size=BATCH_SIZE, epochs=1, validation_data=(test, test),
                               shuffle=True, callbacks=[ae_callback_early, ae_callback_best])
        end = time.perf_counter()

        execution_time = round(end - start, 3)

        result = str(i) + " / " + str(EPOCHS) + " [ " + str(execution_time) + " ms ]" + " - loss : " + str(
            np.round(hist.history["loss"], 4)) + " / val_loss : " + str(np.round(hist.history["val_loss"], 4))

        print(result)
        autoencoder_hist.append(result)

    autoencoder.summary()
    # hist = autoencoder.fit(train, train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test, test),
    #                       shuffle=True, callbacks=[ae_callback_early, ae_callback_best])

    # 이미지 그리기

    # decoded_images = autoencoder.predict(test)

    # draw_image_data(test,"Original Image",IMG_SIZE)
    # draw_image_data(decoded_images,"Reproduction Image",IMG_SIZE)

    autoencoder_hist.append("DONE!")

    # 인코딩된 잠재 벡터
    get_embedded = K.function([autoencoder.get_layer('input').input],
                              [autoencoder.get_layer('embedded').output])

    dataset_dimension = np.vstack(get_embedded([dataset_image]))
    dataset_dimension_data = np.vstack([dataset_dimension])
    dataset_dimension_data = dataset_dimension_data.reshape(-1, latent_dim)

    # history 정보 출력
    print(" history 정보 출력 ")
    for i in range(0, EPOCHS):
        print(i + 1, ' / ', EPOCHS, autoencoder_hist[i])

    return dataset_dimension_data


def factor(n):
    count = 0
    while n % 2 == 0:
        count += 1
        n = int(n / 2)
    return count


# UMAP 함수

"""
n_components : 축소하고자 하는 차원수
_n_neighbors : 작을수록 locality를 잘 나타내고, 커질수록 global structure를 잘 나타냄
_min_dist : 얼마나 점들을 조밀하게 묶을 것인지 (낮을 수록 조밀해짐)
"""


def embedding_UMAP(dataset, _n_neighbors=50, _min_dist=0.1):
    reducer = umap.UMAP(n_components=2, n_neighbors=_n_neighbors, init='random', random_state=0, min_dist=_min_dist)
    embedding = reducer.fit_transform(dataset)
    return embedding


# PCA 함수
"""
n_component : 주성분 분석 개수
반환값 : PCA 실행 결과 
"""


def embedding_PCA(dataset, n_component=2):
    pca = PCA(n_components=n_component)
    pca.fit(dataset)
    dataset_pca = pca.transform(dataset)
    # print(pca.explained_variance_ratio_)
    return dataset_pca


"""Clustering"""

# K-means 함수

"""
dimension_data=잠재벡터
MAX_CLUSTER_SIZE=최대 군집 개수
"""


def clustering_KMEANS(dimension_data, n_cluster, MAX_CLUSTER_SIZE=10):
    # class_dimension_data=잠재벡터,num_cluster=최대 군집 개수

    # n_cluster_list = cal_Silhouette(dimension_data, MAX_CLUSTER_SIZE, 5)
    # best_cluster = n_cluster_list[0]

    dimension_data = dimension_data.reshape(-1, 2)

    Kmean = KMeans(n_clusters=n_cluster)
    Kmean.fit(dimension_data)

    predict = Kmean.predict(dimension_data)

    # for center in n_cluster_list:
    #    draw_cluster_and_center(dimension_data, center)

    return predict


# K-shape 함수

"""
timeseries_data=
n_cluster=군집 수
"""


def clustering_KSHAPE(timeseries_data, n_cluster=2):
    ks = KShape(n_clusters=n_cluster)
    cluster_found_kshape = ks.fit_predict(timeseries_data)

    return cluster_found_kshape


# DBSCAN 함수
"""
eps : 기준점으로부터의 거리
min_samples : 반경 내의 점의 개수
반환값 : dbscan 군집 결과
"""


def clsutering_DBSCAN(dataset, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    predict = dbscan.fit_predict(dataset)

    return predict


"""기타 함수"""


# 실루엣 다이어그램 그리는 함수

def plotSilhouette(X, y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)

        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhoutte_avg = np.mean(silhouette_vals)
    plt.axvline(silhoutte_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('K')
    plt.xlabel('Silhouette value')
    plt.show()


#  실루엣 계수 계산 함수
#  max_cluster_num : 최대 클러스터 개수(k)
#  cluster_num : 추출할 실루엣 계수 높은 클러스터 개수

def cal_Silhouette(data, max_cluster_num, cluster_num):
    max = []

    for i in range(2, max_cluster_num + 1):
        km = KMeans(n_clusters=i, random_state=10)
        km_labels = km.fit_predict(data)
        max.append([i, silhouette_score(data, km_labels)])

        max.sort(key=lambda x: x[1], reverse=True)
    print("max : ", max)
    # 실루엣 계수 높은 상위 (clust_num)개만 추출해서 클러스터 개수 저장
    n_cluster_list = []
    n_cluster_value = []

    for i in max[0:cluster_num]:
        n_cluster_list.append(i[0])
        n_cluster_value.append(round(i[1]*100,2))

    return n_cluster_list,n_cluster_value


def draw_inertia_kshape(dimension_data, MAX_CLUSTER_SIZE=10):
    distortions_kshape = []

    for i in range(2, MAX_CLUSTER_SIZE + 1):
        kshape = KShape(n_clusters=i, max_iter=100)
        cluster_found = kshape.fit_predict(dimension_data)
        distortions_kshape.append(kshape.inertia_)

    plt.plot(range(2, MAX_CLUSTER_SIZE + 1), distortions_kshape, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


"""
cluster_list 의 k 값으로 실루엣 다이어그램 작성 함수

"""


def draw_Silhouette_Diagram(data, n_cluster_list):
    for i in range(len(n_cluster_list)):
        print("Cluster 개수 : ", n_cluster_list[i])

        km = KMeans(n_clusters=n_cluster_list[i], random_state=10)
        km_labels = km.fit_predict(data)
        plotSilhouette(data, km_labels)


"""
center : cluster center 수
data : cluster data
"""


def draw_cluster_and_center(data, center):
    Kmean = KMeans(n_clusters=center)
    Kmean.fit(data)

    plt.scatter(data[:, 0], data[:, 1], s=0.05, c=Kmean.labels_.astype(float))

    print("Center 개수 : ", center)
    for i in range(center):
        plt.scatter(Kmean.cluster_centers_[i, 0], Kmean.cluster_centers_[i, 1], s=50, c='red', marker='s')

    plt.show()


# 시계열 데이터 그리는 함수

"""
dimenstion_data : 2차원의 데이터
predict : 라벨 번호 list (정답)
label : 라벨 이름 list

"""


def draw_vector_data(dimenstion_data, predict=None, label=None, center=None):
    plt.figure(figsize=(15, 15))
    plt.rc('legend', fontsize='20')

    if predict is not None:
        if label is None:
            if center is None:
                plt.scatter(dimenstion_data[:, 0], dimenstion_data[:, 1], s=30, c=predict)
            else:
                plt.scatter(dimenstion_data[:, 0], dimenstion_data[:, 1], s=30, c=predict)
                plt.scatter(center[:, 0], center[:, 1], s=50, c='red', marker='s')
        else:
            plt.scatter(dimenstion_data[0], dimenstion_data[1], s=30, label=label, c=predict)

    else:
        plt.scatter(dimenstion_data[:, 0], dimenstion_data[:, 1], s=30, color='blue')


# 이미지 데이터 그리는 함수

"""
image_data : 이미지 데이터
title : 제목

n_images : 그릴 이미지 수
IMG_SIZE : reshape할 이미지 사이즈(image_data의 크기와 반드시 같아야함)

"""


def draw_image_data(image_data, title, n_images=5, IMG_SIZE=64):
    plt.figure(figsize=(100, 10))

    for i in range(n_images):
        ## display original
        ax = plt.subplot(1, n_images, i + 1)
        ax.set_title("Original Image")
        plt.imshow(image_data[i].reshape(IMG_SIZE, IMG_SIZE))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# 전처리데이터 다운로드 함수

def download_csv(path, dataframe, file_name):
    split_path = path.split('/')
    del split_path[-1]

    higher_path = "/".join(split_path)
    dataframe.to_csv(higher_path + '/' + file_name + '.csv')


# Outlier 보여주는 함수

def show_outlier(cutting_dataset, process_label, predict):
    outlier_list = []

    for index, value in enumerate(predict):
        if value == -1:
            outlier_list.append(index)

    if len(outlier_list) == 0:
        return

    outlier_list = np.array(outlier_list)

    draw_cnt = int(input("출력할 이상치 개수(현재 {}개의 이상치 탐색)".format(len(outlier_list))))

    if draw_cnt == 0:
        return

    for i in range(draw_cnt):
        idx = outlier_list[i]
        col_name = "Process " + str(process_label[idx])
        index_df = pd.DataFrame(cutting_dataset[idx], columns=[col_name])
        ax = index_df.plot()

    return ax


"""
show_scatter_to_plot : x,y 값으로 plot 그리는 함수
show_index_to_plot : index로 plot 그리는 함수
find_centroid_index : centroid와 가장 가까운 데이터(index,value) 반환하는 함수
"""


def whereclose(a, b, rtol=1e-05, atol=1e-08):
    return np.where(isclose(a, b, rtol, atol))


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


def find_scatter_index(embedding, x_value, y_value):
    finded_idx = whereclose(embedding, x_value)[0]

    for idx in finded_idx:
        if isclose(embedding[idx][1], y_value):
            return idx

    return None


def show_scatter_to_plot(embedding_dataset, cutting_dataset, process_label, x_value, y_value):
    idx = find_scatter_index(embedding_dataset, x_value, y_value)

    if idx is None:
        return None

    finded_process = process_label[idx]
    col_name = "Process " + str(finded_process)
    #index_df = pd.DataFrame(cutting_dataset[idx], columns=[col_name])
    #ax = index_df.plot()
    ax = px.line(cutting_dataset[idx])

    return ax


def show_index_to_plot(cutting_dataset, process_label, index):
    if index is None:
        return

    finded_process = process_label[index]
    col_name = "Process " + str(finded_process)
    index_df = pd.DataFrame(cutting_dataset[index], columns=[col_name])
    ax = index_df.plot()

    return ax


def find_centroid_index(embedding, predict):
    X = np.array(embedding)
    y = np.array(predict)
    clf = NearestCentroid()
    clf.fit(X, y)

    centroid = clf.centroids_
    centroid_idx = []

    predict_set = list(sorted(set(predict)))

    if predict_set[0] == -1:
        del predict_set[0]
        centroid = centroid[1:]

    for i in predict_set:
        closet_idx = None
        closet_distance = None

        for j in range(len(embedding)):
            if predict[j] == i:
                if closet_idx is None and closet_distance is None:
                    closet_idx = j
                    closet_distance = euclidean(centroid[i], embedding[j])
                else:
                    tmp_distance = euclidean(centroid[i], embedding[j])
                    if closet_distance > tmp_distance:
                        closet_idx = j
                        closet_distance = tmp_distance

        centroid_idx.append(closet_idx)
    return centroid_idx, centroid


#############################################################


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.css.config.serve_locally = True

# app.css.append_css({"external_url":"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.csshttps://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"})
server = app.server

# Create controls
county_options = [
    {"label": str(COUNTIES[county]), "value": str(county)} for county in COUNTIES
]

well_status_options = [
    {"label": str(WELL_STATUSES[well_status]), "value": str(well_status)}
    for well_status in WELL_STATUSES
]

well_type_options = [
    {"label": str(WELL_TYPES[well_type]), "value": str(well_type)}
    for well_type in WELL_TYPES
]

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("stone.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "돌멩이들",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "짱짱", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        dcc.Tabs(
            id="app-tabs",
            value="tab1",
            className="custom-tabs",
            children=[
                dcc.Tab(
                    id="Preprocess-tab",
                    label="Data Preprocessing",
                    value="tab1",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[
                        html.Div(
                            [
                                dbc.Row([
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dcc.Upload(
                                                    id='upload-data',
                                                    children=html.Div([
                                                        'Drag and Drop or ',
                                                        html.A('Select Files')
                                                    ]),
                                                    style={
                                                        'lineHeight': '60px',
                                                        'borderWidth': '1px',
                                                        'borderStyle': 'dashed',
                                                        'borderRadius': '5px',
                                                        'textAlign': 'center',
                                                    },
                                                    # Allow multiple files to be uploaded
                                                    multiple=True, className="control_label"
                                                ),
                                                html.Label('INPUT Value Column Name', className="dcc_control"),
                                                dcc.Input(id='VC', type='text', value='Value', className="dcc_control"),
                                                html.Button('Show Graph', id='graph-btn', n_clicks=0,
                                                            className="dcc_btn"),
                                                html.Label('Have Process?', className="dcc_control"),
                                                dcc.RadioItems(id='chk-process', options=[{'label': 'O', 'value': 'o'},
                                                                                          {'label': 'X',
                                                                                           'value': 'x'}]),
                                                html.Label([
                                                    "INPUT Process Column Name",
                                                    dcc.Input(id='PCN', className="dcc_control")
                                                ], id='PCN-label'),
                                                html.Label([
                                                    "Cutting method",
                                                    dcc.Dropdown(
                                                        id='cut_radio',
                                                        options=[{'label': i, 'value': i} for i in
                                                                 ['Truncation', 'Padding', 'DTW']],
                                                        value='', className="dcc_control"
                                                    )
                                                ], id='CTM-label'),

                                                html.Label([
                                                    "TIME SLICE",
                                                    dcc.Input(id='TS', type='number', value=172,
                                                              className="dcc_control")
                                                ], id='TS-label'),
                                                html.Label([
                                                    "SHIFT SIZE",
                                                    dcc.Input(id='SS', type='number', value=172,
                                                              className="dcc_control")
                                                ], id='SS-label'),
                                                html.Button('Slice', id='slice-btn', n_clicks=0, className="dcc_btn"),
                                                html.A('Download', id='download-link', n_clicks=0, className="dcc_btn"),
                                                html.Button("Download", id="download-btn"),
                                                Download(id="download"),
                                                html.Div(id='output-data-upload'),
                                                html.Div(id='graph'),
                                                html.Div([
                                                    dbc.Modal(
                                                        [
                                                            dbc.ModalHeader(html.H2("Alert")),
                                                            dbc.ModalBody(html.H4("Cutting Success")),
                                                            dbc.ModalFooter(
                                                                dbc.Button("Close", id="close-md", className="ml-auto")
                                                            ),
                                                        ],
                                                        id="modal",
                                                        size="sm",
                                                        is_open=False,
                                                        backdrop=True,
                                                        # True, False or Static for modal to not be closed by clicking on backdrop
                                                        scrollable=True,  # False or True if modal has a lot of text
                                                        centered=True,  # True, False
                                                        fade=True
                                                    )
                                                ]),
                                            ],
                                            className="pretty_container_side",
                                            id="cross-filter-options",
                                        ),width=4),

                                    dbc.Col([
                                        html.Div([
                                            html.Div(
                                                id="dataTableContainer",
                                                className="pretty_container_preprocessing_df"
                                            ),
                                            html.Div(
                                                id="GraphContainer",
                                                className="pretty_container_preprocessing_graph",
                                            )
                                        ], id="preprocessing_right-column",
                                            className="eight_columns2",
                                        ),
                                    ],width=8
                                        ##
                                        # dbc.Row(
                                        #    dbc.Col(
                                        #        html.Div(
                                        #            id="dataTableContainer",
                                        #            className="pretty_container_preprocessing_df"
                                        #        )
                                        #    )
                                        # ),
                                        # dbc.Row(
                                        #    dbc.Col(html.Div(
                                        #        id="GraphContainer",
                                        #        className="pretty_container_preprocessing_graph",
                                        #    ))
                                        # )
                                        ##

                                    )], no_gutters=True,
                                    #className="no-gutter"
                                )
                                # html.Div(
                                #    [
                                # html.Div(
                                #    id="dataTableContainer",
                                #    className="pretty_container_preprocessing_df",
                                # ),
                                # html.Div(
                                #    id="GraphContainer",
                                #    className="pretty_container_preprocessing_graph",
                                # ),
                                #   ],
                                # id="preprocessing_right-column",
                                # className="eight columns",
                                # ),
                            ],
                            # className="row flex-display",
                        ),
                    ]),
                dcc.Tab(
                    id="Embedding-tab",
                    label="Embedding",
                    value="tab2",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[
                        html.Div([
                            html.Div([
                                html.Label([
                                    "Embedding method",
                                    dcc.Dropdown(
                                        id='embedding_radio',
                                        options=[{'label': i, 'value': i} for i in ['Autoencoder', 'PCA', 'UMAP']],
                                        value='Autoencoder', className="dcc_control"
                                    )
                                ]),
                                html.Div(html.H4('Autoencoder'), id='Autoencoder_label', className="dcc_control"),

                                html.Div(html.Label('Learning_Rate'), id='Learning_Rate_label'),
                                dcc.Input(id='Learning_Rate_input', type='number'),

                                html.Div(html.Label('Batch_Size'), id='Batch_Size_label', className="dcc_control"),
                                dcc.Input(id='Batch_Size_input', type='number', className="dcc_control"),

                                html.Div(html.Label('Epoch'), id='Epoch_label', className="dcc_control"),
                                dcc.Input(id='Epoch_input', type='number', className="dcc_control"),

                                html.Div(html.Label('Test Data Size'), id='Test_Data_Size_label',
                                         className="dcc_control"),
                                dcc.Input(id='Test_Data_Size_input', type='number', className="dcc_control"),

                                html.Div(html.Label('Imaging Algorithm'), id='Imaging_Algorithm_label',
                                         className="dcc_control"),
                                dcc.RadioItems(id='IMAGING_FLAG',
                                               options=[{'label': 'RP', 'value': '1'}, {'label': 'GAF', 'value': '2'}]),

                                html.Div(html.Label('IMAGING_FLAG'), id='Imaging_Size_label',
                                         className="dcc_control"),
                                dcc.RadioItems(id='IMAGING_SIZE_FLAG', options=[{'label': 'small', 'value': '1'},
                                                                                {'label': 'middle', 'value': '2'},
                                                                                {'label': 'large', 'value': '3'}]),

                                html.Div(html.Label('n_neighbors'), id='n_neighbors_label'),
                                dcc.Input(id='n_neighbors_input', type='number'),

                                html.Div(html.Label('min_dist'), id='min_dist_label', className="dcc_control"),
                                dcc.Input(id='min_dist_input', type='number', className="dcc_control"),

                                html.Button('Embedding', id='embedding-btn', n_clicks=0, className="dcc_btn"),

                                html.Div(id='ae_history_div', children=[
                                    html.Label('Autoencoder is Updating',id='ae_hist'),
                                    dcc.Interval(id='interval_component',interval= 1000,n_intervals=0)
                                ], className="scroll_container_hist"),

                            ], className="pretty_container_side four columns",
                            ),

                            html.Div([
                                html.Div(id='learn', children=[], className="pretty_container_embedding_graph")
                            ],
                                id="embedding_right-column",
                                className="eight columns",
                            ),
                        ]
                            , className="row flex-display"
                        )]),
                dcc.Tab(
                    id="Clustering-tab",
                    label="Clustering",
                    value="tab3",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[
                        html.Div([
                            html.Div([
                                html.Label([
                                    "Clustering method",
                                    dcc.Dropdown(
                                        id='clustering_radio',
                                        options=[{'label': i, 'value': i} for i in ['K-Means', 'DBSCAN', 'K-shape']],
                                        value='K-Means'
                                    )
                                ]),
                                html.Div(html.H4('K-Means'), id='K-Means_label', className="dcc_control"),

                                html.Div(html.Label('MAX_CLUSTER_SIZE'), id='MAX_CLUSTER_SIZE_label'),
                                dcc.Input(id='MAX_CLUSTER_SIZE_input', type='number'),

                                html.Div(html.Label('EPS'), id='EPS_label', className="dcc_control"),
                                dcc.Input(id='EPS_input', type='number', className="dcc_control"),

                                html.Div(html.Label('MIN_SAMPLES'), id='MIN_SAMPLES_label', className="dcc_control"),
                                dcc.Input(id='MIN_SAMPLES_input', type='number', className="dcc_control"),

                                html.Button('Clustering', id='Clustering-btn', n_clicks=0, className="dcc_btn"),
                                html.Div(html.Label('Silhouette Coefficient'), id='Silhouette_Coefficient_label'),
                                dcc.Dropdown(
                                        id='k-means_clustering_radio',
                                        options=[],
                                        value=''
                                ),
                            ], className="pretty_container_side four columns",

                            ),
                            html.Div([
                                html.Div(id='Cluster_Plot', children=[], className="scroll_container_plot", ),
                                html.Div(id='Cluster_Graph', children=[
                                dcc.Graph(
                                    id='cluster-result')
                                ], className="scroll_container_graph", ),
                                html.Div(id='Cluster_Hover', children=[], className="scroll_container_plot", ),

                                html.Div(id='K_means_Cluster_Plot', children=[], className="scroll_container_plot", ),
                                html.Div(id='K_means_Cluster_Graph', children=[
                                    dcc.Graph(
                                    id='K-Means-result')
                                ], className="scroll_container_graph", ),
                                html.Div(id='K_means_Cluster_Hover', children=[], className="scroll_container_plot", ),

                                html.Div(id="Outlier_Plot", children=[], className="scroll_container_outlier")
                            ],
                                id="clustering_right-column",
                                className="eight columns",
                            ),
                        ],
                            className="row flex-display")
                    ])]
        )
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# # Create callbacks
# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="resize"),
#     Output("output-clientside", "children"),
#     [Input("count_graph", "figure")],
# )

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            global PATH
            global df
            # Assume that the user uploaded a CSV file

            PATH = io.StringIO(decoded.decode('utf-8'))
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), engine='python', encoding='euc-kr')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df[:5].to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        )
    ])


@app.callback(Output('dataTableContainer', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    global csv_file_name

    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        csv_file_name = list_of_names
        print("csv file name : ", csv_file_name)
        return children


# Show Graph 버튼 누르면 그래프 보여주는 함수

@app.callback(Output('GraphContainer', 'children'),
              Input('graph-btn', 'n_clicks'),
              Input('VC', 'value'))
def show_graph(n_clicks, VC):
    global df
    global dataset
    global dataset_pure

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'graph-btn' in changed_id:
        df = df.astype({VC: 'float32'})
        dataset = df[VC].to_numpy()
        dataset_pure = dataset

        children = [
            dcc.Graph(
                id='example-graph',
                style={"width": "100%", "height": "100%"},
                figure=px.line(df[VC])
            )
        ]
        return children
    else:
        return


# slice 모달 창 띄우기
@app.callback([Output('modal', "is_open"),
               Output('download-link', 'href')],
              [Input('slice-btn', 'n_clicks'),
               Input('cut_radio', 'value'),
               Input('PCN', 'value'),
               Input('VC', 'value'),
               Input('TS', 'value'),
               Input('SS', 'value'),
               Input('close-md', 'n_clicks'),
               Input('chk-process', 'value')],
              [State('modal', "is_open")])
def time_slice(n_clicks, cut_radio, pcn, vc, time_s, shift_s, n_clicks2, chk, is_open):
    global PATH
    global dataset_pure
    global dataset
    global preprocessing_csv
    global process_label
    global dataset_pure_list
    global cutting_dataset
    global cutting_dataset_pure
    global csv_file_name
    global split_name
    # global dataset_image

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # slice-btn 누르거나 close-md(모달창 열렸을때 보이는 close 버튼) 눌렀을 때
    if 'slice-btn' in changed_id:
        split_name = str(csv_file_name).split("'")

        # href = '/dash/urlToDownload?value={}'.format(split_name[1])
        # href = '/dash/urlToDownload'
        href = ''
        print("href: ", href)

        if n_clicks or n_clicks2:
            # print 는 그냥 값 확인하려고 써놓은 것들입니닷
            print("is open: ", is_open)

            # is_open 을 false 로 바꿔주면 모달 창이 닫혀요
            if is_open:
                is_open = False
            # is_open 초기값을 false 로 해놨어요!
            else:
                # Process 존재 X
                if chk == 'x':
                    dataset, preprocessing_csv, process_label, dataset_pure = align_timeseries_dataset(PATH, vc)
                    cutting_dataset = sliding_window(dataset, time_s, shift_s)
                    cutting_dataset_pure = sliding_window(dataset_pure, time_s, shift_s)

                    dataset_pure_list = []
                    process_label = []

                    for i in range(len(cutting_dataset_pure)):
                        dataset_pure_list.append(cutting_dataset_pure[i])
                        process_label.append(i)

                # Process 존재 O
                else:
                    dataset, preprocessing_csv, process_label, dataset_pure = align_timeseries_dataset(PATH, vc, pcn)
                    dataset_pure_list = []
                    for i in range(len(dataset_pure)):
                        dataset_pure_list.append(dataset_pure[i])
                    if cut_radio == 'Truncation':
                        cutting_dataset = data_truncation(dataset)
                        cutting_dataset_pure = data_truncation(dataset_pure)
                    elif cut_radio == 'Padding':
                        cutting_dataset = data_padding(dataset)
                        cutting_dataset_pure = data_padding(dataset_pure)
                    elif cut_radio == 'DTW':
                        cutting_dataset = data_dtw(dataset)
                        cutting_dataset_pure = data_dtw(dataset_pure)

            is_open = not is_open

            return [is_open, href]

        else:
            print("else ", is_open)
            return [is_open, ' ']
    else:
        return ['', '']


# 전처리 파일 다운로드 링크
#@app.server.route('/dash/urlToDownload')
#def download_csv():
#    global preprocessing_csv
#    global split_name

#    str_io = io.StringIO()
#    preprocessing_csv.to_csv(str_io)

#    mem = io.BytesIO()
#    mem.write(str_io.getvalue().encode('utf-8'))
#    mem.seek(0)
#    str_io.close()

    # file_name= "Preprocessing" + (str(split_name[1]))
    # print("file name: ",file_name)
#    return send_file(mem, as_attachment=True,
#                     attachment_filename="Preprocessing.csv",
#                     mimetype='text/csv')

from dash_extensions.snippets import send_data_frame
@app.callback(Output("download","data"),
              Input("download-btn","n_clicks"))
def func(n_clicks):
    global preprocessing_csv
    if preprocessing_csv is None:
        return
    else:
        return send_data_frame(preprocessing_csv.to_csv,'Preprocess.csv')


@app.callback([Output('cut_radio', 'style'),
               Output('PCN', 'style'),
               Output('TS', 'style'),
               Output('SS', 'style'),
               Output('PCN-label', 'style'),
               Output('TS-label', 'style'),
               Output('SS-label', 'style'),
               Output('CTM-label', 'style')
               ],
              Input('chk-process', 'value'))
def preprocessing(value):
    if value == 'o':
        return [{'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}]
    else:
        return [{'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'},
                {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}]


@app.callback([Output('Autoencoder_label', 'children'),
               Output('Learning_Rate_input', 'style'),
               Output('Batch_Size_input', 'style'),
               Output('Epoch_input', 'style'),
               Output('Test_Data_Size_input', 'style'),
               Output('IMAGING_FLAG', 'style'),
               Output('IMAGING_SIZE_FLAG', 'style'),
               Output('n_neighbors_input', 'style'),
               Output('min_dist_input', 'style'),
               Output('ae_history_div', 'style'),

               Output('Learning_Rate_label', 'style'),
               Output('Batch_Size_label', 'style'),
               Output('Epoch_label', 'style'),
               Output('Test_Data_Size_label', 'style'),
               Output('Imaging_Algorithm_label', 'style'),
               Output('Imaging_Size_label', 'style'),

               Output('n_neighbors_label', 'style'),
               Output('min_dist_label', 'style')],
              Input('embedding_radio', 'value'))
def show_vector_parameter(embedding_radio):
    if embedding_radio == "Autoencoder":
        return [html.H4('Autoencoder'), {'display': 'block'},
                {'display': 'block'}, {'display': 'block'}, {'display': 'block'},
                {'display': 'block'}, {'display': 'block'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'},
                {'display': 'block'}, {'display': 'block'}, {'display': 'block'},
                {'display': 'block'}, {'display': 'none'}, {'display': 'none'},
                ]

    elif embedding_radio == "PCA":
        return [html.H4('PCA'), {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                ]

    elif embedding_radio == "UMAP":
        return [html.H4('UMAP'), {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'block'},
                {'display': 'block'},{'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, ]

    else:
        return ['', '', '', '', '', '', '', '', '']


@app.callback(Output('learn', 'children'),
              Input('embedding-btn', 'n_clicks'),
              Input('embedding_radio', 'value'),

              Input('Learning_Rate_input', 'value'),
              Input('Batch_Size_input', 'value'),
              Input('Epoch_input', 'value'),
              Input('Test_Data_Size_input', 'value'),
              Input('IMAGING_FLAG', 'value'),
              Input('IMAGING_SIZE_FLAG', 'value'),
              Input('n_neighbors_input', 'value'),
              Input('min_dist_input', 'value'),
              )
def deep_learning(n_clicks, embedding_radio, learning_rate, batch_size, epoch, test_data_size, IMAGING_FLAG,
                  IMAGING_SIZE_FLAG, n_neighbors, min_dist):
    global cutting_dataset
    global embedding_data
    global autoencoder_hist

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'embedding-btn' in changed_id:
        if embedding_radio == 'Autoencoder':

            autoencoder_hist = []

            embedding_data = embedding_AE(cutting_dataset, learning_rate, batch_size, epoch, IMAGING_FLAG,
                                          IMAGING_SIZE_FLAG, TEST_SIZE=test_data_size)

        elif embedding_radio == 'PCA':
            embedding_data = embedding_PCA(cutting_dataset)

        elif embedding_radio == 'UMAP':
            embedding_data = embedding_UMAP(cutting_dataset, n_neighbors, min_dist)

        children = [
            dcc.Graph(
                id='embedding-result',
                style={"width": "100%", "height": "100%"},
                figure=px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1])
            )
        ]
        return children
    else:
        return

@app.callback(Output('ae_hist','children'),
              Input('interval_component','n_intervals'))
def update_hist(n_invervals):
    global autoencoder_hist
    return autoencoder_hist


@app.callback(Output('K-Means_label', 'children'),
              Output('MAX_CLUSTER_SIZE_input', 'style'),
              Output('k-means_clustering_radio', 'style'),

              Output('EPS_input', 'style'),
              Output('MIN_SAMPLES_input', 'style'),

              Output('MAX_CLUSTER_SIZE_label', 'style'),
              Output('EPS_label', 'style'),
              Output('MIN_SAMPLES_label', 'style'),
              Output('Silhouette_Coefficient_label', 'style'),

            Output('Cluster_Plot', 'style'),
            Output('Cluster_Graph', 'style'),
            Output('Cluster_Hover', 'style'),

              Output('K_means_Cluster_Plot', 'style'),
            Output('K_means_Cluster_Graph', 'style'),
              Output('K_means_Cluster_Hover', 'style'),

              Output('Outlier_Plot', 'style'),

              Input('clustering_radio', 'value'))
def cluster_option(clustering_radio):
    if clustering_radio == 'K-Means':
        return [html.H4('K-MEANS'), {'display': 'block'},{'display': 'block'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'block'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'block'},
                {'display': 'none'}, {'display': 'none'},{'display': 'none'},{'display': 'block'},
                {'display': 'block'},{'display': 'block'},{'display': 'none'}]

    elif clustering_radio == 'DBSCAN':

        return [html.H4('DBSCAN'), {'display': 'none'},{'display': 'none'},
                {'display': 'block'}, {'display': 'block'}, {'display': 'none'},
                {'display': 'block'}, {'display': 'block'}, {'display': 'none'},
                {'display': 'block'}, {'display': 'block'}, {'display': 'block'},{'display': 'none'},
                {'display': 'none'},{'display': 'none'},{'display': 'block'}]

    elif clustering_radio == 'K-shape':
        return [html.H4('K-SHAPE'), {'display': 'block'},{'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'block'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'block'},{'display': 'block'}, {'display': 'block'},{'display': 'none'},
                {'display': 'none'},{'display': 'none'}, {'display': 'none'}]


# cluster-result 는 그래프, radio_cluster_num, options 는 라디오 버튼에 동적으로 추가한 options 이고
# 'radio_cluster_num', 'value' 는 기본값을 뭘로 설정할 지, 'radio_cluster_div','style' 는 처음에 아무거나 넣어서 만들었던 라디오 버튼을
# 숨겼다가 클러스터링 하고 나서 보이게 하려고 한거에요!
@app.callback([Output('Cluster_Plot', 'children'),
               Output('Cluster_Graph', 'children'),
               Output('Outlier_Plot', 'children'),
              Output('k-means_clustering_radio', 'options')],

              [Input('Clustering-btn', 'n_clicks'),
               Input('clustering_radio', 'value'),
               Input('MAX_CLUSTER_SIZE_input', 'value'),
               Input('EPS_input', 'value'),
               Input('MIN_SAMPLES_input', 'value'), ])
def clustering(n_clicks, clustering_radio, MAX_CLUSTER_SIZE, EPS, MIN_SAMPLES):
    global embedding_data
    global predict
    global dataset_pure_list
    global process_label

    global centroid_idx
    global centroid_value
    global Kmean

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    children_Plot = []
    children_Graph = []

    if 'Clustering-btn' in changed_id:

        if clustering_radio == 'K-Means':
            cluster_list,cluster_value = cal_Silhouette(embedding_data, MAX_CLUSTER_SIZE, 5)

            options = []
            for i in range(len(cluster_list)):
                options.append({'label' : str(cluster_list[i])+' ('+str(cluster_value[i])+'%)', 'value' : str(i)})
            """
            for n_cluster in cluster_list:

                predict = clustering_KMEANS(embedding_data, n_cluster, MAX_CLUSTER_SIZE)
                centroid_idx, centroid_value = find_centroid_index(embedding_data, predict)

                if len(set(predict)) > 1:

                    for i in centroid_idx:
                        children_Plot.append(
                            dcc.Graph(
                                id='clustering-plot_' + str(i),
                                figure=px.line(dataset_pure_list[i], title="Process : " + str(process_label[i]))
                            ))

                fig = px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1], width=800, height=400, color=predict)
                fig.add_trace(
                    go.Scatter(x=centroid_value[:, 0], y=centroid_value[:, 1], mode='markers',
                               marker=dict(color='red'), showlegend=False))

                children_Graph.append(
                    dcc.Graph(
                        id='K-Means-result',
                        figure=fig)
                )

            predict = clustering_KMEANS(embedding_data, cluster_list[0], MAX_CLUSTER_SIZE)
            centroid_idx, centroid_value = find_centroid_index(embedding_data, predict)
            """
            # 동적으로 options 에 실루엣 계수 상위 5개 추가
            # options = [{'label': v, 'value': v} for v in predict]

            # children = []

            # # 우선 실루엣 계수 제일 높은 거 한 개만 출력되도록 했습니다
            # Kmean = KMeans(n_clusters=cluster_list[0])
            # Kmean.fit(vector_result)
            # fig = px.scatter(x=vector_result[:, 0], y=vector_result[:, 1], width=800, height=400, color=Kmean.labels_.astype(float))
            # fig.add_trace(go.Scatter(x=Kmean.cluster_centers_[:, 0], y=Kmean.cluster_centers_[:, 1], mode='markers', marker=dict(color='red'), showlegend=False))
            #
            # children = (html.Div([
            #     html.H4(children=["Cluster Num: ", cluster_list[0]]),
            #     dcc.Graph(
            #         id='kmeans-result',
            #         figure=fig
            #     )]
            # ))
            return ["", None, "",options]

        elif clustering_radio == "DBSCAN":
            predict = clsutering_DBSCAN(embedding_data, EPS, MIN_SAMPLES)

            children_Plot = []
            children_Graph = []
            outlier_Plot = []

            if len(set(predict)) > 1:

                centroid_idx, centroid_value = find_centroid_index(embedding_data, predict)

                for j in range(len(centroid_idx)):
                    children_Plot.append(
                        dcc.Graph(
                            id='clustering-plot_' + str(centroid_idx[j]),
                            style={'display': 'inline-block', "autosize": "false", "width": "33.3333%",
                                   "height": "100%"},
                            figure=px.line(dataset_pure_list[centroid_idx[j]], title="DBSCAN Process : " + str(process_label[centroid_idx[j]])+"( {}, {} )".format(round(centroid_value[j][0],2),round(centroid_value[j][1],2)) )
                        ))
                fig = px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1], width=800, height=400, color=predict)
                fig.add_trace(
                go.Scatter(x=centroid_value[:, 0], y=centroid_value[:, 1], mode='markers',
                           marker=dict(color='red'), showlegend=False))

                children_Graph.append(
                    dcc.Graph(
                        id='cluster-result',
                        figure=fig)
                )
            else:
                fig = px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1], width=800, height=400, color=predict)

                children_Graph.append(
                     dcc.Graph(
                         id='cluster-result',
                         figure=fig)
                 )

            if process_label != None:

                outlier_list = []

                for index, value in enumerate(predict):
                    if value == -1:
                        outlier_list.append(index)

                if len(outlier_list) != 0:

                    outlier_list = np.array(outlier_list)

                    for idx in outlier_list:
                        outlier_Plot.append(
                            dcc.Graph(
                                id='outlier-plot_' + str(idx),
                                style={'display': 'inline-block', "autosize": "false", "width": "33.3333%",
                                       "height": "100%"},
                                figure=px.line(dataset_pure_list[idx],
                                               title="Outlier Process : " + str(process_label[idx]))
                            ))

            # dbscan 에서는 라디오 버튼 안보여야 되니까 style 을 display:none 으로 했어요
            return [children_Plot, children_Graph, outlier_Plot,[]]

        elif clustering_radio == "K-shape":

            n_cluster = cal_Silhouette(embedding_data, MAX_CLUSTER_SIZE, 5)[0][0]
            predict = clustering_KSHAPE(embedding_data, n_cluster=n_cluster)

            children_Graph = []
            children_Plot = []

            if len(set(predict)) > 1:

                for j in range(len(centroid_idx)):
                    children_Plot.append(
                        dcc.Graph(
                            id='clustering-plot_' + str(centroid_idx[j]),
                            figure=px.line(dataset_pure_list[centroid_idx[j]],
                                           title="K-SHAPE Process : " + str(process_label[centroid_idx[j]]) + "( {}, {} )".format(
                                               round(centroid_value[j][0], 2), round(centroid_value[j][1], 2)))
                        ))

            fig = px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1], width=800, height=400, color=predict)

            children_Graph.append(dcc.Graph(
                 id='cluster-result',
                 figure=fig
             )
            )
            return [children_Plot, children_Graph, "",[]]
        else:
            return []
    else:
        return [children_Plot, children_Graph, "",[]]

@app.callback([Output('K_means_Cluster_Plot', 'children'),
               Output('K_means_Cluster_Graph', 'children')],
              [Input('k-means_clustering_radio', 'value'),
               Input('MAX_CLUSTER_SIZE_input', 'value')])
def k_means_clustering(k_means_radio,MAX_CLUSTER_SIZE):
    global embedding_data
    global predict
    global dataset_pure_list
    global process_label

    global centroid_idx
    global centroid_value

    if k_means_radio=='':
        return ["",None]

    i = int(k_means_radio)
    children_Plot = []
    children_Graph = []

    cluster_list, cluster_value = cal_Silhouette(embedding_data, MAX_CLUSTER_SIZE, 5)

    predict = clustering_KMEANS(embedding_data, cluster_list[i], MAX_CLUSTER_SIZE)
    centroid_idx, centroid_value = find_centroid_index(embedding_data, predict)

    if len(set(predict)) > 1:

        for j in range(len(centroid_idx)):
            children_Plot.append(
                dcc.Graph(
                    id='clustering-plot_' + str(centroid_idx[j]),
                    style={'display': 'inline-block', "autosize": "false", "width": "33.3333%", "height": "100%"},
                    figure = px.line(dataset_pure_list[centroid_idx[j]],
                    title="K-MEANS Process : " + str(process_label[centroid_idx[j]]) + "( {}, {} )".format(
                                 round(centroid_value[j][0], 2), round(centroid_value[j][1], 2)))
                ))

        fig = px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1], width=800, height=400, color=predict)
        fig.add_trace(go.Scatter(x=centroid_value[:, 0], y=centroid_value[:, 1], mode='markers',marker=dict(color='red'), showlegend=False))

        children_Graph.append(
             dcc.Graph(
                 id='K-Means-result',
                 figure=fig)
        )

        predict = clustering_KMEANS(embedding_data, cluster_list[0], MAX_CLUSTER_SIZE)
        centroid_idx, centroid_value = find_centroid_index(embedding_data, predict)

    else:
        fig = px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1], width=800, height=400, color=predict)

        children_Graph.append(
            dcc.Graph(
                id='K-Means-result',
                figure=fig)
        )

    return [children_Plot,children_Graph]

"""
@app.callback(Output('Real-Result', 'children'),
              Input('cluster_num_radio_for_kmeans', 'value'),
              Input('view-btn', 'n_clicks'),)
def view_graph(cluster_num,n_clicks):
    global embedding_data
    global Kmean

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'view-btn' in changed_id:
        Kmean = KMeans(n_clusters=cluster_num)
        Kmean.fit(embedding_data)

        fig = px.scatter(x=embedding_data[:, 0], y=embedding_data[:, 1], width=800, height=400,
                         color=Kmean.labels_.astype(float))
        fig.add_trace(go.Scatter(x=Kmean.cluster_centers_[:, 0], y=Kmean.cluster_centers_[:, 1], mode='markers',
                                 marker=dict(color='red'), showlegend=False))
        children = (html.Div([
            html.H4(children=["Cluster Num: ", cluster_num]),
            dcc.Graph(
                id='kmeans-result',
                figure=fig
            )]
        ))
        return children
"""

@app.callback(Output('Cluster_Hover', 'children'),
              Input('cluster-result', 'hoverData'))
def cluster_hover(hoverData):

    global process_label
    global embedding_data
    global dataset_pure_list

    if hoverData != None:

        idx = find_scatter_index(embedding_data, hoverData['points'][0]['x'],
                                 hoverData['points'][0]['y'])

        if idx is None:
            return dash.no_update

        children = [
            dcc.Graph(
                id='Hover_Process_Plot',
                figure=px.line(dataset_pure_list[idx],
                               title="Process : " + str(process_label[idx]) + " ({},{})".format(
                                   round(hoverData['points'][0]['x'], 2), round(hoverData['points'][0]['y'], 2)))
            )
        ]
        return children
    else:
        children = []
        return children

@app.callback(Output('K_means_Cluster_Hover', 'children'),
              Input('K-Means-result', 'hoverData'))
def k_means_hover(hoverData):

    global process_label
    global embedding_data
    global dataset_pure_list

    if hoverData != None:

        idx = find_scatter_index(embedding_data,hoverData['points'][0]['x'],
                                   hoverData['points'][0]['y'])

        if idx is None:
            return dash.no_update

        children=[
            dcc.Graph(
                id='Hover_Process_Plot',
                figure=px.line(dataset_pure_list[idx],
                               title="Process : " + str(process_label[idx]) + " ({},{})".format(
                                   round(hoverData['points'][0]['x'], 2), round(hoverData['points'][0]['y'], 2)))
            )
        ]
        return children
    else:
        children = []
        return children

# Main
if __name__ == "__main__":
    app.run_server(debug=True)