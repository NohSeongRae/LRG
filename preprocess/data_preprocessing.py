# 1) Remove self-loop, multi-edge

import pandas as pd
import numpy as np
import pyproj
import folium
import os

def preprocess_edge(city_name):

    city_edge = pd.read_csv("../datasets/cities/original/atlanta_edge_original.csv")
    city_node = pd.read_csv("../datasets/cities/original/atlanta_node_original.csv")

    node_id = list(city_node['id'])
    node_lon = list(city_node['lon'])
    node_lat = list(city_node['lat'])

    edge_src = list(city_edge['src'])
    edge_dst = list(city_edge['dst'])
    edge_length = list(city_edge['weight'])

    ## 1-1) self-loop 삭제

    self_loop = []

    for i in range(len(edge_src)):
        if edge_src[i] == edge_dst[i]:
            self_loop.append(i)

    k = 0
    for i in range(len(self_loop)):
        del edge_length[self_loop[i] - k]
        del edge_src[self_loop[i] - k]
        del edge_dst[self_loop[i] - k]
        k += 1

    ## 1-2) multi-edge 삭제

    src_dst = []
    multi_edge = []

    for i in range(len(edge_src)):
        src_dst.append((edge_src[i], edge_dst[i]))

    for j in range(len(src_dst) - 1):
        if src_dst[j] == src_dst[j + 1]:
            multi_edge.append(j + 1)

    k = 0
    for i in range(len(multi_edge)):
        del edge_length[multi_edge[i] - k]
        del edge_src[multi_edge[i] - k]
        del edge_dst[multi_edge[i] - k]
        k += 1

    ## 1-3) Sorting

    node_dict = {}

    for i in range(len(node_id)):
        node_dict[node_id[i]] = i
        node_id[i] = i

    for i in range(len(edge_src)):
        edge_src[i] = node_dict[edge_src[i]]
        edge_dst[i] = node_dict[edge_dst[i]]

    ## 1-4) Save

    edge_df = pd.DataFrame(edge_src, columns=['src'])
    edge_df['dst'] = edge_dst
    edge_df['weight'] = edge_length

    edge_df.to_csv("../datasets/cities/norm/atlanta_edge_norm.csv", index=False)

# 2) 투영좌표계(PCS)로 변환

def project_array(coord, p1_type, p2_type):
    """
    좌표계 변환 함수
    - coord: x, y 좌표 정보가 담긴 NumPy Array
    - p1_type: 입력 좌표계 정보 ex) epsg:4326
    - p2_type: 출력 좌표계 정보 ex) epsg:2097
    """
    p1 = pyproj.Proj(init=p1_type)
    p2 = pyproj.Proj(init=p2_type)
    fx, fy = pyproj.transform(p1, p2, coord[:, 0], coord[:, 1])
    return np.dstack([fx, fy])[0]

def preprocess_node(city_name):
    city_node = pd.read_csv("../datasets/cities/original/atlanta_node_original.csv")

    node_id = []
    node_lon = list(city_node['lon'])
    node_lat = list(city_node['lat'])

    for i in range(len(node_lon)):
        node_id.append(i)

    df = pd.DataFrame(node_lon)
    df[1] = node_lat

    coord = np.array(df)

    p1_type = "epsg:4326"
    p2_type = "epsg:2097"

    result = project_array(coord, p1_type, p2_type)

    PCS_lon = result[:, 0]
    PCS_lat = result[:, 1]

    ## 2-1) normalization

    norm_lon = []
    norm_lat = []

    if (max(PCS_lon) - min(PCS_lon)) > (max(PCS_lat) - min(PCS_lat)):
        ratio = (max(PCS_lat) - min(PCS_lat)) / (max(PCS_lon) - min(PCS_lon))
        lon_min = min(PCS_lon)
        lon_max = max(PCS_lon)
        norm_lon = list(map(lambda PCS_lon: 2 * (PCS_lon - lon_min) / (lon_max - lon_min) - 1, PCS_lon))
        lat_min = min(PCS_lat)
        lat_max = max(PCS_lat)
        norm_lat = list(map(lambda PCS_lat: (2 * ratio) * (PCS_lat - lat_min) / (lat_max - lat_min) - ratio, PCS_lat))
    else:
        ratio = (max(PCS_lon) - min(PCS_lon)) / (max(PCS_lat) - min(PCS_lat))
        lat_min = min(PCS_lat)
        lat_max = max(PCS_lat)
        norm_lat = list(map(lambda PCS_lat: 2 * (PCS_lat - lat_min) / (lat_max - lat_min) - 1, PCS_lat))
        lon_min = min(PCS_lon)
        lon_max = max(PCS_lon)
        norm_lon = list(map(lambda PCS_lon: (2 * ratio) * (PCS_lon - lon_min) / (lon_max - lon_min) - ratio, PCS_lon))

    # 3) Save

    node_df = pd.DataFrame(node_id, columns=['id'])
    node_df['lon'] = norm_lon
    node_df['lat'] = norm_lat

    node_df.to_csv("../datasets/cities/norm/atlanta_node_norm.csv", index=False)





