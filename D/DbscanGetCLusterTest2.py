# 导入必要的库和模块
import csv
import time

import pandas as pd
import numpy as np
import folium
import pymysql
from ipwhois import IPWhois
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.utils import shuffle
from tqdm import tqdm

# 计算地球上两点之间的距离（单位：千米）
def haversine(lon1, lat1, lon2, lat2):
    """
    计算地球上两点之间的距离（单位：千米）
    :param lat1: 第一个点的纬度
    :param lon1: 第一个点的经度
    :param lat2: 第二个点的纬度
    :param lon2: 第二个点的经度
    :return: 两个点之间的距离（单位：千米）
    """
    # 将经纬度转换为弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # 计算经纬度差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 计算距离
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371 * c  # 地球半径为6371千米

    return distance


def DBSCAN_Conbine(IPs, eps=1.0, min_samples=10):
    
    data = pd.read_csv('TrainData.csv')
    longitude = data.iloc[:, 1]
    latitude = data.iloc[:, 2]
    result = np.column_stack((longitude, latitude))
    result = result.tolist()
    X = [[x, y] for x, y in result]
    X = np.array(X).reshape(-1, 2)

    # 定义聚类参数
    eps = eps
    min_samples = min_samples

    # 使用DBSCAN算法对数据进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    # 将所有噪声点标记为-1，并将其他点标记为它们所属的簇
    labels = dbscan.labels_
    labels[labels == -1] = 0

    # 获取簇的数量（不包括噪声点）
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # 创建Basemap对象，设置地图范围和投影方式
    # m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80,
    #             llcrnrlon=-180, urcrnrlon=180)
    # fig = plt.figure(figsize=(10, 8), dpi=400)
    # 国家陆地填充颜色,湖泊填充颜色
    # m.fillcontinents(color='coral', lake_color='aqua')
    # 绘制海岸线和国家边界线
    # m.drawcoastlines()
    # m.drawcountries()
    #
    # 绘制经纬度数据
    longitude = data.iloc[:, 1].tolist()
    latitude = data.iloc[:, 2].tolist()

    # x, y = m(longitude, latitude)
    # m.plot(x, y, 'bo', markersize=0.3)
    db = mysqlconnect()
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询
    # ip_cluster = np.empty((0, 2))
    str1 = "eps="
    str2 = "_min="
    column_name = str1+str(eps)+str2+str(min_samples)
    sql1 = "ALTER TABLE dbscan_ip_cluster_testdata ADD `{}` varchar(50)".format(column_name)
    cursor.execute(sql1)
    db.commit()
    for IP in IPs:
        sql = "SELECT * FROM testdata where ip='%s'" % IP[0]
        cursor.execute(sql)
        results = cursor.fetchone()
        # print(results[1], results[2])
        preLong = float(results[1])
        preLat = float(results[2])

        # x1, y1 = m(preLong, preLat)
        # m.plot(x1, y1, 'co', markersize=8)

        # count 待预测点权重（是否恶意IP的判断数据） list存放距离较近点的数据
        count = 0
        list = []
        distanceAVG = 0

        for i in range(-1, n_clusters_):
            # 获取第 i 个簇的所有样本
            if i == -1:
                cluster_i = X[labels == i]
            else:
                cluster_i = X[labels == i - 1]
            # 如果簇中样本数大于等于min_samples，则认为该簇是核心对象
            if cluster_i.shape[0] >= min_samples:
                # 计算第 i 个簇的中心点坐标
                center = np.mean(cluster_i, axis=0)

                # 计算第 i 个簇的所有点到中心点的距离
                distances = [haversine(center[0], center[1], x[0], x[1]) for x in cluster_i]

                # 找到距离最远的点
                farthest_idx = np.argmax(distances)
                farthest_point = cluster_i[farthest_idx]

                # 计算距离最远的点到该簇中心点的距离
                distance_to_farthest = haversine(center[0], center[1], farthest_point[0], farthest_point[1])
                distanceAVG = distanceAVG + distance_to_farthest
        distanceAVG = 2 * distanceAVG / n_clusters_
        # 计算每个簇的中心点到最远的点的距离
        for i in range(-1, n_clusters_):
            # 获取第 i 个簇的所有样本
            if i == -1:
                cluster_i = X[labels == i]
            else:
                cluster_i = X[labels == i-1]

            # 如果簇中样本数大于等于min_samples，则认为该簇是核心对象
            if cluster_i.shape[0] >= min_samples:
                # 计算第 i 个簇的中心点坐标
                center = np.mean(cluster_i, axis=0)

                # 计算第 i 个簇的所有点到中心点的距离
                distances = [haversine(center[0], center[1], x[0], x[1]) for x in cluster_i]

                # 找到距离最远的点
                farthest_idx = np.argmax(distances)
                farthest_point = cluster_i[farthest_idx]

                # 计算距离最远的点到该簇中心点的距离
                distance_to_farthest = haversine(center[0], center[1], farthest_point[0], farthest_point[1])

                # 计算待预测的点到该簇中心点的距离
                distance_to_prediction = haversine(center[0], center[1], preLong, preLat)
                if distance_to_farthest > distanceAVG:
                    # print("此处被删除：Cluster {}: center={}, farthest point={}, distance={}".format(i, center,
                    #                                                                                 farthest_point,
                    #                                                               distance_to_farthest))
                    # 4.根据中心点和最远距离画圆圈并放到地图里面
                    # lon, lat = center[0], center[1]
                    # radius = distance_to_farthest
                    # x, y = m(lon, lat)
                    # m.scatter(x, y, s=radius, marker='o', facecolor='none', edgecolor='g', linewidth=0.5)
                    continue
                # 判断预测点与最远距离的大小，若小于等于则在圈内
                if distance_to_prediction <= distance_to_farthest:
                    count = count + 1
                    # 计算预测点到该簇中每个点的距离
                    distances_pre = [haversine(preLong, preLat, x[0], x[1]) for x in cluster_i]

                    # 找到距离最近的点
                    nearest_idx = np.argmin(distances_pre)
                    nearest_point = cluster_i[nearest_idx]
                    # 计算距离最近的点到预测点的距离
                    distance_to_nearest = haversine(preLong, preLat, nearest_point[0], nearest_point[1])
                    list.append(nearest_point)
                    list.append(distance_to_nearest)

        sql2 = "UPDATE dbscan_ip_cluster_testdata SET `{}` = %s WHERE ip = %s".format(column_name)
        values = (count, IP[0])
        cursor.execute(sql2, values)
        db.commit()
        # ip_cluster = np.vstack((ip_cluster, [IP, count]))
    # return ip_cluster

# pymysql链接
def mysqlconnect():
    # 打开数据库连接
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='123',
                         database='dc-fipd')
    return db
# 获取DBSCAN方法的训练集聚类簇占比
if __name__ == '__main__':
    # start_time = time.time()
    db = mysqlconnect()
    cursor = db.cursor()
    sql1 = "SELECT ip FROM testdata"
    cursor.execute(sql1)
    results = cursor.fetchall()
    for result in results:
        ip = result[0]
        sql2 = "INSERT INTO dbscan_ip_cluster_testdata (ip) VALUES (%s)"
        values = (ip)
        cursor.execute(sql2, values)
    db.commit()
    with tqdm(total=70) as pbar:
        for min_sample in range(1, 7, 1):
            for j in range(5, 50, 5):
                eps = j / 10.0
                DBSCAN_Conbine(results, eps, min_sample)
                pbar.update(1)
                # end_time = time.time()
                # run_time = end_time - start_time
                # print("代码运行时间：", run_time, "秒")
                # start_time = time.time()
    print("程序已结束")
   




