import random
import pymysql
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

db = pymysql.connect(host='localhost',
                         user='root',
                         password='123',
                         database='dc-fipd')
cursor = db.cursor()
sql = "SELECT ip,risk FROM testdata"
cursor.execute(sql)
results = cursor.fetchall()
for result in results:
    ip = result[0]
    risk = result[1]
    sql2 = "INSERT INTO dbscan_ip_riskvalue_testdata (ip,risk) VALUES (%s,%s)"
    cursor.execute(sql2, (ip, risk))
db.commit()
with tqdm(total=54) as pbar:
    for min_sample in range(1, 7, 1):
        for j in range(5, 50, 5):
            eps = j / 10.0
            str1 = "eps="
            str2 = "_min="
            column_name = str1 + str(eps) + str2 + str(min_sample)
            sql1 = "SELECT dbscan_ip_cluster_testdata.IP, testdata.ASNSL, testdata.HOPSL,dbscan_ip_cluster_testdata.`{}`, testdata.risk FROM dbscan_ip_cluster_testdata JOIN testdata ON dbscan_ip_cluster_testdata.IP = testdata.IP".format(
                column_name)
            cursor.execute(sql1)
            results = cursor.fetchall()
            sql2 = """
                                        SELECT COLUMN_NAME, COLUMN_COMMENT
                                        FROM information_schema.COLUMNS
                                        WHERE TABLE_SCHEMA = 'dc-fipd'
                                        AND TABLE_NAME = 'dbscan_ip_riskvalue_traindata' AND column_name = %s
                            """
            cursor.execute(sql2, (column_name,))
            weight = cursor.fetchall()
            weight_str = weight[0][1]
            parts = weight_str.split("_")
            best_weights = [float(part) for part in parts]
            ASN_min_value = 0
            ASN_max_value = 0
            Hop_min_value = 0
            Hop_max_value = 0
            cluster_min_value = 0
            cluster_max_value = 0
            for result in results:
                ASN = float(result[1])
                HOP = float(result[2])
                cluter = int(result[3])
                # 更新最小值
                if ASN < ASN_min_value:
                    ASN_min_value = ASN
                # 更新最大值
                if ASN > ASN_max_value:
                    ASN_max_value = ASN

                # 更新最小值
                if HOP < Hop_min_value:
                    Hop_min_value = HOP
                # 更新最大值
                if HOP > Hop_max_value:
                    Hop_max_value = HOP

                # 更新最小值
                if cluter < cluster_min_value:
                    cluster_min_value = cluter
                # 更新最大值
                if cluter > cluster_max_value:
                    cluster_max_value = cluter
            best_weights1 = str(best_weights[0])+"_"+str(best_weights[1])+"_"+str(best_weights[2])
            sql3 = "ALTER TABLE dbscan_ip_riskvalue_testdata ADD `{}` varchar(50) COMMENT %s".format(column_name)
            cursor.execute(sql3, (best_weights1,))
            db.commit()
            for result in results:
                ip = result[0]
                ASN = (float(result[1])-ASN_min_value)/(ASN_max_value - ASN_min_value)
                HOP = (float(result[2])-Hop_min_value)/(Hop_max_value - Hop_min_value)
                cluter = (float(result[3]) - cluster_min_value) /(cluster_max_value - cluster_min_value)
                riskvalue = float(ASN)*best_weights[0]+float(HOP)*best_weights[1]+float(cluter)*best_weights[2]
                sql4 = "UPDATE dbscan_ip_riskvalue_testdata SET `{}` = %s WHERE ip = %s".format(column_name)
                values = (riskvalue, ip)
                cursor.execute(sql4, values)
                db.commit()
            pbar.update(1)
# 关闭数据库连接
cursor.close()
db.close()