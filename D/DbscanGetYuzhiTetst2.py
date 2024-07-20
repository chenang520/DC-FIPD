import matplotlib.pyplot as plt
import pymysql
import numpy as np
from tqdm import tqdm


def getACC(data, threshold):
    i = 0
    TP = 0  # True Positive
    FP = 0  # False Positive
    TN = 0  # True Negative
    FN = 0  # False Negative

    for data1 in data:
        riskvalue = float(data1[1]) * 1000
        risk = data1[2]
        if riskvalue >= threshold:
            predict = 1
        else:
            predict = 0

        if risk == 1 and predict == 1:
            TP += 1
        elif risk == 0 and predict == 1:
            FP += 1
        elif risk == 1 and predict == 0:
            FN += 1
        elif risk == 0 and predict == 0:
            TN += 1

        i += 1

    accuracy = (TP + TN) / i

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1

if __name__ == '__main__':
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='123',
                         database='dc-fipd')
    cursor = db.cursor()
    with tqdm(total=54) as pbar:
        for min_sample in range(1, 7, 1):
            for j in range(5, 50, 5):
                eps = j / 10.0
                str1 = "eps="
                str2 = "_min="
                column_name = str1 + str(eps) + str2 + str(min_sample)
                sql = "SELECT ip,`{}` ,risk FROM dbscan_ip_riskvalue_testdata".format(column_name)
                cursor.execute(sql)
                data = cursor.fetchall()
                sql1 = "SELECT threshold FROM dbscan_parameter_weight_threshold where parameter = %s"
                cursor.execute(sql1, (column_name,))
                thresholds = cursor.fetchall()
                threshold = float(thresholds[0][0])
                acc, precision1, recall, f1 = getACC(data, threshold)

                sql2 = "INSERT INTO dbscan_parameter_threshold_acc (parameter, threshold, acc, f1, precision1, recall) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql2, (column_name, threshold, acc, f1, precision1, recall))
                db.commit()
                pbar.update(1)
    cursor.close()
    db.close()
