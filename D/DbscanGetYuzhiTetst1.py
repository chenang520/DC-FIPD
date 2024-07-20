import matplotlib.pyplot as plt
import pymysql
import numpy as np
from tqdm import tqdm

def dataProcessing(data):
    riskIPvalues = np.array([float(item[1]) for item in data])
    results = riskIPvalues * 1000
    data1 = np.round(results, 1)
    sorted_data = np.sort(data1)
    return sorted_data

def getYuzhi(data):
    lowDataNum = 0
    bigDataNum = 0
    result = {}
    second_data = [item[2] for item in data]

    data_0 = [item for item in data if item[2] == 0]
    data_1 = [item for item in data if item[2] == 1]

    sorted_data0 = dataProcessing(data_0)   #正常IP
    sorted_data1 = dataProcessing(data_1)  #风险IP

    for threshold_value in np.arange(0.1, 1.1, 0.05):
        threshold_index0 = int(np.floor(threshold_value * len(sorted_data0))) - 1
        threshold_index1 = int(np.floor((1-threshold_value) * len(sorted_data1))) - 1

        threshold0 = sorted_data0[threshold_index0]
        threshold1 = sorted_data1[threshold_index1]
        if threshold0 < threshold1:
            threshold = (threshold0+threshold1)/2
            print("阈值：", threshold)

            data = sorted_data0
            for i in range(len(data) - 1):
                integer_part1 = int(data[i])
                integer_part2 = int(data[i + 1])
                if integer_part1 != integer_part2:
                    key = (integer_part1, integer_part2)
                    if key in result:
                        result[key].append(data[i])
                    else:
                        result[key] = [data[i]]
                if data[i] < threshold:
                    lowDataNum = lowDataNum + 1
                else:
                    bigDataNum = bigDataNum + 1

            print("风险IP中小于阈值{}的数量为：{}".format(threshold, lowDataNum))
            print("风险IP中大于阈值{}的数量为：{}".format(threshold, bigDataNum))
            print("风险IP识别正确率：{}，错误率{}".format(bigDataNum / (lowDataNum + bigDataNum),
                                                        lowDataNum / (lowDataNum + bigDataNum)))
            data = sorted_data1
            for i in range(len(data) - 1):
                integer_part1 = int(data[i])
                integer_part2 = int(data[i + 1])
                if integer_part1 != integer_part2:
                    key = (integer_part1, integer_part2)
                    if key in result:
                        result[key].append(data[i])
                    else:
                        result[key] = [data[i]]
                if data[i] < threshold:
                    lowDataNum = lowDataNum + 1
                else:
                    bigDataNum = bigDataNum + 1
            print("正常IP中小于阈值{}的数量为：{}".format(threshold, lowDataNum))
            print("正常IP中大于阈值{}的数量为：{}".format(threshold, bigDataNum))
            print("正常IP识别正确率：{}，错误率{}".format(lowDataNum / (lowDataNum + bigDataNum),
                                                        bigDataNum / (lowDataNum + bigDataNum)))

            return threshold

    
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
                sql = "SELECT ip,`{}` ,risk FROM dbscan_ip_riskvalue_traindata".format(column_name)
                cursor.execute(sql)
                data = cursor.fetchall()
                threshold = getYuzhi(data)
                sql1 = """
                            SELECT COLUMN_NAME, COLUMN_COMMENT
                            FROM information_schema.COLUMNS
                            WHERE TABLE_SCHEMA = 'dc-fipd'
                            AND TABLE_NAME = 'dbscan_ip_riskvalue_traindata' AND column_name = %s
                """
                cursor.execute(sql1, (column_name,))
                result = cursor.fetchall()
                parameter = result[0][0]
                weight = result[0][1]
                # print(f"字段名: {parameter}, 备注: {weight}")
                sql2 = "INSERT INTO dbscan_parameter_weight_threshold (parameter,weight,threshold) VALUES (%s,%s,%s)"
                cursor.execute(sql2, (parameter, weight, threshold))
                db.commit()
    cursor.close()
    db.close()
