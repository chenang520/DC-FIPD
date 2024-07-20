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
    # 统计每两个整数之间的数据
    result = {}
    # 提取第二个数据的值
    second_data = [item[2] for item in data]

    # 根据第二个数据的值将两条数据分成两个数据集
    data_0 = [item for item in data if item[2] == 0]
    data_1 = [item for item in data if item[2] == 1]

    sorted_data0 = dataProcessing(data_0)   #正常IP
    sorted_data1 = dataProcessing(data_1)  #风险IP

    for threshold_value in np.arange(0.1, 1.1, 0.05):
        # 计算要寻找的阈值的索引位置
        threshold_index0 = int(np.floor(threshold_value * len(sorted_data0))) - 1
        threshold_index1 = int(np.floor((1-threshold_value) * len(sorted_data1))) - 1

        # 获取阈值
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

    # # 计算数据的直方图
    # values, bins = np.histogram(data, bins=np.arange(0, np.max(data) + 1, 1))
    #
    # # 生成区间的字符串表示形式
    # keys = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]
    #
    # print(keys)
    # print(values)
    # # 绘制柱状图
    # plt.bar(keys, values)
    # plt.xlabel('riskValue')
    # plt.ylabel('number')
    # # plt.title('riskIP')
    # if risk == 1:
    #     plt.title('riskIP')
    #     print("风险IP中小于阈值{}的数量为：{}".format(yuzhi, lowDataNum))
    #     print("风险IP中大于阈值{}的数量为：{}".format(yuzhi, bigDataNum))
    #     print("风险IP识别正确率：{}，错误率{}".format(bigDataNum/(lowDataNum+bigDataNum), lowDataNum/(lowDataNum+bigDataNum)))
    # else:
    #     plt.title('normalIP')
    #     print("正常IP中小于阈值{}的数量为：{}".format(yuzhi, lowDataNum))
    #     print("正常IP中大于阈值{}的数量为：{}".format(yuzhi, bigDataNum))
    #     print("正常IP识别正确率：{}，错误率{}".format(lowDataNum/(lowDataNum+bigDataNum), bigDataNum/(lowDataNum+bigDataNum)))
    #
    # # x_ticks = [0, 50, 100, 200]  # x轴上的刻度值
    # # x_tick_labels = ['0', '100', '200', '300']  # x轴上的刻度标签
    # x_ticks = [0, 5, 10, 20, 30, 40, 50, 60]  # x轴上的刻度值
    # plt.xticks(x_ticks, x_ticks, rotation=45)
    # # 添加平行于x轴的线
    # plt.axvline(x=yuzhi, color='r', linestyle='--')
    # plt.axvline(x=80, color='r', linestyle='--')
    # plt.axvline(x=120, color='r', linestyle='--')
    # plt.show()

if __name__ == '__main__':
    # 建立数据库连接
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
                # 执行查询语句
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
                # 获取查询结果
                result = cursor.fetchall()
                parameter = result[0][0]
                weight = result[0][1]
                # print(f"字段名: {parameter}, 备注: {weight}")
                sql2 = "INSERT INTO dbscan_parameter_weight_threshold (parameter,weight,threshold) VALUES (%s,%s,%s)"
                cursor.execute(sql2, (parameter, weight, threshold))
                db.commit()
    # 关闭数据库连接
    cursor.close()
    db.close()