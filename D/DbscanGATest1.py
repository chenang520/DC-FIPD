import random
import pymysql
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
def fitness_function(weights, X, y):
    y = y.astype(int)
    
    weighted_sum = np.dot(X, weights)
    predictions = np.where(weighted_sum > 0, 1, 0).reshape(-1, 1)
    accuracy = np.mean(predictions == y)
    true_positives = np.sum(np.logical_and(predictions == 1, y == 1))
    false_positives = np.sum(np.logical_and(predictions == 1, y == 0))
    false_negatives = np.sum(np.logical_and(predictions == 0, y == 1))

    if (true_positives + false_positives) != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = -1  
    if (true_positives + false_negatives) != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = -1  
   
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else -1
    return f1_score
def Ga(X,y):
    population = []
    for _ in range(population_size):
        
        weights = np.random.uniform(low=0, high=1, size=3)
        population.append(weights)

    # print(population)
    for generation in range(num_generations):
        
        fitness_scores = [fitness_function(weights, X, y) for weights in population]
       
        selected_indices = np.random.choice(range(population_size), size=population_size, replace=True,
                                            p=fitness_scores / np.sum(fitness_scores))
        selected_population = [population[i] for i in selected_indices]
        selected_population = np.concatenate(selected_population)

        offspring_population = []
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(range(population_size), size=2, replace=False)
            offspring = (population[parent1] + population[parent2]) / 2
            offspring_population.append(offspring)

        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation = np.random.uniform(low=-0.1, high=0.1, size=3)
                offspring_population[i] = population[i] + mutation

        population = offspring_population

    best_index = np.argmax([fitness_function(weights,X,y) for weights in population])
    best_weights = population[best_index]
    # print("最优属性权重:", best_weights)
    return best_weights
db = pymysql.connect(host='localhost',
                         user='root',
                         password='123',
                         database='dc-fipd')
cursor = db.cursor()
sql = "SELECT ip,risk FROM traindata"
cursor.execute(sql)
results = cursor.fetchall()
for result in results:
    ip = result[0]
    risk = result[1]
    sql2 = "INSERT INTO dbscan_ip_riskvalue_traindata (ip,risk) VALUES (%s,%s)"
    cursor.execute(sql2, (ip, risk))
db.commit()
with tqdm(total=56) as pbar:
    for min_sample in range(1, 7, 1):
        for j in range(5, 50, 5):
            eps = j / 10.0
            str1 = "eps="
            str2 = "_min="
            column_name = str1 + str(eps) + str2 + str(min_sample)
            sql1 = "SELECT dbscan_ip_cluster_traindata.IP, traindata.ASNSL, traindata.HOPSL,dbscan_ip_cluster_traindata.`{}`, traindata.risk FROM dbscan_ip_cluster_traindata JOIN traindata ON dbscan_ip_cluster_traindata.IP = traindata.IP".format(
                column_name)
            cursor.execute(sql1)
            results = cursor.fetchall()
            results1 = results
            my_list = list(results)
            random.shuffle(my_list)
            results = tuple(my_list)
            data = np.array(results)
            data[:, 1:4] = data[:, 1:4].astype(float)
            data[:, 4] = data[:, 4].astype(int)
            # 分割数据
            X = data[:, 1:4] 
            y = data[:, 4]  

            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            population_size = 50 
            mutation_rate = 0.1  
            num_generations = 20  
            best_weights = Ga(X, y)
            while best_weights[0] < 0 or best_weights[1] < 0 or best_weights[2] < 0 or best_weights[2] < best_weights[
                0] or best_weights[2] < best_weights[1]:
                best_weights = Ga(X, y)  # 最优属性权重: [0.28305246 0.19224906 0.350345  ]
            # best_weights = [0.28305246,0.19224906,0.350345]
            ASN_min_value = 0
            ASN_max_value = 0
            Hop_min_value = 0
            Hop_max_value = 0
            cluster_min_value = 0
            cluster_max_value = 0
            for result in results1:
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
            sql3 = "ALTER TABLE dbscan_ip_riskvalue_traindata ADD `{}` varchar(50) COMMENT %s".format(column_name)
            cursor.execute(sql3, (best_weights1,))
            db.commit()
            for result in results1:
                ip = result[0]
                ASN = (float(result[1])-ASN_min_value)/(ASN_max_value - ASN_min_value)
                HOP = (float(result[2])-Hop_min_value)/(Hop_max_value - Hop_min_value)
                cluter = (float(result[3]) - cluster_min_value) /(cluster_max_value - cluster_min_value)
                riskvalue = float(ASN)*best_weights[0]+float(HOP)*best_weights[1]+float(cluter)*best_weights[2]
                sql4 = "UPDATE dbscan_ip_riskvalue_traindata SET `{}` = %s WHERE ip = %s".format(column_name)
                values = (riskvalue, ip)
                cursor.execute(sql4, values)
                db.commit()
            pbar.update(1)
# 关闭数据库连接
cursor.close()
db.close()
