from __future__ import division
from __future__ import print_function
import pandas as pd
from pyod.models.cblof import CBLOF
import csv

insurance = pd.read_csv("D:\datalearn\insurance.csv")
insuranceori = pd.read_csv("D:\datalearn\insurance.csv")
insurance.info()


def map_smoking(column):    # 将香烟换算成离散数值；smoker=1;no-smoker=0
    mapped = []

    for row in column:

        if row == "yes":
            mapped.append(1)
        else:
            mapped.append(0)

    return mapped


def map_sex(column):       # 将性别换算成离散数值；female=1;male=0
    mapped = []

    for row in column:

        if row == "female":
            mapped.append(1)
        else:
            mapped.append(0)

    return mapped


def standard(column):   # 归一化处理
    mapped = []
    for row in column:
        mapped.append((row - min(column))/(max(column)-min(column)))

    return mapped

# 数据预处理


insurance["smoker"] = map_smoking(insurance["smoker"])  # 置换成离散数值
insurance["sex"] = map_smoking(insurance["sex"])        # 置换成离散数值
insurance = insurance.drop('region', 1)                 # 丢弃地区信息
insurance["charges"] = standard(insurance["charges"])   # 归一化处理
insurance["age"] = standard(insurance["age"])
insurance["bmi"] = standard(insurance["bmi"])
insurance["age"] = standard(insurance["age"])
insurance["children"] = standard(insurance["children"])
insurance["smoker"] = standard(insurance["smoker"])

print(insurance)
# train CBLOF detector
clf_name = 'CBLOF'
clf = CBLOF()
clf.fit(insurance)
y_train_scores = clf.decision_scores_  # raw outlier scores
print(y_train_scores)
index = y_train_scores.argsort()
print(index)
show = []
for i in range(0, 5):
    show.append(insuranceori.iloc[index[i]])

print(show)
