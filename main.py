from array import array
from collections import defaultdict
from itertools import count
import json
from implicit.nearest_neighbours import bm25_weight
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import sys
sys.path.append("../")
from metrics import normalized_average_precision

with open("data/users.json", "r") as read_file:
    users: list = json.load(read_file)

with open("data/products.json", "r") as read_file:
    products: list = json.load(read_file)

matrix = []

for index in range (0,len(users)):
    matrix.append([])
    for subIndex in range(0,len(products)):
        matrix[index].append(0)

#print(matrix)

for userIndex in range(0, len(users)):
    for checkIndex in range(0, len(users[userIndex]["checks"])):
        for productIndex in range(0, len(users[userIndex]["checks"][checkIndex])):
            params = users[userIndex]["checks"][checkIndex][productIndex].split(";")
            #print(len(products))
            index = products.index(
                {
                    "name": params[0],
                    "cost": int(params[1]),
                    "merchantName": params[2]
                }
            )
            if index != -1:
                matrix[userIndex][index] += 1

columns = []
rows = []
data = []

for col in range (0,len(users)):
    for row in range(0,len(products)):
        if matrix[col][row] != 0:
            columns.append(col)
            rows.append(row)
            data.append(matrix[col][row])

finalMatrix = csr_matrix((data, (columns, rows)), shape=(len(users), len(products))).tocsr()
#print(finalMatrix)

model = AlternatingLeastSquares(factors=128, regularization=0.05, iterations = 800, num_threads = 4)
model.fit(2 * finalMatrix)

print(finalMatrix.shape)
#print(ids)
#print(scores)
cnt = defaultdict(int)
for x in users:
    for y in x["checks"]:
        for z in y:
            params = z.split(";")
            index = products.index(
                {
                    "name": params[0],
                    "cost": int(params[1]),
                    "merchantName": params[2]
                }
            )
            cnt[index]+=1

indexTopList = sorted(list(cnt.keys()), key = lambda x: -cnt[x])
nameTopList=[]
text = ""
for x in indexTopList:
    #text += str(x)+" "+str(cnt[x])+" "+products[x]['name']+"\n"
    nameTopList.append(products[x]['name'])

#top-30
# scores = []
# for x in users:
#     testing = []
#     for y in range(10):
#         for z in x['checks'][-y]:
#             params = z.split(";")
#             index = products.index(
#                 {
#                     "name": params[0],
#                     "cost": int(params[1]),
#                     "merchantName": params[2]
#                 }
#             )
#             testing.append(index)
#     metr = normalized_average_precision(testing, indexTopList[:30])
#     scores.append(metr)
# print(np.mean(scores))

#0.04157494615140849

#recomended
def list_func_index(lst, func):
    for i in range(len(lst)):
        if func(lst[i]):
          return i

items = finalMatrix.tocsr()
scores = []
for x in users:
    userIndex = list_func_index(users, lambda us: us["userId"] == x["userId"])
    ids, recScores = model.recommend(x['userId'], items[userIndex], N=30, filter_already_liked_items=False, recalculate_user=True)
    testing = []
    for y in range(10):
        for z in x['checks'][-y]:
            params = z.split(";")
            index = products.index(
                {
                    "name": params[0],
                    "cost": int(params[1]),
                    "merchantName": params[2]
                }
            )
            testing.append(index)
    metr = normalized_average_precision(testing, ids)
    scores.append(metr)

print(np.mean(scores))
#0.033203016505148696
#0.021833127570705112
#0.021983666957240824
#0.10230769231564321