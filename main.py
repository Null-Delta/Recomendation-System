from array import array
from collections import defaultdict
from itertools import count
import json
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

finalMatrix = csr_matrix((data, (columns, rows)), shape=(len(users), len(products)))
#print(finalMatrix)

model = AlternatingLeastSquares(factors=16, regularization=0.05)
model.fit(2 * finalMatrix)

userid = 0
ids, scores = model.recommend(userid, finalMatrix[userid], N=10, filter_already_liked_items=False)

#print(finalMatrix[userid])
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

scores = []
for x in users:
    testing = []
    for y in range(3):
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
    print(len(testing))
    metr = normalized_average_precision(testing, indexTopList[:30])
    scores.append(metr)

print(np.mean(scores))
#0.04157494615140849