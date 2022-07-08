from array import array
import json
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

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

print(finalMatrix[userid])
print(ids)
print(scores)
