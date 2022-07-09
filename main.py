from array import array
from collections import defaultdict
from itertools import count
import json
from implicit.nearest_neighbours import bm25_weight
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import sys

#model = AlternatingLeastSquares(factors=128, regularization=0.05, iterations = 200, num_threads = 4)
model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations = 200, num_threads = 4)

data_matrix = csr_matrix((0, 0), dtype=np.int8)
users = []
products = []
merchants = []

def load_data():
    sys.path.append("../")
    from metrics import normalized_average_precision
    with open("data/users.json", "r") as read_file:
        users: list = json.load(read_file)
    with open("data/products.json", "r") as read_file:
        products: list = json.load(read_file)
    with open("data/merchants.json", "r") as read_file:
        merchants: list = json.load(read_file)
    return users, products, merchants

def construct_matrix():
    matrix = []
    for index in range (0,len(users)):
        matrix.append([])
        for subIndex in range(0,len(products)):
            matrix[index].append(0)
    for userIndex in range(0, len(users)):
        for checkIndex in range(0, len(users[userIndex]["checks"])):
            for productIndex in range(0, len(users[userIndex]["checks"][checkIndex])):
                params = users[userIndex]["checks"][checkIndex][productIndex].split(";")
                index = products.index(
                    {
                        "name": params[0],
                        "cost": int(params[1]),
                        "merchantName": params[2]
                    }
                )
                if index != -1:
                    matrix[userIndex][index] += 1
    return matrix

def transform_matrix_to_csr_matrix():
    columns = []
    rows = []
    data = []

    for col in range (0,len(users)):
        for row in range(0,len(products)):
            if matrix[col][row] != 0:
                columns.append(col)
                rows.append(row)
                data.append(matrix[col][row])
    return csr_matrix((data, (columns, rows)), shape=(len(users), len(products))).tocsr()

def list_func_index(lst, func):
    for i in range(len(lst)):
        if func(lst[i]):
            return i

def recomend_to_user(user_id):
    userIndex = list_func_index(users, lambda us: us["userId"] == user_id)
    items = data_matrix.tocsr()
    ids, recScores = model.recommend(user_id, items[userIndex], N=30, filter_already_liked_items=False, recalculate_user=True)

    recomended_products = []

    for id in ids:
        recomended_products.append(products[id])

    return json.dumps(recomended_products)

def recomend_to_user_with_merchants(user_id):
    userIndex = list_func_index(users, lambda us: us["userId"] == user_id)
    items = data_matrix.tocsr()
    ids, recScores = model.recommend(user_id, items[userIndex], N=60, filter_already_liked_items=False, recalculate_user=True)

    listToSort = []
    for id in range(len(ids)):
        koef = 0
        for x in merchants:
            if (x["merchantName"] == products[ids[id]]['merchantName']):
                koef = x["koef"]
                break
        listToSort.append( (recScores[id]*koef, ids[id]) )

    idsScores = sorted(listToSort, reverse = True)

    recomended_products = []

    for id in range(len(idsScores)//2):
        recomended_products.append(products[idsScores[id][1]])

    return json.dumps(recomended_products)

users, products, merchants = load_data()
matrix = construct_matrix()
data_matrix = transform_matrix_to_csr_matrix()
model.fit(2 * data_matrix)

#print(recomend_to_user_with_merchants(2217))

fp = open('1.json', 'w')
fp.write(recomend_to_user(2217))
fp.close()
fp = open('2.json', 'w')
fp.write(recomend_to_user_with_merchants(2217))
fp.close()