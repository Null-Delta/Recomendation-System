from array import array
from collections import defaultdict
from itertools import count
import json
from fuzzywuzzy import fuzz
from metrics import get_top_metrics, get_user2product_metrics, get_user2user_metrics
import modelWork
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from implicit.als import AlternatingLeastSquares
import sys

# model = AlternatingLeastSquares(
#     factors=512, 
#     regularization=0.1, 
#     iterations =500, 
# )

model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations = 200, num_threads = 4)

data_matrix = csr_matrix((0, 0), dtype=np.int8)
users = []
products = []
merchants = []
connected_products = []

def load_data():
    global users, products, merchants, connected_products
    sys.path.append("../")
    with open("data/products.json", "r") as read_file:
        products = json.load(read_file)
    with open("data/users.json", "r") as read_file:
        users = json.load(read_file)
    with open("data/merchants.json", "r") as read_file:
        merchants = json.load(read_file)
    with open("data/1.json", "r") as read_file:
        connected_products = json.load(read_file)
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
                if index != -1: # при обучении добавлять: and userIndex < len(users)//2
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
    ids, recScores = model.recommend(user_id, items[userIndex], N=8, filter_already_liked_items=False, recalculate_user=True)

    recomended_products = []

    for id in ids:
        recomended_products.append(products[id])

    return json.dumps(recomended_products, ensure_ascii=False )

def similar_items(item):
    itemIndex = list_func_index(products, lambda it: (it["name"] + ";" + str(it["cost"]) + ";" + it["merchantName"]) == item)
    ids, recScores = model.similar_items(itemIndex, N=30)

    similar_products = []

    for id in ids:
        similar_products.append(products[id])

    return json.dumps(similar_products[1:])

def similar_users(user):
    user = int(user)
    userIndex = list_func_index(users, lambda it: it["userId"] == user)
    ids, recScores = model.similar_users(userIndex, N=30)

    similar_users = []

    for id in ids:
        similar_users.append(users[id])

    cnt = defaultdict(int)
    for x in similar_users:
        for y in x["checks"]:
            for z in y:
                cnt[z] +=1
    tmp = cnt.keys()
    indexTopList = sorted(list(tmp), key = lambda x: -cnt[x])

    rezult = []
    for x in indexTopList[:30]:
        params = x.split(";")
        rezult.append({"name": params[0],"cost": int(params[1]),"merchantName": params[2]})
    return json.dumps(rezult)

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

def with_this_products():
    matrixMapCouple = []
    for productId in range(len(products)):
        print(productId)
        prod = products[productId]
        mapCouple = defaultdict(int)
        for x in users:
            #print(x["userId"])
            for y in x["checks"]:
                if len(list(filter(lambda x: x == prod['name'] + ";" + str(prod["cost"]) + ";" + prod["merchantName"], y))) != 0:
                    for z in y:
                        params = z.split(";")
                        index = products.index(
                                {
                                    "name": params[0],
                                    "cost": int(params[1]),
                                    "merchantName": params[2]
                                }
                            )
                        if (products[productId]["merchantName"] != params[2]):
                            mapCouple[str(index)] += int(1) 
                        else:
                            mapCouple[str(index)] += int(2)                  
        tmp = mapCouple.keys()
        indexTopList = sorted(list(tmp), key = lambda x: -mapCouple[x])
        matrixMapCouple.append(indexTopList[1:30])

    #save
    fp = open('1.json', 'w')
    fp.write(json.dumps(matrixMapCouple))
    fp.close()

def get_connected_products(product):
    productIndex = list_func_index(products, lambda it: (it["name"] + ";" + str(it["cost"]) + ";" + it["merchantName"]) == product)
    #print(len(connected_products))
    #print(productIndex)
    connectedIds = connected_products[productIndex]
    
    result = []
    for id in connectedIds:
        result.append(products[int(id)])

    return json.dumps(result[1:], ensure_ascii=False )

def searchProducts(name):
    searched = []
    for x in products:
        procent = fuzz.WRatio(name, x["name"])
        if procent>55:
            searched.append((procent,x))
    searched = sorted(searched, key = lambda x: -x[0])
    rezult = []
    for x in searched:
        rezult.append(x[1])
    
    return json.dumps(rezult, ensure_ascii=False)

def merchantProduct(user_id, name):
    user_id = int(user_id)
    userIndex = list_func_index(users, lambda us: us["userId"] == user_id)
    items = data_matrix.tocsr()
    ids, recScores = model.recommend(user_id, items[userIndex], N=len(products), filter_already_liked_items=False, recalculate_user=True)

    merchantProducts = []
    for x in merchants:
        if name == x["merchantName"]:
            for y in x["products"]:
                params = y.split(";")
                index = products.index(
                    {
                        "name": params[0],
                        "cost": int(params[1]),
                        "merchantName": params[2]
                    }
                )
                merchantProducts.append(index)
            break

    recomended_products_in_merchant = []
    for id in ids:
        if (id in merchantProducts):
            recomended_products_in_merchant.append(products[id])

    return json.dumps(recomended_products_in_merchant, ensure_ascii=False)

def updateModel(user, check):
    global data_matrix
    userIndex = list_func_index(users, lambda us: us["userId"] == user)
    items = data_matrix.tocsr()
    item = items[userIndex]

    check = json.loads(check)
    users[userIndex]["checks"].append(check)
    for x in check:
        index = products.index(x)
        if index in item.indices:
            if (item.data[ np.where(item.indices == index)] < 10):
                item.data[ np.where(item.indices == index)]+=3
            else:
                item.data[ np.where(item.indices == index)]+=1
        else:
            array = item.toarray()
            array[0][index] = 5
            item = csr_matrix(array)
    
    model.partial_fit_users([user], item)
    items = lil_matrix(items)
    items[userIndex] = item
    data_matrix = csr_matrix(items)

def top_merchantProduct(name):
    cnt = defaultdict(int)
    for x in users:
        for y in x["checks"]:
            for z in y:
                params = z.split(";")
                if (name == params[2]):
                    index = products.index(
                        {
                            "name": params[0],
                            "cost": int(params[1]),
                            "merchantName": params[2]
                        }
                    )
                    cnt[index]+=1

    indexTopList = sorted(list(cnt.keys()), key = lambda x: -cnt[x])
    rezult = []
    for x in indexTopList[:30]:
        rezult.append(products[x])
    return json.dumps(rezult, ensure_ascii=False)

def search_merchantProducts(name, merchantName):
    searched = []
    for x in merchants:
        if (x["merchantName"] == merchantName):
            for y in x["products"]:
                params = y.split(";")
                procent = fuzz.WRatio(name, params[0])
                if procent>55:
                    params = {"name":params[0], "cost":params[1], "merchantName": params[2]}
                    searched.append((procent,params))
            break
    searched = sorted(searched, key = lambda x: -x[0])
    rezult = []
    for x in searched:
        rezult.append(x[1])
    
    return json.dumps(rezult, ensure_ascii=False)

def on_click_product(user, product):
    global data_matrix
    userIndex = list_func_index(users, lambda us: us["userId"] == user)
    items = data_matrix.tocsr()
    item = items[userIndex]
    
    product = json.loads(product)
    index = products.index(product)
    if index in item.indices:
        item.data[ np.where(item.indices == index)]+=1

    else:
        array = item.toarray()
        array[0][index] = 2
        item = csr_matrix(array)
    
    model.partial_fit_users([user], item)
    items = lil_matrix(items)
    items[userIndex] = item
    data_matrix = csr_matrix(items)

def start():
    global users, products, merchants, matrix, data_matrix, model
    users, products, merchants = load_data()
    matrix = construct_matrix()
    data_matrix = transform_matrix_to_csr_matrix()


#with_this_products()
# fp = open('1.txt', 'w')
# fp.write(json.dumps(ensure_ascii=False, recomend_to_user_with_merchants(2217)))
# fp.close()

#print(recomend_to_user_with_merchants(2217))

#print(similar_items('Вино;899;Пятёрочка'))

#start()
# fp = open('211.txt', 'w')
# fp.write(merchantProduct(635, "Магнит"))
# fp.close()
# start()

# print(searchProducts("Стол"))
# start()

# model = modelWork.loadModel("model_0")
# # print(get_connected_products("Пиво;100;Магнит"))
# # print(recomend_to_user(6661))
# updateModel(6661,json.dumps([{"name": "Влажный корм для взрослых кошек", "cost": 26, "merchantName": "PetShop.ru"}, {"name": "Поилка-фонтан", "cost": 3280, "merchantName": "PetShop.ru"}, 
# {"name": "Когтеточка", "cost": 800, "merchantName": "PetShop.ru"},{"name": "Кусочки в соусе для кошек", "cost": 63, "merchantName": "PetShop.ru"} ]))
# print(recomend_to_user(6661))
# print(similar_items("Говядина;1399;Пятёрочка"))
# print(similar_users(635))
#model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations = 200, num_threads = 4)
#model.fit(2*data_matrix)
#get_top_metrics(users, products) 
#0.046095630991409724
#get_user2product_metrics(users, products, model, data_matrix) 
#0.061565854045994295
#get_user2user_metrics(users, products, model)
#0.07238609012584461