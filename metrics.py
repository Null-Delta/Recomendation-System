from collections import defaultdict

import modelWork
import main
import numpy as np


def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k

def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0

    ap = average_precision(actual, recommended, k=k)
    ap_ideal = average_precision(actual, list(actual)[:k], k=k)
    return ap / ap_ideal

def get_top_metrics(users, products):
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

    scores = []
    for x in users:
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
        metr = normalized_average_precision(testing, indexTopList[:30])
        scores.append(metr)
    print(np.mean(scores))
    #0.04157494615140849

def get_user2product_metrics(users,products,model,finalMatrix):
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
    #0.10230769231564321


main.start()
main.model = modelWork.loadModel("model_0")

get_top_metrics(main.users, main.products)
get_user2product_metrics(main.users, main.products, main.model, main.data_matrix)