from flask import Flask
from flask import request
from implicit.als import AlternatingLeastSquares
import main
import modelWork

main.start()
modelName = input("Введите название модели или _: ")
if modelName != "_":
    main.model = modelWork.loadModel(modelName)
else:
    main.model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations = 200, num_threads = 4)
    main.model.fits(2 * main.data_matrix)


app = Flask(__name__)

@app.route('/recomend')
def hello():
    #print(main.recomend_to_user(2217)[0])
    return str(main.recomend_to_user(2217))

@app.route('/similar_items')
def hello2():
    product_name = request.args.get('product')
    return str(main.similar_items(product_name))

@app.route('/connected')
def hello3():
    product_name = request.args.get('product')
    return str(main.get_connected_products(product_name))

@app.route('/similar_users')
def hello4():
    user_id = request.args.get('user')
    return str(main.similar_users(user_id))

@app.route('/globalSearch')
def hello5():
    search = request.args.get('search')
    return str(main.searchProduct(search))

@app.route('/merchant')
def hello6():
    name = request.args.get('name')
    return str(main.merchantProduct(name))

    #Творог;40;Пятёрочка