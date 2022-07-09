from flask import Flask
from flask import request
import main
import modelWork

main.start()
modelName = input("Введите название модели: ")
main.model = modelWork.loadModel(modelName)


app = Flask(__name__)

@app.route('/recomend')
def hello():
    return str(main.recomend_to_user(2217))

@app.route('/similar')
def hello2():
    product_name = request.args.get('product')
    return str(main.similar_items(product_name))

@app.route('/connected')
def hello3():
    product_name = request.args.get('product')
    return str(main.get_connected_products(product_name))

    #Творог;40;Пятёрочка