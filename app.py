from flask import Flask
from flask import request
import main

#main.init_recommendation_system()

app = Flask(__name__)

@app.route('/recomend')
def hello():
    print(main.recomend_to_user(2217)[0])
    return str(main.recomend_to_user(2217))

@app.route('/similar')
def hello2():
    product_name = request.args.get('product')
    return str(main.similar_items(product_name))

    #Творог;40;Пятёрочка