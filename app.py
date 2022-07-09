from flask import Flask
import main

#main.init_recommendation_system()

app = Flask(__name__)

@app.route('/recomend')
def hello():
    return str(main.recomend_to_user(2217))