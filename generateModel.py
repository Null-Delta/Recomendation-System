import main
import modelWork
import metrics

from implicit.als import AlternatingLeastSquares


# Генерация модели и последующее ее сохранение в файл
main.start()

main. model = AlternatingLeastSquares(
   factors=512, 
   regularization=0.1, 
   iterations =500, 
)

main.model.fit(2 * main.data_matrix)
name = "model_" + str(main.model.factors) + "_" + str(main.model.iterations) + "_" + str(int(main.model.regularization * 100))

modelWork.saveModel(main.model, name)

# А так же вычисление метрик
main.model = modelWork.loadModel("model_512_500_10")

metrics.get_top_metrics(main.users, main.products)
metrics.get_user2product_metrics(main.users, main.products, main.model, main.data_matrix)
