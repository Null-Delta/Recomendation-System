import readline
import main
import modelWork
import metrics


main.start()
main.model.fit(2 * main.data_matrix)
name = "model_" + str(main.model.factors) + "_" + str(main.model.iterations) + "_" + str(int(main.model.regularization * 100))

modelWork.saveModel(main.model, name)

#main.start()
main.model = modelWork.loadModel(name)

#get_top_metrics(main.users, main.products)
metrics.get_user2product_metrics(main.users, main.products, main.model, main.data_matrix)