
from implicit.als import AlternatingLeastSquares
import main
import modelWork


main.start()
name = "model_2"

model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations = 200, num_threads = 4)
model.fit(2*main.data_matrix)

modelWork.saveModel(model, name)