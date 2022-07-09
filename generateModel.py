import readline
import main
import modelWork


main.start()
name = input("Введите название модели: ")

modelWork.saveModel(main.model, name)