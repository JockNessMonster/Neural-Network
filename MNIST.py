from neuralnetwork import *

model = NeuralNetwork()

model.add_layer(size=784)

model.add_layer(size=64)

model.add_layer(size=64)

model.add_layer(size=10)

model.load_model("fashion_mnist_model.npz")

image = DataLoader.preprocess_image("./MNIST_FASHION/dress.jpeg")

model.predict(image, individual=True)



