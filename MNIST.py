from neuralnetwork import *

model = NeuralNetwork()

model.add_layer(size=784)

model.add_layer(size=64)

model.add_layer(size=64)

model.add_layer(size=10)

model.load_model("fashion_mnist_model.npz")


images_test = DataLoader.load_images("./MNIST_FASHION/t10k-images-idx3-ubyte")
labels_test = DataLoader.load_labels("./MNIST_FASHION/t10k-labels-idx1-ubyte")

image = DataLoader.preprocess_image("./MNIST_FASHION/ankleboot.jpeg")

model.predict(image, individual=True)



