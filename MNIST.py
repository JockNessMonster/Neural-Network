from neuralnetwork import *

import sys

model = NeuralNetwork()

model.add_layer(size=784)

model.add_layer(size=64)

model.add_layer(size=64)

model.add_layer(size=10)

model.load_model("fashion_mnist_model.npz")

filename = sys.argv[1]

images_test = DataLoader.load_images("./MNIST_FASHION/t10k-images-idx3-ubyte")
labels_test = DataLoader.load_labels("./MNIST_FASHION/t10k-labels-idx1-ubyte")

image = DataLoader.preprocess_image(f"./MNIST_FASHION/{filename}")

print("\n*" + "-" * 15 + "*")
model.predict(image, individual=True)
print("*" + "-" * 15 + "*\n")


