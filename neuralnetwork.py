import numpy as np
import math
import struct
from PIL import Image

class Layer:
    def __init__(self, size: int, input_length: int):
        # This is the amount of neurons in the layer
        self.size = size

        bound = math.sqrt(6 / (input_length + self.size)) if input_length > 0 else 0

        # This initialises the weights with Xavier initialisation
        self.weights = np.random.uniform(low=-bound, high=bound, size=(size, input_length))

        # This is the value of all the activations in the layer
        self.activations = np.zeros(size)
        # This is the value of all the weighted sums in the layer
        self.weighted_sums = np.zeros(size)
        # This is the value of all the biases in the layer
        self.biases = np.zeros(size)
        # This is the value of all the delta's (back propagation)
        self.deltas = np.zeros(size)



    def __str__(self):
        string = f"\nLayer\n"
        for neuron_index, activation in enumerate(self.activations):
            string += f"Neuron ({neuron_index}) has bias: {self.biases[neuron_index]}, weights: {self.weights[neuron_index]} delta: {self.deltas[neuron_index]:.2f} and activation: {activation:.2f}\n"
        return string

class NeuralNetwork:
    
   
    def __init__(self, activation="sigmoid", loss="L2", softmax=False):
        self.activation = activation
        self.loss = loss
        self.layers = []

    
    def add_layer(self, size=1):

        if not self.layers:
            input_length = 0
        else:
            input_length = self.layers[-1].size

        new_layer = Layer(size, input_length)
        self.layers.append(new_layer)

   
    def __str__(self):
        string = ""
        for layer in self.layers:
            string += str(layer)
        return string
    
   
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    
    @staticmethod
    def derivativesigmoid(sig):
        return sig * (1 - sig)
    
    
    def take_input(self, x_input):
        self.layers[0].activations = x_input
    
    
    def forward_propagation(self, x_input):

        self.take_input(x_input)

        for previous_layer_index, layer in enumerate(self.layers[1:]):
            previous_activations = self.layers[previous_layer_index].activations
            z = np.dot(layer.weights, previous_activations) + layer.biases 
            layer.weighted_sums = z
            
            layer.activations = NeuralNetwork.sigmoid(z)
        
        return self.layers[-1].activations
        
   
    def back_propagation(self, learning_rate = 0.1, expected=np.array([]), x_input=np.array([])):

        # This forward propagates
        self.forward_propagation(x_input=x_input)

        # This finds the deltas for the final layer
        self.layers[-1].deltas = 2 * (self.layers[-1].activations - expected) * self.derivativesigmoid(self.layers[-1].activations)

        # Update output layer weights and biases
        output_layer = self.layers[-1]
        prev_activations = self.layers[-2].activations
        output_layer.biases -= learning_rate * output_layer.deltas
        output_layer.weights -= learning_rate * np.outer(output_layer.deltas, prev_activations)

        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            layer.deltas = np.dot(next_layer.weights.T, next_layer.deltas) * self.derivativesigmoid(layer.activations)
            layer.biases -= learning_rate * layer.deltas
            layer.weights -= learning_rate * np.outer(layer.deltas, self.layers[i - 1].activations)
    
   
    def train(self, x_values, y_values, learning_rate=0.1, epochs=1):
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")
            for x_value, y_value in zip(x_values, y_values):
                expected = np.zeros(len(self.layers[-1].activations))
                expected[y_value] = 1
                self.back_propagation(learning_rate=learning_rate, expected=expected, x_input=x_value)

   
    def predict(self, x_value, individual=False):
        items = {0: "T-Shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
        values = self.forward_propagation(x_value)
        prediction = np.argmax(values)
        if individual:
            print(items[prediction])
        return prediction
    
   
    def accuracy(self, x_values, y_values):
        count = 0
        for x_value, y_value in zip(x_values, y_values):
            prediction = self.predict(x_value)
            if prediction == y_value:
                count += 1
                print(f"Testing: {count}/{len(x_values)}")  
        print(f"Accuracy: {count/len(x_values)}")

    def save_model(self, filename):
        # Prepare data to save
        params = {}
        for i, layer in enumerate(self.layers):
            # Makes the parameter dictionary have weight{i} have these weights and biases
            params[f'weights_{i}'] = layer.weights
            params[f'biases_{i}'] = layer.biases
        # Saves them to a file
        np.savez(filename, **params)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        # Loads the data with numpy
        data = np.load(filename)
        # For each layer, it corresponds the weights and biases
        for i, layer in enumerate(self.layers):
            layer.weights = data[f'weights_{i}']
            layer.biases = data[f'biases_{i}']


class DataLoader:

    @staticmethod
    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            label_data = np.frombuffer(f.read(), dtype=np.uint8)
            return label_data  # Return as NumPy array

    @staticmethod
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            image_data = image_data.reshape((num, rows * cols)).astype(np.float32) / 255.0
            return image_data  # Return as NumPy array
        
    @staticmethod
    def preprocess_image(filename):
        # Converts image to black and white
        img = Image.open(filename).convert('L')

        # Resizes the image to 28 by 28 pixels
        img = img.resize((28, 28))

        img_array = np.array(img) / 255.0

        img_array = 1 - img_array

        img_array = img_array.flatten()

        return img_array


if __name__ == "__main__":
    model = NeuralNetwork()

    model.add_layer(size=784)

    model.add_layer(size=64)

    model.add_layer(size=64)

    model.add_layer(size=10)

    labels = DataLoader.load_labels("./MNIST_FASHION/train-labels-idx1-ubyte")
    images = DataLoader.load_images("./MNIST_FASHION/train-images-idx3-ubyte")


    images_test = DataLoader.load_images("./MNIST_FASHION/t10k-images-idx3-ubyte")
    labels_test = DataLoader.load_labels("./MNIST_FASHION/t10k-labels-idx1-ubyte")


    small_images = images[:10000]
    small_labels = labels[:10000]

    model.train(small_images, small_labels, epochs=30)

    model.save_model("fashion_mnist_model.npz")