import random
from PIL import Image, ImageDraw
import math
import struct
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import center_of_mass
from scipy.ndimage import shift

'''
Layer Class:
 - Layer count
 - Neurons
'''
class Layer:

    layer_count = 1

    def __init__(self, size):
        self.size = size
        self.neurons = [Neuron(0, 0, random.uniform(0, 0.5), self) for _ in range(size)]
        self.layer_index = Layer.layer_count
        self.layer_type = "Input" if Layer.layer_count == 1 else "Output"
        Layer.layer_count += 1
    
    def __str__(self):
        string = f"{self.layer_type} layer {self.layer_index} with {self.size} neurons: \n\n"
        for neuron in self.neurons:
            string += f"{str(neuron)}\n"
        string += "_" * 150
        return string
    

'''
Neuron Class:
 - Activation (Floating Number)
 - Weighted Sum (Floating Number)
 - Weights [(Input, Output)] - (Instance of Weight Object for each)
'''
class Neuron:

    index = 0
    all_neurons = []

    def __init__(self, activation: float, weighted_sum: float, bias: float, layer: Layer):
        # Activation is the activation value after the sigmoid function
        self.activation = activation
        # Weighted Sum is the weighted sum of the neuron, which is sum(a[L - 1] * w_k)
        self.weighted_sum = weighted_sum
        # Weights is a list of input weights
        self.weights = []
        # Defining what layer the Neuron is in
        self.layer = layer
        # Defining the bias
        self.bias = bias

        self.delta = None

        self.index = Neuron.index

        Neuron.index += 1

        Neuron.all_neurons.append(self)

    def __str__(self):
        return f"Neuron ({self.index}) with activation {self.activation} and weights {str(self.weights)} in layer {self.layer.layer_index}"

'''
Weight Class: 
 - weight (Floating Number)
 - bias (Floating Number)
 - Connected Neurons (Input Neuron, Output Neuron)
'''
class Weight:

    index = 0
    all_weights = []

    def __init__(self, weight: float, neurons: tuple):
        # This is the floating point value of the weight
        self.weight = weight
        # A tuple of the input neuron and output neuron
        self.neurons = neurons
        self.index = Weight.index
        Weight.index += 1
        Weight.all_weights.append(self)
    
    def __repr__(self):
        return f"Weight (from={self.neurons[0].index} to={self.neurons[1].index})" #) with value: {self.weight} and bias: {self.bias}"
    
'''
Neural Network Class:
 - Connect it all together
 - Activation Type
 - Cost Calculator
 - Add Layer
 - Activation calculation
 - Forward Propagation
 - Backward Propagation
'''
class NeuralNetwork:

    def __init__(self, activation="sigmoid", loss="L2"):
       self.activation = activation 
       self.loss = loss
       self.layers = []
        
    def add_layer(self, size=1):
        new_layer = Layer(size)

        if new_layer.layer_type != "Input":
            for current_neuron in new_layer.neurons:
                for previous_neuron in self.layers[-1].neurons:
                    prev_layer_size = len(self.layers[-1].neurons)
                    weight = Weight(random.gauss(0, 1 / math.sqrt(prev_layer_size)), (previous_neuron, current_neuron))
                    current_neuron.weights.append(weight)
                        
        for layer in self.layers[1:]:
            layer.layer_type = "Hidden"
        
        self.layers.append(new_layer)

    @staticmethod
    def sigmoid(number: float):
        return 1/(1 + math.exp(-number))
    
    @staticmethod
    def derivativesigmoid(number: float):
        sig = 1/(1 + math.exp(-number))
        return sig * (1 - sig)
    
    def take_input(self, x_input):
        for neuron, x_value in zip(self.layers[0].neurons, x_input):
            neuron.activation = x_value

    def forward_propagation(self, x_input):
        self.take_input(x_input)

        for layer in self.layers[1: ]:
            for neuron in layer.neurons:
                sum = 0
                for weight in neuron.weights:
                    input_neuron = weight.neurons[0]
                    sum += input_neuron.activation * weight.weight
                neuron.weighted_sum = sum + neuron.bias
                activation = self.sigmoid(neuron.weighted_sum)
                neuron.activation = activation

        return [neuron.activation for neuron in self.layers[-1].neurons]
        
    def back_propagation(self, learning_rate=0.1, expected=[], x_input=[]):
        self.forward_propagation(x_input=x_input)
        for layer_index, layer in list(enumerate(reversed(self.layers)))[:-1]:
            
            for neuron_index, neuron in enumerate(layer.neurons):

                reversed_layer_list = list(reversed(self.layers))
                if neuron.layer.layer_type == "Output":
                    neuron.delta = 2 * (neuron.activation - expected[neuron_index]) * self.derivativesigmoid(neuron.weighted_sum)

                else:
                    # Get next layer (closer to output)
                    next_layer = reversed_layer_list[layer_index - 1]
                    sum_deltas = 0
                    for next_neuron in next_layer.neurons:
                        for weight in next_neuron.weights:
                            if weight.neurons[0] == neuron:  # if this neuron connects to next_neuron
                                sum_deltas += weight.weight * next_neuron.delta
                    neuron.delta = sum_deltas * self.derivativesigmoid(neuron.weighted_sum)
                    
                neuron.bias -= learning_rate * neuron.delta

                for weight in neuron.weights:
                    input_neuron = weight.neurons[0]  # source
                    weight.weight -= learning_rate * (neuron.delta * input_neuron.activation)

    
    def train(self, x_values, y_values, learning_rate=0.1, epochs=1):
        for epoch in range(epochs):
            count = 0
            for x_value, y_value in zip(x_values, y_values):
                print(f"{count}/{len(x_values)} complete! | Epoch: {epoch + 1}")
                self.back_propagation(learning_rate=learning_rate, expected=[1 if i == y_value else 0 for i in range(10)], x_input=x_value)
                count += 1

    def predict(self, x_value):
        values = self.forward_propagation(x_value)
        max_value = max(values)
        return values.index(max_value)
    
    def accuracy(self, x_values, y_values):
        count = 0
        for x_value, y_value in zip(x_values, y_values):
            prediction = self.predict(x_value)
            if prediction == y_value:
                count += 1
                print(f"Testing: {count}/{len(x_values)}")  
        print(f"Accuracy: {count/len(x_values)}")

    def save(self, filename):
          with open(filename, "w") as f:
              f.write(f"Layer Count: {len(self.layers)} | Neuron Count: {[len(layer.neurons) for layer in self.layers]}\n")
              for layer in self.layers:
                  f.write(f"Layer {layer.layer_index} | Neuron Count: {len(layer.neurons)}\n")
                  for neuron in layer.neurons:
                      f.write(f"Neuron ({neuron.index}) | Bias: {neuron.bias}\n")
                      for weight in neuron.weights:
                          f.write(f"Weight {{{weight.index}}}: {weight.weight}\n")
    @staticmethod                         
    def get_model(filename):
        # Clear previous state
        Neuron.index = 0
        Neuron.all_neurons.clear()
        Weight.index = 0
        Weight.all_weights.clear()

        with open(filename, "r") as f:
            model = NeuralNetwork()
            lines = f.readlines()

            # Parse neuron counts and add layers
            meta_data = lines[0].split("|")
            neuron_counts = re.findall(r"\d+", meta_data[1])
            for neuron_count in neuron_counts:
                model.add_layer(size=int(neuron_count))

            # Assign biases and weights
            for line in lines[1:]:
                # Neuron bias line
                neuron_match = re.search(r"Neuron \((\d+)\) \| Bias: (-?[\d.]+)", line)
                if neuron_match:
                    neuron_index = int(neuron_match.group(1))
                    neuron_bias = float(neuron_match.group(2))
                    Neuron.all_neurons[neuron_index].bias = neuron_bias

                # Weight value line
                weight_match = re.search(r"Weight \{(\d+)\}: (-?[\d.]+)", line)
                if weight_match:
                    weight_index = int(weight_match.group(1))
                    weight_value = float(weight_match.group(2))
                    Weight.all_weights[weight_index].weight = weight_value

        return model
                
            



    def show(self, filename="neural_network.png", scale=2):
        # Config
        width, height = 800, 800
        neuron_radius = 20 * scale
        layer_spacing = width * scale // (len(self.layers) + 1)
        vertical_padding = 50 * scale
        img = Image.new('RGB', (width * scale, height * scale), 'white')
        draw = ImageDraw.Draw(img)

        # Store neuron centers for drawing connections
        neuron_centers = []

        # Normalize activations for coloring neurons
        all_activations = [neuron.activation for layer in self.layers for neuron in layer.neurons]
        min_act = min(all_activations)
        max_act = max(all_activations)
        act_range = max_act - min_act if max_act != min_act else 1

        def activation_to_red(act):
            norm_act = (act - min_act) / act_range
            min_brightness = 30
            intensity = int(min_brightness + (255 - min_brightness) * norm_act)
            return (intensity, 0, 0)  # Pure red shades

        # Draw neurons
        for li, layer in enumerate(self.layers):
            layer_x = (li + 1) * layer_spacing
            num_neurons = len(layer.neurons)
            total_height = (num_neurons - 1) * 2 * neuron_radius + (num_neurons - 1) * 40 * scale
            layer_y_start = (height * scale - total_height) // 2

            centers = []
            for ni, neuron in enumerate(layer.neurons):
                y = layer_y_start + ni * (2 * neuron_radius + 40 * scale)
                bbox = [
                    layer_x - neuron_radius, y - neuron_radius,
                    layer_x + neuron_radius, y + neuron_radius
                ]
                fill_color = activation_to_red(neuron.activation)
                draw.ellipse(bbox, outline='black', fill=fill_color, width=3 * scale)
                centers.append((layer_x, y))
            neuron_centers.append(centers)

        # Normalize weights for coloring connections
        all_weights = []
        for li in range(1, len(self.layers)):
            for neuron in self.layers[li].neurons:
                for w in neuron.weights:
                    all_weights.append(w.weight)
        max_weight = max(abs(w) for w in all_weights) if all_weights else 1

        def weight_to_color(w):
            norm = abs(w) / max_weight if max_weight else 0
            intensity = int(50 + 205 * norm)
            if w >= 0:
                return (0, 0, intensity)  # Blue for positive weights
            else:
                return (intensity, 0, 0)  # Red for negative weights

        # Draw connections
        for li in range(1, len(self.layers)):
            current_layer = self.layers[li]
            previous_layer_centers = neuron_centers[li - 1]
            current_layer_centers = neuron_centers[li]
            for ni, neuron in enumerate(current_layer.neurons):
                start_points = previous_layer_centers
                end_point = current_layer_centers[ni]
                for wi, weight in enumerate(neuron.weights):
                    start_point = start_points[wi]
                    color = weight_to_color(weight.weight)
                    width_line = max(1, int(1 + 5 * (abs(weight.weight) / max_weight)))
                    draw.line([start_point, end_point], fill=color, width=width_line * scale)

        # Downscale for anti-aliasing
        img = img.resize((width, height), resample=Image.LANCZOS)
        img.save(filename)


    def __str__(self):
        string = ""
        for layer in self.layers:
            string += f"{str(layer)}\n"
        
        return string
    
class DataLoader:
    
    @staticmethod
    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            label_data = f.read()
            return list(label_data)
        
    @staticmethod
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            image_data = f.read()
            images = []
            for i in range(num):
                start = i * rows * cols
                end = start + rows * cols
                image = [pixel / 255.0 for pixel in image_data[start:end]]
                images.append(image)
            return images
        
    @staticmethod
    def preprocess_image(filename):
        img = Image.open(filename).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = 1.0 - img_array  # invert colors

        # Centering
        cy, cx = center_of_mass(img_array)
        shift_y = int(img_array.shape[0] // 2 - cy)
        shift_x = int(img_array.shape[1] // 2 - cx)
        img_array = shift(img_array, shift=(shift_y, shift_x), mode='constant')


        return img_array.flatten().tolist()
        
    @staticmethod
    def visualize_image(img_array):
        # Reshape back to 28x28 for visualization
        img_array = img_array.reshape(28, 28)
        plt.imshow(img_array, cmap='gray')
        plt.show()

if __name__ == "__main__":

    model = NeuralNetwork()

    model.add_layer(size=4)

    model.add_layer(size=3)

    model.add_layer(size=3)

    model.add_layer(size=10)

    model.show()

    # labels = DataLoader.load_labels("./MNIST/train-labels.idx1-ubyte")
    # images = DataLoader.load_images("./MNIST/train-images.idx3-ubyte")


    # images_test = DataLoader.load_images("./MNIST/t10k-images.idx3-ubyte")
    # labels_test = DataLoader.load_labels("./MNIST/t10k-labels.idx1-ubyte")


    # small_images = images[:30000]
    # small_labels = labels[:30000]

    # model.train(small_images, small_labels, epochs=20)

    # # model.save("neuralmodel.txt")

    # model.accuracy(images_test, labels_test)   


