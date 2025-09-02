
'''
Allow custom activation | Depending on the activation it will determine which weight initialisation to do
Allow custom kernel size, e.g. 5 x 5 x 1, allow the amount of kernels in each convolution to be edited
Allow custom pooling, e.g. max pooling, min pooling, average pooling, as well as pooling size, and make the stride the value of the width of the pool kernel, but can be changed if specified.

'''

'''
To do:

    - Add initialisation based on the activation
    - 


'''

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math


class TesterFunctions:

    @staticmethod
    def show_image_grid(array):
        plt.imshow(array, cmap="gist_gray", interpolation=None)
        plt.colorbar()
        plt.title("Image Array")

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                plt.text(j, i, str(array[i, j]), fontsize=6, va="center", ha="center", color="red")

        plt.show()

    @staticmethod
    def show_image(array):
        plt.imshow(array, cmap="gist_gray", interpolation=None)
        plt.colorbar()
        plt.title("Image")

        plt.show()


class Activations:

    @staticmethod
    def sigmoid(x):
        # Works on both single numbers and numpy arrays
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        # NumPy has a built-in tanh
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        # Element-wise max
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x, axis=None):
        # Subtract max for numerical stability
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


class ConvLayer:

    def __init__(self, activation="sigmoid", filters=1, filter_size=(3, 3, 1), stride=1):
        self.activation = activation
        self.filters =  filters
        self.filter_size = filter_size
        self.stride = stride

        # Change the initialised kernel values depending on the activation
        # self.kernels = np.random.uniform(low=-3, high=3, size=(filters, filter_size[2], filter_size[0], filter_size[1]))

        if activation.lower() == "relu":
            std = np.sqrt(2.0 / np.prod(filter_size[:2]))
            self.kernels = np.random.randn(filters, filter_size[2], filter_size[0], filter_size[1]) * std
        elif activation.lower() == "tanh":
            std = np.sqrt(1.0 / np.prod(filter_size[:2]))
            self.kernels = np.random.randn(filters, filter_size[2], filter_size[0], filter_size[1]) * std
        else:  # sigmoid or others
            limit = np.sqrt(6 / np.prod(filter_size[:2]))
            self.kernels = np.random.uniform(-limit, limit, size=(filters, filter_size[2], filter_size[0], filter_size[1]))


        self.biases = np.zeros(shape=(filters))

    @staticmethod
    def to_chw(image: np.ndarray) -> np.ndarray:

        if image.ndim == 2:
            return np.expand_dims(image, axis=(0, 1))
        
        elif image.ndim == 3:
            h, w, c = image.shape

            if c in (1, 3, 4):
                new_image = image.transpose(2, 0, 1)
                return np.expand_dims(new_image, axis=0)

            elif h in (1, 3, 4):
                return np.expand_dims(image, axis=0)
            
        elif image.ndim == 4:
            return image
            
        else:
            raise ValueError(f"Unexpected shape {image.shape}, can't decide format")



    def forward(self, image):

        '''
        Will be a 2d array, in the first dimension it will be dependent on what kernel it is, in the second dimesion it will be what filter inside the individual kernel it is
        E.g. 3 kernels (3 x 3 x 2) [[image1 image2] [image1 image2] [image1 image2]]

        The depth of the neural network is the amount of channels in the input, e.g. if its a black and white it will be (3 x 3 x 1) however if its RGB (3 x 3 x 3) or later in the neural network.

        img_shape = (amount of images, amount of colours in filter, height, width)
        kernel_shape (amount of kernels, amount of filters in each kernel, height, width)

        amount of filters in each kernel has to equal to amount of colours in filter, otherwise doesn't work
        
        '''

        activations = {"sigmoid": Activations.sigmoid, "tanh": Activations.tanh, "relu": Activations.relu, "softmax": Activations.softmax}

        image = ConvLayer.to_chw(image)

        if (self.kernels.shape[1] != image.shape[1]):
            raise ValueError(f"Layers in Image ({image.shape[1]}) has to equal amount of Filters in Kernel ({self.kernels.shape[1]})")
            
        
        img_height = image.shape[2]
        img_width = image.shape[3]

        kernel_height = self.kernels.shape[2]
        kernel_width = self.kernels.shape[3]

        output_height = int(((img_height - kernel_height) / self.stride) + 1)
        output_width = int(((img_width - kernel_width) / self.stride) + 1)

        outputs = np.zeros(shape=(self.kernels.shape[0], output_height, output_width))

        # Goes over every kernel
        for kernel in range(self.kernels.shape[0]):
            # Goes over every filter
            for filter in range(self.kernels.shape[1]):
                for height in range(output_height):
                    for width in range(output_width):
                        patch = image[0, filter, height * self.stride : height * self.stride + kernel_height, width * self.stride : width * self.stride + kernel_width]
                        
                        outputs[kernel, height, width] += np.sum(patch * self.kernels[kernel, filter])

            outputs[kernel] = activations[self.activation.lower()](outputs[kernel] + self.biases[kernel])

                


        
                # To find image that goes with the filter 
                # print(image[0, filter])

        return np.expand_dims(outputs, axis=0)


    @staticmethod
    def print_forward(output):

        '''
        Visual Representation of the outputted values
        '''

        kernel_representation = ""

        for kernel_index, kernel in enumerate(output):
            kernel_representation += "-" * 40 + "\n"
            kernel_representation += f"Kernel {kernel_index + 1}\n\n{kernel}"
            
        print(kernel_representation)




    def __str__(self):

        '''
        Visual Representation of the kernels in a convolutional layer
        '''

        kernel_representation = ""


        for kernel_index, kernel in enumerate(self.kernels):
            kernel_representation += "-" * 40 + "\n"
            kernel_representation += f"Kernel {kernel_index + 1}\n\n"
            for filter_index, filter in enumerate(kernel):
                kernel_representation += f"Filter {filter_index + 1} in Kernel {kernel_index + 1}\n\n{filter}\n\n"
            
        return kernel_representation
    


class PoolingLayer:
    def __init__(self, size=(2, 2), stride=2, type="max"):

        self.size = size

        self.stride = stride if stride is not None else size[0]

        self.type = type

    def pool(self, image):

        image = ConvLayer.to_chw(image)

        batch_size, filters, image_height, image_width = image.shape

        pool_height, pool_width = self.size

        output_height = int(math.floor(((image_height - self.size[0]) / self.stride) + 1))

        output_width = int(math.floor((((image_width - self.size[1]) / self.stride) + 1)))

        outputs = np.zeros(shape=(batch_size, filters, output_height, output_width))

        print(outputs.shape)

        for b in range(batch_size):
            for f in range(filters):
                for h in range(output_height):
                    for w in range(output_width):
                        patch = image[b, f, h * self.stride : h * self.stride + pool_height, w * self.stride : w * self.stride + pool_width]

                        if (self.type.lower() == "max"):
                            outputs[b, f, h, w] = np.max(patch)
                        elif (self.type.lower() == "average"):
                            outputs[b, f, h, w] = np.mean(patch)
                        elif (self.type.lower() == "min"):
                            outputs[b, f, h, w] = np.min(patch)
                        else:
                            raise ValueError(f"Unknown pooling type: {self.type}")
        
        return outputs

        



    


        





image = Image.open("./MNIST_FASHION/sydney.jpg")

image = image.resize((9, 9))

img_array = np.array(image)

img_array = img_array / 255.0

layer_1 = ConvLayer(filters=3, filter_size=(3, 3, 3))

layer_2 = PoolingLayer()

output = layer_1.forward(img_array)


pooled = layer_2.pool(output)




















