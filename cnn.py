
'''
Allow custom activation | Depending on the activation it will determine which weight initialisation to do
Allow custom kernel size, e.g. 5 x 5 x 1, allow the amount of kernels in each convolution to be edited
Allow custom pooling, e.g. max pooling, min pooling, averrage pooling, as well as pooling size, and make the stride the value of the width of the pool kernel, but can be changed if specified.

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
                plt.text(j, i, str(array[i, j]), fontsize=8, va="center", ha="center", color="red")

        plt.show()

    @staticmethod
    def show_image(array):
        plt.imshow(array, cmap="gist_gray", interpolation=None)
        plt.colorbar()
        plt.title("Image")

        plt.show()


class Activations:

    @staticmethod
    def sigmoid(number: float):
        return 1/(1 + math.exp(-number))
    
    @staticmethod
    def tanh(number: float):
        return (math.exp(number) - math.exp(-number)) / (math.exp(number) + math.exp(-number))
    
    @staticmethod
    def ReLU(number: float):
        return max(0, number)
    
    @staticmethod
    def softmax(array):
        sum = np.sum(np.exp(array))
        return np.exp(array) / sum


class Kernel:

    def __init__(self, size: tuple=(3, 3), stride=1):
        self.kernel = np.random.uniform(low=-3, high=3, size=size)
        
        self.stride = stride
        
    def convolute(self, img):

        # Gets the original images dimensions
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Gets the kernels dimesions
        kernel_height = self.kernel.shape[0]
        kernel_width = self.kernel.shape[1]

        # Gets the output's dimensions
        height_output = int(((img_height - kernel_height) / self.stride) + 1)
        width_output = int(((img_width - kernel_width) / self.stride) + 1)

        # Sets the output img array with zeroes
        output_img = np.zeros((height_output, width_output))

            
        for height in range(height_output):
            for width in range(width_output):
                patch = img[height * self.stride : height * self.stride + kernel_height, width * self.stride : width * self.stride + kernel_width]

                output_value = np.sum(patch * self.kernel)

                output_img[height, width] = output_value

        TesterFunctions.show_image_grid(output_img)

class ConvLayer:

    def __init__(self, activation="sigmoid", filters=1, filter_size=(3, 3, 1), stride=1):
        self.activation = activation
        self.filters =  filters
        self.filter_size = filter_size
        self.stride = stride

        # Change the initialised kernel values depending on the activation
        self.kernels = np.random.uniform(low=-3, high=3, size=(filters, filter_size[2], filter_size[0], filter_size[1]))

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

        image = ConvLayer.to_chw(image)

        print(f"Image Shape: {image.shape}")
        print(f"Kernel Shape: {self.kernels.shape}\n\n\n")

        if (self.kernels.shape[1] != image.shape[1]):
            raise ValueError(f"Layers in Image ({image.shape[1]}) has to equal amount of Filters in Kernel ({self.kernels.shape[1]})")
            
        
        img_height = image.shape[2]
        img_width = image.shape[3]

        kernel_height = self.kernels.shape[2]
        kernel_width = self.kernels.shape[3]

        output_height = int(((img_height - kernel_height) / self.stride) + 1)
        output_width = int(((img_width - kernel_width) / self.stride) + 1)

        outputs = np.zeros(shape=(self.kernels.shape[0], self.kernels.shape[1], output_height, output_width))

        print(outputs)

        # for kernel_count_index, kernel_count in enumerate(outputs):
        #     print(kernel_count)

        return outputs


    @staticmethod
    def print_forward(output):

        '''
        Visual Representation of the outputted values
        '''

        kernel_representation = ""

        for kernel_index, kernel in enumerate(output):
            kernel_representation += "-" * 40 + "\n"
            kernel_representation += f"Kernel {kernel_index + 1}\n\n"
            for filter_index, filter in enumerate(kernel):
                kernel_representation += f"Filter {filter_index + 1} in Kernel {kernel_index + 1}\n\n{filter}\n\n"
            
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

        





image = Image.open("./MNIST_FASHION/sydney.jpg")

image = image.resize((9, 9))

img_array = np.array(image)

layer_1 = ConvLayer(filters=1, filter_size=(3, 3, 3))

output = layer_1.forward(img_array)




# TesterFunctions.show_image_grid(img_array)












