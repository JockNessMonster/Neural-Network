
'''
Allow custom activation | Depending on the activation it will determine which weight initialisation to do
Allow custom kernel size, e.g. 5 x 5 x 1, allow the amount of kernels in each convolution to be edited
Allow custom pooling, e.g. max pooling, min pooling, averrage pooling, as well as pooling size, and make the stride the value of the width of the pool kernel, but can be changed if specified.

'''

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



class TesterFunctions:

    @staticmethod
    def show_image_grid(array):
        plt.imshow(array, cmap="gist_gray", interpolation=None)
        plt.colorbar()
        plt.title("Image Array")

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                plt.text(j, i, str(array[i, j]), fontsize=6, va="center", ha="center", color="blue")

        plt.show()

    @staticmethod
    def show_image(array):
        plt.imshow(array, cmap="gist_gray", interpolation=None)
        plt.colorbar()
        plt.title("Image")

        plt.show()

class Kernel:

    def __init__(self, size: tuple=(3, 3), stride=1):
        self.kernel = np.random.uniform(low=-3, high=3, size=size)

        print(self.kernel)
        
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

        self.kernels = np.random.uniform(low=-3, high=3, size=(filters, *filter_size))

    def print_kernels(self):

        kernel_representation = ""

        for kernel_index, kernel in enumerate(self.kernels):
            kernel_representation += f"Kernel {kernel_index + 1}\n\n"

    
        print(kernel_representation)

        





# image = Image.open("./MNIST_FASHION/sydney.jpg")

# image = image.resize((9, 9))

# img_array = np.array(image)

layer_1 = ConvLayer(filters=2)

layer_1.print_kernels()


# TesterFunctions.show_image_grid(img_array)













