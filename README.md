# Neural-Network

This is my file for a custom built Neural Network.

I built two Neural Networks: - Firstly: I built a Neural Network using an pythonic Object Orientated approach with objects for the Layers, Weights, Neurons and Neural Network. This was to develop my understanding of the Neural Network

    - Secondly: I build a Neural Network using numpy as well as objects for Layer's and Neural Networks.

These Neural Networks use Sigmoid Activation and L2 Loss with Xavier weight initialisation.

Instructions:

    - To create a Neural Network | model_name = NeuralNetwork()

    - To add Layers | model_name.add_layer(size=n) | This adds one layer to the neural network

    - To train the model | model_name.train(x_inputs, y_labels, epochs=n)

    - To save the model | model_name.save(filename)

    - To load the model | model.load_model(filename)

    - To predict a certain image | model.predict(x_value, individual=True)
