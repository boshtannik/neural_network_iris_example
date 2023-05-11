from math import exp
import os
from typing import List
from random import randint, choice
import json


PERCEPTRON_SAVE_FILE = 'perceptron.json'
IRIS_FILE = "./iris.csv"


class Connection:
    def __init__(self, from_neuron, to_neuron, weight):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron

        self.weight = weight

        to_neuron.input_connections.append(self)
        from_neuron.output_connections.append(self)

    def __repr__(self):
        return f'Connection({self.from_neuron}, {self.to_neuron}, {self.weight})'

    def get_weighted_value(self):
        return self.from_neuron.output * self.weight

    def forward(self):
        self.to_neuron.input += self.get_weighted_value()

    def back_propagate(self):
        self.from_neuron.error += self.to_neuron.error * self.weight

    def adjust_weight(self, learning_rate: float = 1.0):
        self.weight += (learning_rate * self.from_neuron.output * self.to_neuron.get_error_derivative())
        

class Neuron:
    def __init__(self, is_bias: bool = False):
        self.error = 0
        self.input = 0
        self.output = 1 if is_bias else 0

        self.is_bias = is_bias

        self.output_connections = []
        self.input_connections = []

    def activation_function(self, x):
        if x < 0:
            return 1 - 1 / (1 + exp(x))
        return 1 / (1 + exp(-x))

    def derivative_activation_function(self, x):
        return x * (1 - x)

    def activate(self):
        self.output = self.activation_function(self.input)

    def get_error(self, expected_value: float) -> float:
        self.error = expected_value - self.output
        return self.error

    def get_error_derivative(self) -> float:
        return self.error * self.derivative_activation_function(self.output)


class PerceptronSaver:
    def __init__(self, perceptron: 'Perceptron') -> None:
        self.perceptron = perceptron

    def save_to_file(self, filename: str):
        # Get configuration of layers.
        # Get configuration of biases.
        # Iterate over layers -> neurons -> connections -> save connections to file.
        layers_configuration = []
        for layer in self.perceptron.layers:
            layers_configuration.append( sum( 1 for n in layer if not n.is_bias ) )

        biases_configuration = []
        for layer in self.perceptron.layers:
            biases_configuration.append( sum( 1 for n in layer if n.is_bias ) )

        weights_configuration = []
        for layer in self.perceptron.layers:
            for neuron in layer:
                for connection in neuron.output_connections:
                    weights_configuration.append(connection.weight)

        object_to_save = {
            "layers_configuration": layers_configuration,
            "biases_configuration": biases_configuration,
            "weights_configuration": weights_configuration
        }

        with open(filename, "w") as f:
            json.dump(object_to_save, f)

    def load_from_file(self, filename: str):
        # Load configuration of layers -> instantiate it.
        # Load configuration of biases -> instantiate them.
        # Iterate over existing weights, being created by NN -> set each weight, loaded from file.
        with open(filename, "r") as f:
            load_object = json.load(f)

        layers_configuration = load_object.get("layers_configuration")
        biases_configuration = load_object.get("biases_configuration")
        weights_configuration = load_object.get("weights_configuration")

        self.perceptron.build(layers_configuration)

        for layer_index, biases_count in enumerate(biases_configuration):
            for _ in range(biases_count):
                self.perceptron.add_bias(layer_number=layer_index)

        for layer in self.perceptron.layers:
            for neuron in layer:
                for connection in neuron.output_connections:
                    connection.weight = weights_configuration.pop(0)


class Perceptron:
    def __init__(self, layers: List[int]):
        """
        Param layers - is list of counts of neurons in each layer
        """
        self.layers = []
        self.build(layers)

        self.saver = PerceptronSaver(self)

    def build(self, layers: List[int]):
        self.layers = []
        for layer in layers:
            self.layers.append([Neuron() for _ in range(layer)])

        for i in range(len(self.layers) - 1):
            for from_neuron in self.layers[i]:
                for to_neuron in self.layers[i + 1]:
                    Connection(from_neuron, to_neuron, randint(-100, 100) * 0.01)

        # hidden layers are all except input and output
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def save(self):
        self.saver.save_to_file(PERCEPTRON_SAVE_FILE)

    def load(self):
        self.saver.load_from_file(PERCEPTRON_SAVE_FILE)

    def add_bias(self, layer_number: int):
        """
        Param layer_number - number of layer. Count starts from 0

        Adds special neuron, that is in previous layer,
        and has connections to the next layer - which it should affect to.
        """

        if layer_number == len(self.layers) - 1:
            raise ValueError(f"Bias can't be added to output layer. Because there is no next layer to affect to.")

        if layer_number < 0 or layer_number > len(self.layers) - 1:
            raise ValueError(f"Number of layer should be in range between 0 and {len(self.layers) - 2}")

        bias_neuron = Neuron(is_bias=True)

        current_layer = self.layers[layer_number]
        next_layer = self.layers[layer_number + 1]

        current_layer.append(bias_neuron)

        for neuron in next_layer:
            Connection(from_neuron=bias_neuron, to_neuron=neuron, weight=randint(-100, 100) * 0.01)

    def predict(self, inputs: List[float]) -> List[float]:
        if len(inputs) != len([n for n in self.input_layer if not n.is_bias]):
            raise ValueError('Number of inputs must be equal to number of input neurons')

        # Set the output of the input layer to the input values
        for i in range(len(inputs)):
            if self.input_layer[i].is_bias:
                continue
            self.input_layer[i].output = inputs[i]

        # Propagate the values through the network
        for i in range(len(self.layers) - 1):
            # current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            for next_neuron in next_layer:
                next_neuron.input = 0
                for connection in next_neuron.input_connections:
                    connection.forward()
                next_neuron.activate()

        # Return the output values of the output layer
        return [neuron.output for neuron in self.output_layer]

    def get_errors(self, expected_values: List[float]) -> List[float]:
        if len(expected_values) != len([n for n in self.output_layer if not n.is_bias]):
            raise ValueError('Number of expected values must be equal to number of output neurons')

        errors = []
        for i in range(len(expected_values)):
            errors.append(self.output_layer[i].get_error(expected_values[i]))
        return errors
    
    def train(self, inputs: List[float], expected_values: List[float], learning_rate: float):
        if len(inputs) != len([n for n in self.input_layer if not n.is_bias]):
            raise ValueError('Number of inputs must be equal to size of input layer')

        if len(expected_values) != len(self.output_layer):
            raise ValueError('Number of expected_values must be equal to size of output layer')
        
        # Forward propagation
        self.predict(inputs)
        
        # Calculate the error for each output neuron
        errors = self.get_errors(expected_values)
        total_error = sum(errors)

        # 1 Clean all neuron errors
        for layer in self.layers:
            for neuron in layer:
                neuron.error = 0

        # 2 Set errors to last layer
        for neuron in self.output_layer:
            neuron.error = errors.pop(0)

        # 3 Iterate trough layers from last to previous ones, in order to:
        #   1 - propagate error back.
        #   2 - adjust weights accordingly to propagated error.
        # For every income connection of neuron - call:
        #       1 back_propagate() -> Should: get error of current neuron,
        #                             multiply it by weight of connection, and add
        #                             result value to error of previous neuron
        #       2 adjust_weight()  -> Should: add to weight result of (learning_rate * connection.from_neuron.output * connection.to_neuron.get_error_derivative)

        # Backpropagation
        for layer in reversed(self.layers):
            for neuron in layer:
                for input_connection in neuron.input_connections:
                    input_connection.back_propagate()
                    input_connection.adjust_weight(learning_rate=learning_rate)
        
        # Print the average error for this epoch
        print(f'Average error: {total_error/len(inputs)}')


def generate_expected_data(class_index: int, classes_count: int) -> List[float]:
    result = []
    for i in range(classes_count):
        result.append(0.99 if class_index == i else 0.01)
    return result


def test_neural_network():
    nn = Perceptron(layers=[4, 4, 3])
    for i in range(nn.layers.__len__() - 1):
        nn.add_bias(layer_number=i)

    # Prepare data in order (<Data to be feed in>, <Expected data to be received>)
    if os.path.exists(PERCEPTRON_SAVE_FILE):
        nn.load()
        # Set data for training.
        with open(IRIS_FILE, "r") as f:
            all_lines = f.readlines()

        all_lines = [l.replace('\n', '') for l in all_lines]
        all_lines = [l.replace('"', '') for l in all_lines]
        all_lines = [l.split(',') for l in all_lines]
        all_lines = [l for l in all_lines]
        all_lines.pop(0) # Pops out the header text.

        # Determine all available classes of data.
        all_classes = {}
        for l in all_lines:
            name = l[-1]
            if name in all_classes:
                continue
            all_classes[name] = len(all_classes)

        # Fill all data into classified dataset (Dict).
        train_data = {}
        for l in all_lines:
            name = l[-1]
            index = all_classes[name]
            if index not in train_data:
                train_data[index] = list()

            sepal_length, sepal_width, petal_length, petal_width = l[0], l[1], l[2], l[3]

            train_data[index].append(
                (
                    float(sepal_length),
                    float(sepal_width),
                    float(petal_length),
                    float(petal_width)
                )
            )

        # Grab some data to test results.
        test_data = {}
        for each_class in all_classes.values():
            test_data[each_class] = list()

        for each_class_index, each_class_values in train_data.items():
            test_data[each_class_index].append(each_class_values.pop(randint(0, len(each_class_values) - 1)))

        for test_data_index, test_data_list in test_data.items():
            expected_data = generate_expected_data(class_index=test_data_index, classes_count=len(all_classes))
            for test_data_set in test_data_list:
                results = nn.predict(inputs=test_data_set)
                print('expected_data', expected_data)
                print('results', results)
                print()
    else:
        # Set data for training.
        with open(IRIS_FILE, "r") as f:
            all_lines = f.readlines()

        all_lines = [l.replace('\n', '') for l in all_lines]
        all_lines = [l.replace('"', '') for l in all_lines]
        all_lines = [l.split(',') for l in all_lines]
        all_lines = [l for l in all_lines]
        all_lines.pop(0) # Pops out the header text.

        # Determine all available classes of data.
        all_classes = {}
        for l in all_lines:
            name = l[-1]
            if name in all_classes:
                continue
            all_classes[name] = len(all_classes)

        # Fill all data into classified dataset (Dict).
        train_data = {}
        for l in all_lines:
            name = l[-1]
            index = all_classes[name]
            if index not in train_data:
                train_data[index] = list()

            sepal_length, sepal_width, petal_length, petal_width = l[0], l[1], l[2], l[3]

            train_data[index].append(
                (
                    float(sepal_length),
                    float(sepal_width),
                    float(petal_length),
                    float(petal_width)
                )
            )

        # Grab some data to test results.
        test_data = {}
        for each_class in all_classes.values():
            test_data[each_class] = list()

        for each_class_index, each_class_values in train_data.items():
            test_data[each_class_index].append(each_class_values.pop(randint(0, len(each_class_values) - 1)))

        print("All classes", all_classes)
        print('all_data', train_data)
        print('test_data', test_data)

        train_generations = 10000
        for _ in range(train_generations):
            for train_data_index, train_data_list in train_data.items():
                for inputs in train_data_list:
                    nn.train(
                        inputs=inputs,
                        expected_values=generate_expected_data(
                            class_index=train_data_index,
                            classes_count=len(all_classes)
                        ),
                        learning_rate=0.1
                    )

        for test_data_index, test_data_list in test_data.items():
            expected_data = generate_expected_data(class_index=test_data_index, classes_count=len(all_classes))
            for test_data_set in test_data_list:
                results = nn.predict(inputs=test_data_set)
                print('expected_data', expected_data)
                print('results', results)
                print()

    # Test the neural network
    if not os.path.exists(PERCEPTRON_SAVE_FILE):
        nn.save()


if __name__ == '__main__':
    test_neural_network()
