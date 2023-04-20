import random
import math

# Neuron Class
class Neuron:
    def __init__(self):
        self.collector = 0.0        # value of neuron as float
        self.connections = []       # list of connections
        self.weights = []           # list of weights
        self.delta = 0.0            # delta value of neuron as float
        self.activation_error = 0.0 

# Read File for Layer Sizes
def read_file(file):
    with open(file, 'r') as f:      # open file
        temp = f.read().split(',')  # seperate by commas
        for num in temp:            # for each number
            network_layers.append(int(num.strip())) # add number as int to network_layers array

# Read File for Input Values
def read_input(file):
    with open(file, 'r') as f:  # open file
        lines = f.readlines()      # read lines
        for line in lines:  # for each line
            temp = line.split(',')  # seperate by commas   
            row = [float(num.strip()) for num in temp]
            inputs.append(row)          # add row to inputs array

# Make Input Layer of Network
def make_input_layer(network, layers, input):
    neurons = []                        # create layer to add to network
    for i in range(layers[0]):          # for each neuron in layer
        neuron = Neuron()               # create neuron
        neuron.collector = input[i]     # set input to neuron.collector
        neurons.append(neuron)          # add neuron to layer
    network.append(neurons)             # add input layer to network

# Make Remaining Layers of Network
def make_hidden_layers(network, layers):
    layers = layers[1:]                 # disclude input layer
    for layer in layers:                # for each layer in remaining layers
        new_layer = []                  # create layer to add to network
        for i in range(layer):          # for each neuron in layer
            neuron = Neuron()           # create neuron
            new_layer.append(neuron)    # add neuron to layer
        network.append(new_layer)       # add layer to network

# Make Connections Between Layers
def make_connections(network):
    # for each neuron in network 
    for layer in network:          
        for neuron in layer:       
            try:   # to make connection (will fail on last layer)
                # add next layer to neuron connections
                neuron.connections.append(network[network.index(layer)+1])

                # setting weights along with connection
                for i in range(len(network[network.index(layer)+1])):
                    neuron.weights.append(round(random.uniform(-1,1), 2))
                weights_network.append(neuron.weights)
            except: # pass on last layer
                print('except - make connections')

# Set Collectors of Neurons
def set_collectors(network):
    # for each connection in network, add value of previous neuron to connected neuron
    for layer in network:
        for neuron in layer:        
            for connection in neuron.connections: 
                for connected_neuron in connection:
                    connected_neuron.collector += neuron.collector
                    
# Print Layers of Network
def print_layers(network):
    # for each layer in network
    for layers in network:
        for neuron in layers:
            print(neuron.collector, end=' ')    # print value of neuron, end with space not newline
        print()   
        
# Print Weights of Network
def print_weights(network):
    # for each layer in network
    for layers in network:
        for neuron in layers:
            print(neuron.weights, end=' ')    # print weight of neuron, end with space not newline            
        print()

# Print connections of network
def print_connections(network):
    # for each layer in network
    for layers in network:
        for neuron in layers:
            for connection in neuron.connections:
                print(connection, end=' ')
            print(neuron.weights, end=' ')
            print()          
        print()

# Train Network
def train(network, train_data, lr, n_epochs, target_error):
    num_outputs = network_layers[-1]
    epoch_list = []
    for epoch_num, epoch in enumerate(range(n_epochs), 1):
        sum_error = 0
        for row in train_data:
            outputs = forward_prop(network, row)
            expected = [0 for i in range(num_outputs)]
            expected = [row[-1]]

            error = sum((expected[i]-outputs[i])**2 for i in range(len(expected)))
            sum_error += error
            backward_prop(network, expected)
            update_weights(network, row, lr)


        if sum_error <= target_error:
            epoch_list.append('--->epoch=%d, lr=%.2f, error=%.3f' % (epoch_num, lr, sum_error, ))
            for epoch in epoch_list:
                print(epoch)
            print('target error reached=%.3f' % sum_error)
            return
        
        print(f'outputs {outputs}')

        epoch_list.append('>epoch=%d, lr=%.2f, error=%.3f' % (epoch_num, lr, sum_error))
    for epoch in epoch_list:
        print(epoch)

# Forward Propagation
def forward_prop(network, row):
    inputs = row
    for i in range(len(network)-1):
        new_inputs = []
        for j in range(len(network[i+1])):
            neuron = network[i+1][j]
            activation = 0.0
            for k in range(len(network[i])):
                prev_neuron = network[i][k]
                activation += inputs[k] * prev_neuron.weights[j]

            print(f'activation {activation}')
            neuron.collector = transfer(activation)
            print(f'ACTIVATED {neuron.collector}')
            new_inputs.append(neuron.collector)
            

            # for connection in neuron.connections:
            #     for connected_neuron in connection:


            # activation = activate(neuron.weights, inputs)
            # print(f'activation {activation}')
            # neuron.connections[i] = transfer(activation)
            # print(f'ACTIVATED {neuron.collector}')
            # new_inputs.append(neuron.collector)

            # if new_inputs != [] : inputs = new_inputs
            # else:
            #     for j in range(len(network[-2])):
            #         neuron.collector += network[-2][j].collector * network[-2][j].weights[i]
            #     neuron.collector = transfer(neuron.collector)
            #     new_inputs.append(neuron.collector)
        inputs = new_inputs
        print(f'inputs {inputs}')
    return inputs

# Backward Propagation
def backward_prop(network, expected):
    for i in reversed(range(len(network))):
 #       print(f'layer {i}')
        errors = []
        if i != len(network)-1: # 3-1 not last layer
            for j in range(len(network[i])):
                neuron = network[i][j]
                weighted_sum = 0.0
                for k in range(len(network[i+1])):
                    prev_neuron = network[i+1][k]
                    weighted_delta = neuron.weights[k] * prev_neuron.delta
                    weighted_sum += weighted_delta
          #      print(f'weighted sum {weighted_sum}')
    #         print(f'transfer derivative {transfer_derivative(neuron.collector)}')
                errors.append(weighted_sum)
                neuron.delta = (weighted_sum * transfer_derivative(neuron.collector))
        else: # last layer
            for j in range(len(network[i])):
                neuron = network[i][j]
                weighted_sum = neuron.collector - expected[j]
                errors.append(weighted_sum)
                neuron.delta = (errors[j] * transfer_derivative(neuron.collector))
        
        # for layer in network:
        #     for neuron in layer:
        #         print(neuron.delta, end=' ')
        #     print()


    # for i in reversed(range(len(network))):
    #     if i != len(network)-1: # 3-1 not last layer
    #         for j in range(len(network[i])):
    #             neuron = network[i][j]
    #             print(f'neuron {neuron.activation_error}')
    #             for k in range(len(network[i-1])):
    #                 prev_neuron = network[i-1][k]
    #                 prev_neuron.activation_error = neuron.weight_delta * neuron.activation_error * transfer_derivative(prev_neuron.collector)
    #                 prev_neuron.weight_delta = prev_neuron.activation_error * prev_neuron.collector

    #     else: # last layer
    #         for j in range(len(network[i])):
    #             neuron = network[i][j]
    #             for k in range(len(network[i-1])):
    #                 prev_neuron = network[i-1][k]
    #                 prev_neuron.activation_error = neuron.collector - expected[j]
    #                 prev_neuron.weight_delta = prev_neuron.activation_error * prev_neuron.collector

# Update Weights
def update_weights(network, row, lr):
    for i, layer in enumerate(network):
        if i != len(network)-1: # not last layer
 #       print(f'layer {i}')
            inputs = []
            for neuron in network[i]:
                inputs.append(neuron.collector)

            print(inputs)
            print()

            for j in range(len(network[i])):
                neuron = network[i][j]
                for k in range(len(network[i+1])):
  #                  print(f'neuron weights {j} {neuron.weights} delta {neuron.delta} collector {inputs[j]}')
                    neuron.weights[k] -= lr * neuron.delta * inputs[j]
  #                  print(f'neuron weights {j} {neuron.weights} delta {neuron.delta} collector {inputs[j]}')

# Activate Neuron
def activate(neuron_weights, inputs):
    activation = 0.0
    print(f'activation inputs {inputs}')
    print(f'activation weights {neuron_weights}')
    for i in range(len(neuron_weights)):
        activation += neuron_weights[i] * inputs[i]
        print(f'activation sum {activation}')
    return activation


    # for connection in neuron.connections:
    #     for i, connected_neuron in enumerate(connection):   
    #         connection[i].collector += neuron.collector * neuron.weights[i]
    #     connection[i].collector = transfer(connection[i].collector)

# Activation Function (Sigmoid)
def transfer(collector):
    return 1 / (1 + math.exp(-collector))

# Derivative of Activation Function
def transfer_derivative(collector):
    return (collector) * (1.0 - (collector))

# Create Network
def new_network(row):
    # new network
    # make layers
    network = []
    make_input_layer(network, network_layers, row)
    make_hidden_layers(network, network_layers)
    # make connections
    make_connections(network)
    return network

### Main
if __name__ == '__main__':
    # declare arrays
    neural_network = []        # main network
    network_layers = [] # size of each layer
    inputs = []         # input values for input layer
    weights_network = []    # weights for each neuron

    # read file for layer sizes and input values
    read_file('layers')
    read_input('inputs')
    
    # create network
    neural_network = new_network(inputs[0])
    print_weights(neural_network)
    # train network
    train(neural_network, inputs, lr = 0.4, n_epochs = 1, target_error = 0.05)

    print_weights(neural_network)
    # for layer in neural_network:
    #     for neuron in layer:
    #         print(neuron.delta, end=' ')
    #     print()
    # for layer in neural_network:
    #     for neuron in layer:
    #         print(neuron.collector, end=' ')
    #     print()