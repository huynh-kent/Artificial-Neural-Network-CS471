import random
import math

# Neuron Class
class Neuron:
    def __init__(self):
        self.collector = 0.0        # value of neuron as float
        self.connections = []       # list of connections
        self.weights = []           # list of weights
        self.delta = 0.0            # delta value of neuron as float

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
            
        epoch_list.append('>epoch=%d, lr=%.2f, error=%.3f' % (epoch_num, lr, sum_error))
    for epoch in epoch_list:
        print(epoch)

# Forward Propagation
def forward_prop(network, row):
    inputs = row
    for layers in network:
        new_inputs = []
        for i, neuron in enumerate(layers):
            for connection in (neuron.connections):
                for connected_neuron in connection:
                    activate(neuron)
                new_inputs.append(connected_neuron.collector)

            if new_inputs != [] : inputs = new_inputs
            else:
                for j in range(len(network[-2])):
                    neuron.collector += network[-2][j].collector * network[-2][j].weights[i]
                neuron.collector = transfer(neuron.collector)
                new_inputs.append(neuron.collector)
                inputs = new_inputs
    return inputs

# Backward Propagation
def backward_prop(network, expected):
    for i in reversed(range(len(network))):
 #       print(f'layer {i}')
        errors = []
        if i != len(network)-1: # 3-1 not last layer
            for j in range(len(network[i])):
                for k in range(len(network[i][j].weights)):
                    weighted_delta = (network[i][j].weights[k] * network[i+1][k].delta)
                    weighted_sum += weighted_delta
                    #print(f'weighted delta {weighted_delta}')
        #            print(f'neuron {network[i][j]}')
        #            print(f'weight {network[i][j].weights[k]}')
                errors.append(weighted_sum)
        else: # last layer
            for j in range(len(network[i])):
                neuron = network[i][j]
                weighted_sum = (neuron.collector - expected[j])
                errors.append(weighted_sum)
        # calculate delta
        for j in range(len(network[i])):
            neuron = network[i][j]
            neuron.delta = (errors[j] * transfer_derivative(neuron.collector))

# Update Weights
def update_weights(network, row, lr):
    print(f'row')
    for i, layer in enumerate(network):
        print(f'layer {i}')
        collectors = []
        if i != 0: collectors = [prev_neuron.collector for prev_neuron in network[i-1]]
        else: collectors = row[:-1]

        for neuron in layer:
            print(f'collectors {collectors}')
            try:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= lr * neuron.delta * collectors[j]
                neuron.weights[-1] -= lr * neuron.delta
            except: # last layer
                pass

# Activate Neuron
def activate(neuron):
    for connection in neuron.connections:
        for i, connected_neuron in enumerate(connection):   
            connection[i].collector += neuron.collector * neuron.weights[i]
        connection[i].collector = transfer(connection[i].collector)

# Activation Function (Sigmoid)
def transfer(collector):
    return 1 / (1 + math.exp(-collector))

# Derivative of Activation Function
def transfer_derivative(collector):
    return collector * (1.0 - collector)

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
    # train network
    train(neural_network, inputs, lr = 0.4, n_epochs = 1, target_error = 0.05)