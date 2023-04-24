import random
import math
import pandas as pd
import numpy as np
import os.path as path

# Neuron Class
class Neuron:
    def __init__(self):
        self.collector = 0.0        # value of neuron as float
        self.connections = []       # list of connections
        self.weights = []           # list of weights
        self.delta = 0.0            # delta value of neuron as float

def read_csv_data(file):
    with open(file, 'r') as f:  # open file
        lines = f.readlines()      # read lines
        for line in lines:  # for each line
            temp = line.split(',')  # seperate by commas
            # for num in temp:
            #     if num == 'A': temp.remove(num)  

            # unnormalized data
            row = [(float(num.strip())/255) for num in temp]
            if row[-1] != 0.0: row[-1] = 1.0
            #print(len(row))

            # # normalize data
            # row = []
            # for num in temp:
            #    normalized_num = float(num.strip())/255
            #    row.append(normalized_num)
            csv_inputs.append(row)          # add row to inputs array

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
def make_input_layer(network, layers):
    neurons = []                        # create layer to add to network
    for i in range(layers[0]):          # for each neuron in layer
        neuron = Neuron()               # create neuron
    #    neuron.collector = input[i]     # set input to neuron.collector
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
                #neuron.weights.append(1.0) # bias weight
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

def new_input_layer(network, input):
    for i in range(len(network[0])):
        network[0][i].collector = input[i]

def reset_neurons(network):
    for layer in network:
        for neuron in layer:
            neuron.delta = 0.0
            neuron.collector = 0.0

# Train Network
def train(network, data, lr, n_epochs, target_error, n_batches, sample_size):
    num_outputs = network_layers[-1]
#    epoch_list = []
    for n_batches, batch in enumerate(range(n_batches), 1):
        print(f'starting batch {n_batches}')
        # same sample for each epoch
        train_data = get_sample(data, sample_size)
        for epoch_num, epoch in enumerate(range(n_epochs), 1):
            sum_error = 0.0
            # new sample each epoch
            #train_data = get_sample(data)
            for row in train_data:
            #    reset_neurons(network)
                new_input_layer(network, row)
                outputs = forward_prop(network, row)
                expected = [0 for i in range(num_outputs)]
                expected = [row[-1]]

                # test_error = 0.0
                # for i in range(len(expected)):
                #     test_error += (expected[i] - outputs[i])**2
                #print(f'expected {expected} output {outputs[0]:2f}')

                
                error = sum((expected[i]-outputs[i])**2 for i in range(len(expected)))
                sum_error += error

                backward_prop(network, expected)
                update_weights(network, lr)

                #print(f'output {outputs} expected {expected[0]:.0f} error {error:.3f}')

            # if sum_error <= target_error:
            #     epoch_list.append('--->epoch=%d, lr=%.2f, error=%.3f' % (epoch_num, lr, sum_error, ))
            #     for epoch in epoch_list:
            #         print(epoch)
            #     print('target error reached=%.3f' % sum_error)
            #     return
            

            print('>epoch=%d, lr=%.2f, error=%.3f' % (epoch_num, lr, sum_error))
            # reaches desired accuracy
            if sum_error <= target_error:
                print('target error reached=%.3f' % sum_error)
                return
        print(f'batch {n_batches} complete with sum error {sum_error:.3f}')
            #epoch_list.append('>epoch=%d, lr=%.2f, error=%.3f' % (epoch_num, lr, sum_error))
        # for epoch in epoch_list:
        #     print(epoch)

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
                #activation += prev_neuron.weights[-1] # bias
                activation += inputs[k] * prev_neuron.weights[j]
            neuron.collector = transfer(activation)
            new_inputs.append(neuron.collector)
            
        inputs = new_inputs
    return inputs

# Backward Propagation
def backward_prop(network, expected):
    for i in reversed(range(len(network))):
 #       errors = []
        if i != len(network)-1: # 3-1 not last layer
            for j in range(len(network[i])):
                neuron = network[i][j]
                weighted_sum = 0.0
                for k in range(len(network[i+1])):
                    prev_neuron = network[i+1][k]
                    weighted_delta = neuron.weights[k] * prev_neuron.delta
                    weighted_sum += weighted_delta
                neuron.delta = weighted_sum * transfer_derivative(neuron.collector)
        else: # last layer
            for j in range(len(network[i])):
                neuron = network[i][j]
                weighted_sum = neuron.collector - expected[j]
 #               errors.append(weighted_sum)
                neuron.delta = weighted_sum * transfer_derivative(neuron.collector)

# Update Weights
def update_weights(network, lr):
    for i in range(len(network)-1):
        for j in range(len(network[i])):
            neuron = network[i][j]
            for k in range(len(network[i+1])):
                delta = network[i+1][k].delta
                neuron.weights[k] -= lr * delta * neuron.collector
            #neuron.weights[-1] -= lr * delta # bias

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

# load csv into pandas df
def get_df(file, letter):
    df = pd.read_csv(file)
    # create expected output column
    df['expected'] = np.where(df['letter'] == f'{letter}', 1, 0)
    # drop letter label column
    df.drop(columns=['letter'], inplace=True)

    return df

# get random sample of training data from df
def get_sample(df, sample_size):
    letter_sample = df[df['expected']==1].sample(n=int(sample_size/5)) # 20% sample of training letter
    not_letter_sample = df[df['expected']==0].sample(n=int(sample_size*4/5)) # 80% sample of training not letter
    sample = pd.merge(not_letter_sample, letter_sample, how='outer')    # merge samples into one df

    # convert df to list of inputs
    sample_inputs = []
    for index, row in sample.iterrows():
        row = [(float(num)/255.0) for num in row] # normalize inputs
        if row[-1] != 0.0:  
            row[-1] = 1.0   # set expected output to 1.0
        sample_inputs.append(row)

    return sample_inputs



# Create Network
def new_network():
    # new network
    # make layers
    network = []
    make_input_layer(network, network_layers)
    make_hidden_layers(network, network_layers)
    # make connections
    make_connections(network)
    return network

def load_network(file):
    network = []
    make_input_layer(network, network_layers)
    make_hidden_layers(network, network_layers)
    load_weights(network, file)
    return network

def save_weights(network, path):
    with open(path, 'w') as f:
        for layer in network:
            for neuron in layer:
                f.write(str(neuron.weights))
                f.write('\n')

def load_weights(network, file):
    with open(file, 'r') as f:
        for i in range(len(network)-1):
            for j in range(len(network[i])):
                neuron = network[i][j]
                line = f.readline().strip('[]\n')
                neuron.weights = [float(num) for num in line.split(',')]


# TODO prediction
def predict(network, row):
    outputs = forward_prop(network, row)
    #return outputs.index(max(outputs))
    return outputs


### Main
if __name__ == '__main__':
    # declare arrays
    neural_network = []        # main network
    network_layers = [] # size of each layer
    inputs = []         # input values for input layer
    weights_network = []    # weights for each neuron

    # handwriting testing
    # letter desired
    letter = 'A'
    # get network layers
    read_file('handwriting_layers')
    # load data
    df = get_df('A_Z_cleaned.csv', letter)
    # weights file path
    weights_file = f'weights_{letter}.txt'

    # check if saved weights
    if path.exists(weights_file):
        neural_network = load_network(weights_file)  # load saved weights
    else: neural_network = new_network()                        # create new network

    

    # for row in csv_inputs:
    #     print(row)
    #     print('------------------------------')

    # train
    train(neural_network, df, lr = 0.8, n_epochs = 10, target_error = 0.05, n_batches=1,sample_size=100)

    # save trained weights
    save_weights(neural_network, weights_file)
