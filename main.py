# Neuron Class
class Neuron:
    def __init__(self):
        self.collector = 0.0        # value of neuron as float
        self.connections = []       # list of connections
        self.weights = []           # list of weights # TODO: add weights

# Read File for Layer Sizes
def read_file(file):
    with open(file, 'r') as f:      # open file
        temp = f.read().split(',')  # seperate by commas
        for num in temp:            # for each number
            network_layers.append(int(num.strip())) # add number as int to network_layers array

# Read File for Input Values
def read_input(file):
    with open(file, 'r') as f:      # open file
        temp = f.read().split(',')  # seperate by commas             
        for num in temp:            # for each number
            inputs.append(float(num.strip()))   # add number as float to inputs array

# Make Input Layer of Network
def make_input_layer(layers, input):
    neurons = []                        # create layer to add to network
    for i in range(layers[0]):          # for each neuron in layer
        neuron = Neuron()               # create neuron
        neuron.collector = input[i]     # set input to neuron.collector
        neurons.append(neuron)          # add neuron to layer
    network.append(neurons)             # add input layer to network

# Make Remaining Layers of Network
def make_hidden_layers(layers):
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
            try:   # try to make connection (will fail on last layer)
                neuron.connections.append(network[network.index(layer)+1]) # add next layer to neuron connections
            except: # pass on last layer
                pass

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
        print()                                 # print newline

# main
if __name__ == '__main__':
    # declare arrays
    network = []        # main network
    network_layers = [] # size of each layer
    inputs = []         # input values for input layer

    # read file for layer sizes and input values
    read_file('layers')
    read_input('inputs')

    # make layers
    make_input_layer(network_layers, inputs)
    make_hidden_layers(network_layers)

    # make connections
    make_connections(network)
    
    # set collectors
    set_collectors(network)

    # print values of layers
    print_layers(network)