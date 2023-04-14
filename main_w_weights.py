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
#        print(inputs)

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
            try:   # to make connection (will fail on last layer)
                # add next layer to neuron connections
                neuron.connections.append(network[network.index(layer)+1])

                # setting weights along with connection
                for i in range(len(network[network.index(layer)+1])):
                    neuron.weights.append(round(random.uniform(0,1), 2))
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
        
def print_weights(network):
    # for each layer in network
    for layers in network:
        for neuron in layers:
            print(neuron.weights, end=' ')    # print weight of neuron, end with space not newline            
        print()

def print_connections(network):
    # for each layer in network
    for layers in network:
        for neuron in layers:
            for connection in neuron.connections:
                print(connection, end=' ')
            print(neuron.weights, end=' ')
            print()          
        print()

def train(network, train_data, lr, n_epochs, target_error):
    num_inputs = network_layers[0]
    for epoch in range(n_epochs):
        sum_error = 0
        for row in train_data:
        #row = train_data
            new_network(row)
            forward_prop(row)
            print(network)
            expected = [row[-1]]
            print(f'expected {expected}')
            for i in range(len(network[-1])):
                expected.append(row[num_inputs+i])
                sum_error += (row[num_inputs+i] - network[-1][i].collector)**2
            if sum_error <= target_error:
                print('target error reached=%.3f' % sum_error)
                return
            """
            print('---------------------------')
            print('expected')
            print('---------------------------')
            print(expected)
            """
            backward_prop(expected)
#            update_weights(network, row, lr)
            
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lr, sum_error))
        
def forward_prop(row):
    print('---------------------------')
    print('forwardprop')
    print('---------------------------')
    inputs = row
    for layers in network:
        print()
        print('layer')
        new_inputs = []
        for neuron in layers:
            for connection in (neuron.connections):
                for connected_neuron in connection:
                    print(f'neuron {neuron.collector} {neuron.weights}')
                    activate(neuron)
                    new_inputs.append(neuron.collector)
                    print(connected_neuron.collector)
                    #connected_neuron.collector
#        print('done')
        for connection in (neuron.connections):
            for connected_neuron in connection:
#                print(connected_neuron.collector)
                connected_neuron.collector = transfer(connected_neuron.collector)
#                print(connected_neuron.collector)
        #print('done')
        inputs = new_inputs
#        for neuron in layers:
#            print(f'neuron collector {neuron.collector}')
    print(network[-1][0].collector)
    """
    print(f'new inputs {inputs}')
    return inputs
    """

def backward_prop(expected):
    print('---------------------------')
    print('backprop')
    print('---------------------------')
    for i in reversed(range(len(network))):
        #layer = network[i]
        errors = []
        print()
        print(f'layer {i} size {len(network[i])}')
        #print(errors)

        if i != len(network)-1:
            print(f'network - {network[i]}')
            for j in range(len(network[i])):
                print(f'i j {i} {j}')
                error = 0.0
                error_layer = []
                print(f'neuron weights - {network[i][j].weights}')
                for k in range(len(network[i][j].weights)):
                    print(f'neuron #{j} - {network[i][j]}   ')
                    print(f'weight {network[i][j].weights[k]}')
                    print(f'error_delta {network[i+1][k].delta}')
   #                 print(f'error {network[i][j].weights[k] * network[i+1][k].delta}')
                    print()
                    error += (network[i][j].weights[k] * network[i+1][k].delta)

                errors.append(error)
         #       print(f'errors {errors}')
          #      print()

        else:
            print('ELSE')
            for j in range(len(network[i])):
                neuron = network[i][j]
#                print(f'output - {neuron.collector}')
#                print(f'expected j {expected[j]}')
                error = (neuron.collector - expected[j]) #* transfer_derivative(neuron.collector) 
                errors.append(error)
#                print(f'errors j {error} {errors}')
        for j in range(len(network[i])):
 #       for j, neuron in enumerate(network[i], -1):
            neuron = network[i][j]
        #    print(f'setting delta')
            print(f'i j {i} {j}')
            print(f'neuron #{j} - {neuron} - {neuron.collector} error {errors[j]}')
            neuron.delta = (errors[j] * transfer_derivative(neuron.collector))
            print(f'neuron delta - {neuron.delta}')
            print()

def activate(neuron):
    collectors = []
    collector = 0.0
#    for i in range(len(neuron.connections)):
#    for i in range(len(inputs)):
    for connection in neuron.connections:
        for i, connected_neuron in enumerate(connection):   
#            print(f'neuron collector {neuron.collector} - weights {neuron.weights[i]}')            
            connection[i].collector += neuron.collector * neuron.weights[i]
#            print(f'collector {connection[i].collector} neuron coll {neuron.collector * neuron.weights[i]}  - {i}')
#                print(f'{connected_neuron.connections} - connections')

#            print(f'collector {i} {connection[i].collector} - transfer collector {transfer(collector)}')

    # last layer 
    if len(neuron.connections) < 1:
        print('last')
        connected_neuron.collector = transfer(connected_neuron.collector)
        print(f'output {connected_neuron.collector}')

def transfer(collector):
    return 1 / (1 + math.exp(-collector))

def transfer_derivative(collector):
    return collector * (1.0 - collector)

def update_weights(network, row, lr):
    print('---------------------------')
    print('update weights')
    print('---------------------------')
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron.collector for neuron in network[i-1]]
        print()
        print(f'inputs {inputs}')
        print()
        for neuron in network[i]:
            try:
                for j in range(len(network[i+1])):
                    #  print(neuron.weights)
                    print(f'weights {neuron.weights} {neuron.weights[j]} - i {i} j {j}')
                    neuron.weights[j] += lr * neuron.delta * inputs[j]
                    print(f'lr {lr} delta {neuron.delta} inputs {inputs[j]}')
                    print(f'new weight {neuron.weights[j]}')
                neuron.weights[-1] += lr * neuron.delta
                print('neuron')
            except:
                print('except last layer')

            

def new_network(row):
    # new network
    # make layers
    make_input_layer(network_layers, row)
    make_hidden_layers(network_layers)
    # make connections
    make_connections(network)

### Main
if __name__ == '__main__':
    # declare arrays
    network = []        # main network
    network_layers = [] # size of each layer
    inputs = []         # input values for input layer

    # read file for layer sizes and input values
    read_file('layers')
    read_input('inputs')
    

    # set collectors
    #set_collectors(network)

    # print values of layers
    #print_layers(network)

    # print weights
#    print_weights(network)

    # print connections and weights
#    print_connections(network)


  #  print(network[-1])

    train(network, inputs, lr = 0.1, n_epochs = 1, target_error = 0.01)


