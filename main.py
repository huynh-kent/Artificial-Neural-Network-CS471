
class Neuron:
    def __init__(self):
        self.collector = 0.0
        self.connections = []
        self.weights = []

    def make_layers(self, layers, current_layers = 0):
        # stop recursions
        if current_layers >= len(layers):
            return

        for i in range(layers[current_layers]):
            neuron = Neuron()
            # neuron.collector += self.collector
            self.connections.append(neuron)
        
        self.connections[0].make_layers(layers, current_layers+1)

        for i in range(len(self.connections)):
            self.connections[i].connections = self.connections[0].connections[:]

    def print_layers(self, layers, current_layers = 0):
        if current_layers >= len(layers):
            print(self.collector)
        
        for i in range(len(self.connections)):
            try:
                print(f"{current_layers} with weight of {self.weights[i]}")
            except:
                pass
            self.connections[i].print_layers(layers, current_layers+1)
        return

def read_file(file):
    with open(file, 'r') as f:
        temp = f.read().split(',')
        print(temp)
        for num in temp:
            network_layers.append(int(num.strip()))
        #return f.read().split(',')

def read_input(file):
    with open(file, 'r') as f:
        temp = f.read().split(',')
        print(temp)
        for num in temp:
            inputs.append(float(num.strip()))


def make_input_layer(layers, input):
    neurons = []
    for i in range(layers[0]):
        neuron = Neuron()
        neuron.collector = input[i] 
        neurons.append(neuron)
    network.append(neurons)

def make_hidden_layers(layers, current_layer = 1):
    #if current_layer >= len(layers):
    #    return

    layers = layers[1:]

    for layer in layers:
        new_layer = []
        for i in range(layer):
            neuron = Neuron()
            new_layer.append(neuron)
            # print('test')
        network.append(new_layer)

def make_connections(network):
    for layer in network:
        for neuron in layer:
            try:
                neuron.connections.append(network[network.index(layer)+1])
            except:
                pass

def set_collectors(network):
    for layer in network:
        for neuron in layer:
            print(neuron.collector, sep=' ')
            for connection in neuron.connections:
                for connected_neuron in connection:
                    connected_neuron.collector += neuron.collector

# make next layer
    # for each neuron in previous layer
        # add new layer to connections
        # add collector to nerons in new layer
        # repeat until no more layers

def print_layers(layers):
    for layer in layers:
        print(layer)

# main
if __name__ == '__main__':
    # declare variables
    network = []
    network_layers = []
    inputs = []
    # read file layers
    read_file('layers')
    # read input
    read_input('inputs')
    # make input layer
    make_input_layer(network_layers, inputs)
    make_hidden_layers(network_layers)
    # make next layer
        # for each neuron in previous layer
            # add new layer to connections
            # add collector to nerons in new layer
            # repeat until no more layers

    print(network_layers)
    print(inputs)
    #print(network)
    # master.print_layers(network_layers)
    make_connections(network)
    set_collectors(network)
    #print_layers(network[0])
    #print(network[0][0].collector)
    #print_layers(network)