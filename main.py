import random

class Neuron:
    def __init__(self):
        self.connections = [] # connection to next layer
        self.weights = [] # weights of connection to children
        self.bias = 1
        self.collector = 0

    def make_layers(self, current_layer, nodes_per_layer): # make nodes
        if current_layer >= len(nodes_per_layer): # terminate recursion
            return

        for i in range(nodes_per_layer[current_layer]):
            self.connections.append(Neuron())

        self.connections[0].make_layers(current_layer+1, nodes_per_layer)

        for i in range(len(self.connections)):
            self.connections[i].connections = self.connections[0].connections[:]

    def set_weights(self, current_layer, nodes_per_layer):
        if current_layer >= len(nodes_per_layer): # end recursion
            return
        
        self.weights = [0] * len(self.connections)

        for i in range(len(self.connections)): # recursion
            self.weights[i] = random.uniform(0, 1)
            self.connections[i].set_weights(current_layer+1, nodes_per_layer)

    def print_layers(self, current_layer, nodes_per_layer):
        indent = f'{current_layer}'

        if current_layer >= len(nodes_per_layer):
            print(f"{indent} {self.weights}")
        
        for i in range(len(self.connections)):
            try:
                print(f"{indent} with weight of {self.weights[i]}")
            except:
                pass
            self.connections[i].print_layers(current_layer+1, nodes_per_layer)
        return

network = []

def make_nodes():
    pass

def read_string(string):
    return string.split(',')

def read_file(file):
    with open(file, 'r') as f:
        tmp = f.read.split(',')
        for t in tmp:
            network.append(int(t.strip()))

def create_layers(num_layers, num):
    layer = []
    for i in range(num_nodes):
        layer.append(Neuron())


def main():
    # read string data
    #print(read_string('1,2,3'))
    neuron = Neuron()
    neuron.make_layers(0, [2, 3, 2])
    neuron.set_weights(0, [2, 3, 2])
    neuron.print_layers(0, [2, 3, 2])

if __name__ == "__main__":
    main()