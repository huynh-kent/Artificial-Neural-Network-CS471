
class Neuron:
    def __init__(self):
        self.value = 0
        self.collector = 0
        self.connections = []
        self.weights = []

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
        neuron.value = input[i] 
        neurons.append(Neuron())
    network.append(neurons)

    

network_layers = []
network = []
inputs = []
read_file('layers')
read_input('inputs')
print(network_layers)
print(inputs)
make_input_layer(network_layers, inputs)
print(network)