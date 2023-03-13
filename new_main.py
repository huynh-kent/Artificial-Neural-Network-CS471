
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
            network.append(int(num.strip()))
        #return f.read().split(',')

def read_input(file):
    with open(file, 'r') as f:
        temp = f.read().split(',')
        print(temp)
        for num in temp:
            inputs.append(float(num.strip()))


def make_layers(layers, ):
    for layer in layers:
        for i in len(layer):
            neurons = []

    

network_layers = []
network = []
inputs = []
read_file('layers')
read_input('inputs')
print(network)
print(inputs)