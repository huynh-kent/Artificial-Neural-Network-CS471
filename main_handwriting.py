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

# Read File for Layer Sizes
def read_layers(file):
    network_layers = []             # create array to store layer sizes
    with open(file, 'r') as f:      # open file
        temp = f.read().split(',')  # seperate by commas
        for num in temp:            # for each number
            network_layers.append(int(num.strip())) # add number as int to network_layers array
    
    return network_layers

# Create Network
def new_network(layers):
    # new network
    # make layers
    network = []
    make_input_layer(network, layers)
    make_hidden_layers(network, layers)
    # make connections
    make_connections(network)
    return network

# Make Network with Loaded Weights
def load_network(file, layers):
    network = []
    make_input_layer(network, layers)
    make_hidden_layers(network, layers)
    load_weights(network, file)
    return network

# Save Weights into File
def save_weights(network, path):
    with open(path, 'w') as f:
        for layer in network:
            for neuron in layer:
                f.write(str(neuron.weights))
                f.write('\n')

# Load Weights from File
def load_weights(network, file):
    with open(file, 'r') as f:
        for i in range(len(network)-1):
            for j in range(len(network[i])):
                neuron = network[i][j]
                line = f.readline().strip('[]\n')
                # print(line)
                # print()
                neuron.weights = [float(num) for num in line.split(',')]

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
                    neuron.weights.append(round(random.uniform(-0.5,0.5), 2))
                #neuron.weights.append(1.0) # bias weight
            except: # pass on last layer
                print('New Weights')

                    
# Set Input Layer of New Row of Data
def new_input_layer(network, input):
    for i in range(len(network[0])):
        network[0][i].collector = input[i]

# Train Network
def train(network, data, lr, n_epochs, target_error, n_batches, sample_size):
    num_outputs = network_layers[-1]
#    epoch_list = []
    for n_batch, batch in enumerate(range(n_batches), 1):
        print(f'starting batch {n_batch}/{n_batches}')
        # same sample for each epoch
        #train_data = get_sample(data, sample_size)
        train_data = random.sample(data, sample_size)
        for epoch_num, epoch in enumerate(range(n_epochs), 1):
            sum_error = 0.0
            # new sample each epoch
            #train_data = get_sample(data)
            for row in train_data:
            #    reset_neurons(network)
                new_input_layer(network, row)
                outputs = forward_prop(network, row)
                expected = [0.0 for i in range(num_outputs)]
                expected[int(row[-1])] = 1.0

                # test_error = 0.0
                # for i in range(len(expected)):
                #     test_error += (expected[i] - outputs[i])**2
                #print(f'expected {expected} output {outputs[0]:2f}')

                
                error = sum((expected[i]-outputs[i])**2 for i in range(len(expected)))
                sum_error += error

                backward_prop(network, expected)
                update_weights(network, lr)

                #print(f'output {outputs} expected {expected[0]:.0f} error {error:.3f}')
            
            # print error each epoch
            #print('>epoch=%d error=%.3f -------- letter=%s lr=%.2f sample size=%d layers=%s' % (epoch_num, sum_error, letter, lr, sample_size, layers))
            print('>epoch=%d/%d error=%.3f -------- lr=%.2f sample size=%d' % (epoch_num, n_epochs, sum_error, lr, sample_size))

            # if target error reached, go to next batch
            if sum_error <= target_error:
                print('target error reached=%.3f' % sum_error)
                break
            
        print(f'batch {n_batch} complete with sum error {sum_error:.3f}')
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

# Activation Function (Sigmoid)
def transfer(collector):
    return 1 / (1 + math.exp(-collector))

# Derivative of Activation Function
def transfer_derivative(collector):
    return (collector) * (1.0 - (collector))

# Load CSV into Pandas DataFrame
def get_df(file, letter):
    df = pd.read_csv(file)
    # create expected output column
    df['expected'] = np.where(df['letter'] == f'{letter}', 1.0, 0.0)
    # drop letter label column
    df.drop(columns=['letter'], inplace=True)

    return df

# Get Random Sample of Data from DataFrame
def get_sample(df, sample_size):
    letter_sample = df[df['expected']==1].sample(n=int(sample_size/2)) # 50% sample of training letter
    not_letter_sample = df[df['expected']==0].sample(n=int(sample_size/2)) # 50% sample of training not letter
    sample = pd.merge(not_letter_sample, letter_sample, how='outer')    # merge samples into one df

    # convert df to list of inputs
    sample_inputs = []
    for index, row in sample.iterrows():
        row = [(float(num)/255.0) for num in row] # normalize inputs
        if row[-1] != 0.0:  
            row[-1] = 1.0   # set expected output to 1.0
        sample_inputs.append(row)

    return sample_inputs


# Return Prediction
def predict(network, row):
    outputs = forward_prop(network, row)
    return outputs

# Test Network with Test Data
def test(network, test_data):
    correct = 0
    for row in test_data:
        outputs = predict(network, row)
        error = abs(row[-1] - outputs[-1])
        if error < 0.5:
            correct += 1
        #print(f'expected: {row[-1]}, predicted: {outputs[-1]}')
    accuracy = (float(correct) / len(test_data)) * 100.0
    
    print(f'accuracy: {accuracy:.2f}% correct/total: {correct}/{len(test_data)}')
    return accuracy

# Predict Letter
def a_z_predict(network, row):
    outputs = forward_prop(network, row)
    outputs = softmax(outputs)
    #print(outputs, sep=' - ')
    output = letter_output.get(outputs.index(max(outputs)))
    #print(output)
    return output

# Test A-Z Network with Test Data
def a_z_test(network, test_data):
    correct = 0
    for row_num, row in enumerate(test_data, 1):
        #expected = [0 for i in range(len(network[-1]))]
        expected = row[-1]
        output = a_z_predict(network, row)
        #print(f'expected: {expected} output: {outputs}')
        # print(f'{letter_output.get(output)}')
        expected_letter = letter_output.get(expected)
        #output_letter = letter_output.get(output)
        print(f'row {row_num} - expected letter: {expected_letter} output letter: {output}')
        if expected_letter == output:
            correct += 1
        # else: print(f'failed - expected letter: {expected_letter} output letter: {output_letter}')
        # error = abs(row[-1] - outputs[-1])
        # if error < 0.5:
        #     correct += 1
    #    print(f'expected: {row[-1]}, predicted: {outputs[-1]}')
    accuracy = (float(correct) / len(test_data)) * 100.0
    
    print(f'accuracy: {accuracy:.2f}% correct/total: {correct}/{len(test_data)}')
    return accuracy

# Softmax Function for Outputs
def softmax(outputs):
    e_x = [math.exp(i) for i in outputs]
    sum_e_x = sum(e_x)
    softmax_outputs = [i/sum_e_x for i in e_x]
    #print(softmax_outputs)
    return softmax_outputs

# Create Test Data from CSV
def create_test_data(file, test_sample_size, letter):
    df = pd.read_csv(file)
    letter_data = df[df['letter']==f'{letter}']
    letter_sample = letter_data.sample(n=int(test_sample_size*2/10)) # 20% sample of test letter
    else_data = df[df['letter']!=f'{letter}']
    else_sample = else_data.sample(n=int(test_sample_size*8/10)) # 80% sample of test random
    test_data = pd.merge(else_sample, letter_sample, how='outer')    # merge samples into one df
    test_data['expected'] = np.where(test_data['letter'] == f'{letter}', 1, 0)
    test_data.drop(columns=['letter'], inplace=True)
    test_data = test_data.div(255.0)
    test_data['expected'] = np.where(test_data['expected'] > 0.0, 1.0, 0.0)
    test_data.to_csv(f'test_data_{letter}.csv', index=False)

# Load Test Data from CSV
def load_test_data(letter):
    df = pd.read_csv(f'test_data_{letter}.csv')
    test_inputs = []
    for index, row in df.iterrows():
        row = [num for num in row]
        test_inputs.append(row)

    return test_inputs

# Load Data from CSV
def load_data(file):
    df = pd.read_csv(file)
    inputs = []
    for index, row in df.iterrows():
        row = [num for num in row]
        inputs.append(row)
    return inputs

### Main
if __name__ == '__main__':
    # declare variables
    accuracy = 0.0      # accuracy of network
    letter_output = {0.0:'A',
                    1.0:'B',
                2.0:'C',
                3.0:'D',
                4.0:'E',
                5.0:'F',
                6.0:'G',
                7.0:'H',
                8.0:'I',
                9.0:'J',
                10.0:'K',
                11.0:'L',
                12.0:'M',
                13.0:'N',
                14.0:'O',
                15.0:'P',
                16.0:'Q',
                17.0:'R',
                18.0:'S',
                19.0:'T',
                20.0:'U',
                21.0:'V',
                22.0:'W',
                23.0:'X',
                24.0:'Y',
                25.0:'Z',
                    }


    ############# single letter model
    # # handwriting testing
    # # letter desired
    # letter = 'S'

    # ### create test data
    # #create_test_data('A_Z_cleaned.csv', test_sample_size=1000, letter=letter)

    # # get network layers
    # network_layers = read_layers('handwriting_layers')
    # # network layers
    # layers = str(network_layers).replace(' ', '')
    # # weights file path
    # weights_file = f'weights_{letter}_{layers}.txt'

    # # load data
    # df = get_df('A_Z_cleaned.csv', letter)

    # # check if saved weights
    # if path.exists(weights_file):
    #     neural_network = load_network(weights_file, network_layers)  # load saved weights
    # else: neural_network = new_network(network_layers)               # create new network

    # # train until 95% accurate
    # while accuracy < 95.0:
    # # train
    #     train(neural_network, df, lr = 0.4, n_epochs = 10, target_error = 0.05, n_batches=10, sample_size=20)
    # # test
    #     accuracy = test(neural_network, test_data=load_test_data(letter))
    # # save trained weights
    #     save_weights(neural_network, weights_file)
    # print('finished training')

    
    #################### a_z model
    df_inputs = load_data('A_Z_Data_normalized.csv') # load data
    weights_file = 'weights_a_z.txt'                 # weights file path
    network_layers = read_layers('a_z_layers')       # get network layers

    if path.exists(weights_file):                                    # check if saved weights
        neural_network = load_network(weights_file, network_layers)  # load saved weights
    else: neural_network = new_network(network_layers)               # create new network

    # train until 95% accurate
    while accuracy < 95.0:
    # train
        train(neural_network, df_inputs, lr = 0.4, n_epochs = 10, target_error = 0.05, n_batches=5, sample_size=20)
    # test
        accuracy = a_z_test(neural_network, test_data=random.sample(df_inputs, 100))
    # save trained weights
        save_weights(neural_network, 'weights_a_z.txt')
    # reached 95% accuracy on test sample
    print('finished training')



