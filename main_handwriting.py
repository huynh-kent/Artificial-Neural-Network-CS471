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
def read_layers(file):
    network_layers = []             # create array to store layer sizes
    with open(file, 'r') as f:      # open file
        temp = f.read().split(',')  # seperate by commas
        for num in temp:            # for each number
            network_layers.append(int(num.strip())) # add number as int to network_layers array
    
    return network_layers


# Read File for Input Values
def read_input(file):
    with open(file, 'r') as f:  # open file
        lines = f.readlines()      # read lines
        for line in lines:  # for each line
            temp = line.split(',')  # seperate by commas   
            row = [float(num.strip()) for num in temp]
            inputs.append(row)          # add row to inputs array

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

def load_network(file, layers):
    network = []
    make_input_layer(network, layers)
    make_hidden_layers(network, layers)
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

# load csv into pandas df
def get_df(file, letter):
    df = pd.read_csv(file)
    # create expected output column
    df['expected'] = np.where(df['letter'] == f'{letter}', 1.0, 0.0)
    # drop letter label column
    df.drop(columns=['letter'], inplace=True)

    return df

# get random sample of training data from df
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

def predict(network, row):
    outputs = forward_prop(network, row)
    return outputs

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

def combine_network(layers, letters, layers_path):
    network = []
    new_layers = combined_layers(layers, letters)
    make_input_layer(network, new_layers)
    make_hidden_layers(network, new_layers)
    combine_weights(network, letters, layers_path)
    return network

def combined_layers(layers, letters):
    combined_layers = [layers[0]]
    combined_layers.extend(layer*len(letters) for layer in layers[1:])
    return combined_layers

def combine_weights(network, letters, layers_path):
    for i in range(len(network)-1):
        if i == 0:
            for j in range(len(network[i])):
                neuron = network[i][j]
                neuron.weights = []
                for letter in letters:
                    with open(f'weights_{letter}_{layers_path}.txt') as f:
                        line = f.readline().strip('[]\n')
                        # print(line)
                        # print()
                        neuron.weights.extend(float(num) for num in line.split(','))
        else:
            neurons_per_section = len(network[i])//len(letters)
            for j, letter in enumerate(letters):
                with open(f'weights_{letter}_{layers_path}.txt') as f:
                    for k in range(neurons_per_section):
                        #print(f'{i} {j} {k} {j*k}')
                        neuron = network[i][k+(j*neurons_per_section)]
                        line = f.readline().strip('[]\n')
                        # print(line)
                        # print()
                        neuron.weights = [float(num) for num in line.split(',')]

def combined_forward_prop(network, row, letters):
    inputs = row
    for i in range(len(network)-1):
        new_inputs = []
        if i == 0:
            for j in range(len(network[i+1])):
                neuron = network[i+1][j]
                activation = 0.0
                for k in range(len(network[i])):
                    prev_neuron = network[i][k]
                    #activation += prev_neuron.weights[-1] # bias
                    activation += inputs[k] * prev_neuron.weights[j]
                neuron.collector = transfer(activation)
                new_inputs.append(neuron.collector)

        else:
            for j in range(0, len(letters)):
                neurons_per_section = len(network[i+1])//len(letters)
                prev_neurons_per_section = len(network[i])//len(letters)
                for k in range(neurons_per_section):
                    #print(f'{i} {j} {k} {k+(j*neurons_per_section)}')
                    neuron = network[i+1][k+(j*neurons_per_section)]
                    activation = 0.0
                    for l in range(prev_neurons_per_section):
                        prev_neuron = network[i][l+(j*prev_neurons_per_section)]
                        # print(f'{i} {j} {k} {l+(j*prev_neurons_per_section)}')
                        # print(f'inputs {inputs[l+(j*prev_neurons_per_section)]}')
                        #print(f'weights {prev_neuron.weights}')
                        # print(f'weights {prev_neuron.weights[k]}')
                        activation += inputs[l+(j*prev_neurons_per_section)] * prev_neuron.weights[k]

                    neuron.collector = transfer(activation)
                    new_inputs.append(neuron.collector)

        inputs = new_inputs
    return inputs

# Update Weights
def combined_update_weights(network, lr):
    for i in range(len(network)-1):
        for j in range(len(network[i])):
            neuron = network[i][j]
            for k in range(len(network[i+1])):
                #print(f'{i} {j} {k}')
                delta = network[i+1][k].delta
                try:
                    neuron.weights[k] -= lr * delta * neuron.collector
                except IndexError:
                    pass
            #neuron.weights[-1] -= lr * delta # bias

def combined_predict(network, row, letters):
    outputs = combined_forward_prop(network, row, letters)
    print(outputs, sep=' - ')
    # print(outputs.index(max(outputs)))
    letter_output = {0:0.0, 
                     1:15.0, 
                     2:11.0,
                     3:20.0,
                     4:18.0,
                     }
    outputs = softmax(outputs)
    output = letter_output.get(outputs.index(max(outputs)))

    return output

def a_z_predict(network, row):
    outputs = forward_prop(network, row)
    outputs = softmax(outputs)
    #print(outputs, sep=' - ')
    output = letter_output.get(outputs.index(max(outputs)))
    #print(output)
    return output

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

def combined_load_test_data(file, test_sample_size):
    df = pd.read_csv(file)
    sample_df = pd.DataFrame()
    sample_df = df.sample(test_sample_size)
    test_inputs = []
    for index, row in sample_df.iterrows():
        row = [num for num in row]
        test_inputs.append(row)
    return test_inputs


def softmax(outputs):
    e_x = [math.exp(i) for i in outputs]
    sum_e_x = sum(e_x)
    softmax_outputs = [i/sum_e_x for i in e_x]
    #print(softmax_outputs)
    return softmax_outputs


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

def load_test_data(letter):
    df = pd.read_csv(f'test_data_{letter}.csv')
    test_inputs = []
    for index, row in df.iterrows():
        row = [num for num in row]
        test_inputs.append(row)

    return test_inputs

def load_data(file):
    df = pd.read_csv(file)
    inputs = []
    for index, row in df.iterrows():
        row = [num for num in row]
        inputs.append(row)
    return inputs

def create_compiled_test_data(file, test_sample_size, letters):
    df = pd.read_csv(file)
    test_data = pd.DataFrame()
    for letter in letters:
        letter_data = df[df['letter']==f'{letter}']
        letter_sample = letter_data.sample(n=int(test_sample_size*2/10)) # 20% sample of test letter
        test_data = test_data.append(letter_sample)
    else_data = df[df['letter'].isin(letters)==False]
    else_sample = else_data.sample(n=int(test_sample_size*8/10)) # 80% sample of test random
    test_data = test_data.append(else_sample)
    test_data['expected'] = np.where(test_data['letter'].isin(letters), 1, 0)
    test_data.drop(columns=['letter'], inplace=True)
    test_data = test_data.div(255.0)
    test_data['expected'] = np.where(test_data['expected'] > 0.0, 1.0, 0.0)
    test_data.to_csv(f'test_data_compiled.csv', index=False)

# Backward Propagation
def combined_backward_prop(network, expected, outputs):
    for i in reversed(range(len(network))):
        if i == 1:
            pass
        elif i != len(network)-1: # 3-1 not last layer
            for j in range(len(network[i])):
                neuron = network[i][j]
                weighted_sum = 0.0
                for k in range(len(network[i+1])):
                    #print(f'{i} {j} {k}')
                    prev_neuron = network[i+1][k]
                    weighted_delta = neuron.weights[k] * prev_neuron.delta
                    weighted_sum += weighted_delta
                neuron.delta = weighted_sum * transfer_derivative(neuron.collector)
        else: # last layer
            for j in range(len(network[i])):
                neuron = network[i][j]
                weighted_sum = outputs[j] - expected[j]
                neuron.delta = weighted_sum * transfer_derivative(neuron.collector)

# Train Network
def combined_train(network, data, lr, n_epochs, target_error, n_batches, sample_size, letters):
    num_outputs = network_layers[-1]*len(letters)
    letter_output = {0.0:0, 
                    15.0:1, 
                    11.0:2,
                    20.0:3,
                    18.0:4,
                    0:0.0, 
                    1:15.0, 
                    2:11.0,
                    3:20.0,
                    4:18.0,
                    }
    for n_batch, batch in enumerate(range(n_batches), 1):
        print(f'starting batch {n_batch}/{n_batches}')
        # same sample for each epoch
        train_data = random.sample(data, sample_size)
        for epoch_num, epoch in enumerate(range(n_epochs), 1):
            sum_error = 0.0
            # new sample each epoch
            #train_data = get_sample(data)
            for row in train_data:
            #    reset_neurons(network)
                new_input_layer(network, row)
                outputs = combined_forward_prop(network, row, letters)
                softmax_outputs = softmax(outputs)
                # print(f'outputs {outputs}')
                # print(f'row[-1] {row[-1]}')
                # print(f'letter_output {letter_output.get(row[-1])}')
                expected = [0.0 for i in range(num_outputs)]
                expected[int(letter_output.get(row[-1]))] = 1.0
                print(f"expected {expected} softmax output {['%.3f' % output for output in softmax_outputs]} output {['%.3f' % output for output in outputs]}")

                # test_error = 0.0
                # for i in range(len(expected)):
                #     test_error += (expected[i] - outputs[i])**2
                #print(f'expected {expected} output {outputs[0]:2f}')

                
                error = sum((expected[i]-softmax_outputs[i])**2 for i in range(len(expected)))
                sum_error += error

                combined_backward_prop(network, expected, outputs)
                combined_update_weights(network, lr)

                #print(f'output {outputs} expected {expected[0]:.0f} error {error:.3f}')
            
            # print error each epoch
            print('>epoch=%d error=%.3f -------- letter=%s lr=%.2f sample size=%d layers=%s' % (epoch_num, sum_error, letter, lr, sample_size, layers))

            # if target error reached, go to next batch
            if sum_error <= target_error:
                print('target error reached=%.3f' % sum_error)
                break
            
        print(f'batch {n_batch} complete with sum error {sum_error:.3f}')
            #epoch_list.append('>epoch=%d, lr=%.2f, error=%.3f' % (epoch_num, lr, sum_error))
        # for epoch in epoch_list:
        #     print(epoch)

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

    ### train models
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


    # ### combined models
    # # combined letters
    # letters = ['A', 'P', 'L', 'U', 'S']

    # if path.exists('weights_APLUS.txt'): combined_network=load_network('weights_APLUS.txt', network_layers)
    # else: combined_network = combine_network(network_layers, letters, layers)
    # test_data=combined_load_test_data('APLUS.csv', test_sample_size=10000)
    # # training combined network
    # combined_train(combined_network, test_data, lr = 0.4, n_epochs = 5, target_error = 0.05, n_batches=10000, sample_size=10, letters=letters)

    # # save weights
    # save_weights(combined_network, 'weights_APLUS.txt')
    # # testing combined network
    # accuracy = combined_test(combined_network, test_data=test_data, letters=letters)



    ### train a_z model
    # load data
    df_inputs = load_data('A_Z_Data_normalized.csv')

    weights_file = 'weights_a_z.txt'
    network_layers = read_layers('a_z_layers')

    # check if saved weights
    if path.exists(weights_file):
        neural_network = load_network(weights_file, network_layers)  # load saved weights
    else: neural_network = new_network(network_layers)               # create new network

    # train until 95% accurate
    while accuracy < 95.0:
    # train
        train(neural_network, df_inputs, lr = 0.4, n_epochs = 20, target_error = 0.05, n_batches=1, sample_size=20)
    # test
        accuracy = a_z_test(neural_network, test_data=random.sample(df_inputs, 100))
    # save trained weights
        save_weights(neural_network, 'weights_a_z.txt')

    print('finished training')



