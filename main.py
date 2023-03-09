import random
import string
import itertools
import operator

NODE_COUNT_PER_LAYER = [] #amount of nodes and layers
arr = [] #holds nodes
route = [] #holds route from node to node through each layer

class Node:
	def __init__(self):
		self.children = [] #connection to children
		self.weight = [] #weight of connection to children
		self.value = 0
		self.node_name = ''
		random_letters = []

		for i in range(3): #set node name to random 3 uppercase letters
			random_letters.append( random.choice(string.ascii_uppercase) )
		self.node_name = ''.join(random_letters)


	def make_children(self, current_layer_number, node_per_layer_map): #make nodes for each layer
		if current_layer_number >= len(node_per_layer_map): #terminate recursion
			return

		for i in range ( node_per_layer_map[current_layer_number] ): 
			self.children.append( Node() )

		self.children[0].make_children( current_layer_number + 1, node_per_layer_map)

		for i in range(0, len(self.children) ):
			self.children[i].children = self.children[0].children[:]
		
	def print_connections(self, current_layer_number, node_per_layer_map):
		indent = '	       ' * current_layer_number
		
		if current_layer_number >= len(node_per_layer_map): #end function
			print(f"{indent} {self.node_name}")
			return

		print(f"{indent} -{self.node_name} is connected to:")

		for i in range( len(self.children) ):
			try:
				print(f"{indent}    with weight of {self.weight[i]}")
			except:
				pass
			self.children[i].print_connections(current_layer_number + 1, node_per_layer_map,)
		return

	def set_random_weights(self, current_layer_number, node_per_layer_map):
		if current_layer_number >= len(node_per_layer_map): #end function
			return

		self.weight = [0] * len(self.children)

		for i in range( len(self.children) ):
			self.weight[i] = random.uniform(0, 100)
			self.children[i].set_random_weights(current_layer_number + 1, node_per_layer_map)
		return

	def insertionSort(self, arr, names): #sorts nodes by weight, does the same to list of node names
		#traverse through 1 to len(arr)
		for i in range(1, len(arr)):
			key = arr[i]
			secondkey = names[i]
			#move elements of arr[0... i-1], that are
			#greater than key, to one position ahead
			#of their current position
			j = i - 1
			while j >= 0 and key < arr[j] :
				arr[j+1] = arr[j]
				names[j+1] = names[j]
				j -= 1
			arr[j+1] = key
			names[j+1] = secondkey
		return

	def addNodesToArray(self, current_layer_number, node_per_layer_map): #adds node of layer to list arr
		
		for i in range(len(self.children)):
			arr.append(self.children[i])
		return

	def allInOne(self, current_layer_number, node_per_layer_map, names): #where everything happens

		if current_layer_number >= len(node_per_layer_map): #end recursion
			route.append(self.node_name) #adds destination node to route before ending
			return
		
		else:
			self.addNodesToArray(current_layer_number, NODE_COUNT_PER_LAYER) #adds nodes of layer to list

			print("---LAYER UNSORTED---") #lists names and weights of current layer nodes
			for i in range(len(names)):
				print(f"{names[i].node_name} --- {self.weight[i]}")

			self.insertionSort(self.weight, names) #sorts nodes by weight

			print("---LAYER SORTED---") #lists names and weights of sorted current layer nodes
			for i in range(len(names)):
				print(f"{names[i].node_name} --- {self.weight[i]}")
			
			route.append(self.node_name) #adds node name to route of nodes taken

			print(f"--- {self.node_name} -> {names[-1].node_name} ---") #prints route taken for that layer
			
			temp = names[-1] #sets temp to the highest weighted node aka next node
			names.clear() #clears list of nodes of layer to prepare for next layer
			temp.allInOne(current_layer_number + 1, node_per_layer_map, names) #run again with next node / layer
			
def generateLayers(numberOfLayers): #generates layers and nodes
        for i in range(numberOfLayers):
            NODE_COUNT_PER_LAYER.append(numberOfLayers + 1 - i)


#main

generateLayers(5)


new_node = Node() #make master node
new_node.make_children(0, NODE_COUNT_PER_LAYER) #make children from master node

#print node connections w/o weights
print("--------NODE CONNECTIONS--------")
new_node.print_connections(0, NODE_COUNT_PER_LAYER)

#adding weights
new_node.set_random_weights(0, NODE_COUNT_PER_LAYER)
#print connections w/ weights
print("\n--------CONNECTIONS WITH WEIGHTS--------")
new_node.print_connections(0, NODE_COUNT_PER_LAYER)

#processes and prints which route to take
new_node.allInOne(0, NODE_COUNT_PER_LAYER, arr)

routeString = "" #clean printable route list
for i in range(len(route)): #adds each node name from route list to stringe
	routeString += route[i]
	if i < len(route) - 1: #to format no arrow at the end of string
		routeString += " ---> "

#prints layers and each node per layer
print(NODE_COUNT_PER_LAYER)

#prints route taken due to highest weight
print(routeString)