import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			total = np.sum(branches,axis = 0)
			temp = branches/total
			for i in range(temp.shape[0]):
				for j in range(temp.shape[1]):
					if temp[i,j] > 0 :
						temp[i,j] = -temp[i,j]*np.log2(temp[i,j])
					else:
						temp[i,j] = 0
			temp = np.sum(temp,axis = 0)
			percetage = total/np.sum(total)
			entropy = np.sum(temp*percetage)
			return entropy
		
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################

			# if not 'min_entropy' in locals():
			border = 1000000000
			featurevalue = np.array(self.features)[:,idx_dim]
			if None in featurevalue:
				continue
			else:
				branch_values = np.unique(featurevalue)
				branches = np.zeros((self.num_cls,len(branch_values)))
				for i in range(len(branch_values)):
					label = np.array(self.labels)
					zz = []
					for j in range(len(label)):
						if featurevalue[j] == branch_values[i]:
							zz.append(label[j])
					for gg in zz:
						branches[gg,i]+=1
				
				entropy = conditional_entropy(branches)
				if entropy < border:
					border = entropy
					self.dim_split = idx_dim
					self.feature_uniq_split = branch_values.tolist()

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		label = np.array(self.features)
		featurevalue = label[:,self.dim_split]
		x = np.array(self.features,dtype = object)
		x[:,self.dim_split] = None
		for vv in self.feature_uniq_split:
			y = []
			for i in range(len(featurevalue)):
				if featurevalue[i] == vv:
					y.append(i)
			y = np.array(y)

			x_ = x[y].tolist()
			y_ = np.array(self.labels)[y].tolist()
			children = TreeNode(x_, y_, self.num_cls)
			check = np.array([None]*len(x_[0]))
			xx = np.array(x_).size
			if xx == 0 or (check == x_[0]).all():
				children.splittable = False

			self.children.append(children)
			

		
		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split]+feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



