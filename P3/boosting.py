import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		value11 = np.zeros(len(features))

		for i in range(len(self.clfs_picked)):
			first = self.clfs_picked[i]
			second = self.betas[i]
			value11+= second * np.array(first.predict(features))
		value = []
		for i in range(len(value11)):
			if value11[i] > 0:
				value.append(1)
			else:
				value.append(-1)

		return value
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(labels)
		w = np.array([1/N] * N)
		for t in range(self.T):
			check = 1000000000000000
			for number in self.clfs:
				fea = number.predict(features)
				# print(fea)
				test1 = []
				for i in range(len(labels)):
					if labels[i] != fea[i]:
						test1.append(True)
					else:
						test1.append(False)
				test1 = np.array(test1)
				
				error = np.sum(test1*w)
				if error < check:
					htt = number
					hg = fea
					check = error
			self.clfs_picked.append(htt)

			b = 1/2 * np.log((1 - check) / check)
			self.betas.append(b)

			for i in range(N):
				if labels[i] == hg[i]:
					w[i] *= np.exp(-b)
				else:
					w[i] *= np.exp(b)

			sv = np.sum(w)
			w /=sv
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	