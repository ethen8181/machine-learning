import random
import numpy as np
from collections import namedtuple


class GA:
	"""
	Genetic Algorithm for a simple optimization problem

	Parameters
	----------
	generation : int
		number of iteration to train the algorithm

	pop_size : int
		number of chromosomes in the population

	chromo_size : int
		number of possible values (genes) per chromosome

	low, high : int
		lower_bound and upper_bound possible value of the randomly generated chromosome

	retain_rate : float 0 ~ 1
		the fraction of the best chromosome to retain. used to mate
		the children for the next generation

	mutate_rate : float 0 ~ 1
		the probability that each chromosome will mutate
	"""
	def __init__( self, generation, pop_size, chromo_size, low, high, 
				  retain_rate, mutate_rate ):		
		self.low  = low
		self.high = high
		self.pop_size = pop_size
		self.chromo_size = chromo_size
		self.generation  = generation
		self.retain_len  = int(pop_size * retain_rate)
		self.mutate_rate = mutate_rate
		self.info = namedtuple( 'info', ['cost', 'chromo'] )
	
	
	def fit(self, target):
		"""
		target : int
        	the targeted solution
		"""
		
		# randomly generate the initial population, and evaluate its cost
		array_size = self.pop_size, self.chromo_size
		pop = np.random.randint(self.low, self.high + 1, array_size)
		graded_pop = self._compute_cost( pop, target )

		# store the best chromosome and its cost for each generation,
    	# so we can get an idea of when the algorithm converged
		self.generation_history = []
		for _ in range(self.generation):
			graded_pop, generation_best = self._evolve(graded_pop, target)
			self.generation_history.append(generation_best)

		self.best = self.generation_history[self.generation - 1]
		self.is_fitted = True
		return self


	def _compute_cost(self, pop, target):
		"""
		combine the cost and chromosome into one list and sort them
		in ascending order
		"""
		cost = np.abs( np.sum(pop, axis = 1) - target )
		graded = [ self.info( c, list(p) ) for p, c in zip(pop, cost) ]
		graded = sorted(graded)
		return graded

	
	def _evolve(self, graded_pop, target):
		"""
		core method that does the crossover, mutation to generate
		the possibly best children for the next generation
		"""
		
		# retain the best chromos (number determined by the retain_len)
		graded_pop = graded_pop[:self.retain_len]
		parent = [ p.chromo for p in graded_pop ]

		# generate the children for the next generation 
		children = []
		while len(children) < self.pop_size:
			child = self._crossover(parent)
			child = self._mutate(child)
			children.append(child)

		# evaluate the children chromosome and retain the overall best,
		# overall simply means the best from the parent and the children, where
		# the size retained is determined by the population size
		graded_children = self._compute_cost(children, target)
		graded_pop.extend(graded_children)
		graded_pop = sorted(graded_pop)
		graded_pop = graded_pop[:self.pop_size]
		
		# also return the current generation's best chromosome and its cost
		generation_best = graded_pop[0]
		return graded_pop, generation_best 

	
	def _crossover(self, parent):
		"""
		mate the children by randomly choosing two parents and mix 
		the first half element of one parent with the later half 
		element of the other
		"""
		index1, index2 = random.sample( range(self.retain_len), k = 2 )
		male, female = parent[index1], parent[index2]
		pivot = len(male) // 2
		child = male[:pivot] + female[pivot:]
		return child


	def _mutate(self, child):
		"""
		randomly change one element of the chromosome if it
		exceeds the user-specified threshold (mutate_rate)
		"""
		if self.mutate_rate > random.random():
			idx_to_mutate = random.randrange(self.chromo_size)
			child[idx_to_mutate] = random.randint(self.low, self.high)

		return child

