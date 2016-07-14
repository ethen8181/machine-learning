import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import combinations

class TSPGA(object):
	"""
	Travel Salesman Problem using Genetic Algorithm

	Parameters
	----------
	generation : int
		number of iteration to train the algorithm

	population_size : int
		number of tours in the population

	retain_rate : float between 0 ~ 1
		the fraction of the best tour (shortest total distance) 
		in the population to retain, which is then used in the 
		crossover and mutation staget to generate the children 
		for the next generation
	
	mutate_rate : float between 0 ~ 1
		the probability that each tour will mutate

	Example
	-------
	%matplotlib inline
	import pandas as pd
	from tsp_solver import TSPGA
	import matplotlib.pyplot as plt

	# toy dataset
	tsp_data = pd.read_table( 
		'TSP_berlin52.txt', 
		skiprows = 1, # the first row is simply the number of cities
		header = None, 
		names = [ 'city', 'x', 'y' ], 
		sep = ' '
	)
	
	# specify the parameters and fit the data
	tsp_ga = TSPGA( 
		generation = 5000, 
		population_size = 250, 
		retain_rate = 0.5, 
		mutate_rate = 0.25
	)
	tsp_ga.fit(tsp_data)
	
	# distance convergence plot, and the best tour's distance
	# and the corresponding city tour
	tsp_ga.convergence_plot()
	tsp_ga.best_tour

	Reference
	---------
	http://www.theprojectspot.com/tutorial-post/applying-a-genetic-algorithm-to-the-travelling-salesman-problem/5
	"""
	def __init__( self, generation, population_size, retain_rate, mutate_rate ):
		self.generation  = generation
		self.retain_rate = retain_rate
		self.mutate_rate = mutate_rate		
		self.population_size = population_size
		self.retain_len  = int( population_size * retain_rate )
	
	
	def fit( self, tsp_data ):
		"""
		fit the genetic algorithm on the tsp data

		Parameters
		----------
		tsp_data : DataFrame 
			contains the 'city' and its 'x', 'y' coordinate, note that 
			the column name must match or the code will break (ordering 
			of the column does not matter)
		"""
		
		# store the coordinate and total number of cities
		self.city_num = tsp_data.shape[0]
		self.cities = tsp_data[['x', 'y']].values
		
		# a mapping from city to index to make life easier to
		# compute and store the pairwise distance
		self.city_dict = { city: index for index, city in enumerate(tsp_data['city']) }
		self.pairwise_distance = self._compute_pairwise_distance()
		
		# 1. data structure that stores the tour (city's order) and it's distance
		# 2. randomly generate the initial population (list of tour_info)
		self.tour_info = namedtuple( 'tour_info', [ 'dist', 'tour' ] )
		population = self._generate_tours( city = tsp_data['city'] )
		
		# train the genetic algorithm, during the process
		# store the best tour and its distance for each generation,
		# so we can get an idea of when the algorithm converged
		self.generation_history = []
		for _ in range(self.generation):
			population, generation_best = self._evolve(population)
			self.generation_history.append(generation_best)
		
		self.best_tour = self.generation_history[self.generation - 1]
		self.is_fitted = True
		return self


	def _compute_pairwise_distance(self):
		"""
		readable but not optimized way of computing and storing 
		the symmetric pairwise distance for between all city pairs
		"""
		pairwise_distance = np.zeros( ( self.city_num, self.city_num ) )
		for i1, i2 in combinations( self.city_dict.values(), 2 ):
			distance = np.linalg.norm( self.cities[i1] - self.cities[i2] )
			pairwise_distance[i1][i2] = pairwise_distance[i2][i1] = distance

		return pairwise_distance


	def _generate_tours( self, city ):
		"""
		or in genetic algorithm terms, generate the populations.
		generate a new random tour with the size of the population_size
		and compute the distance for each tour
		"""
		tours = []
		for _ in range(self.population_size):
			tour = city.values.copy()
			np.random.shuffle(tour)
			tours.append(tour)

		# combine the tour and distance into a single namedtuple,
		# store all of them in a list, this serves as the population
		# and sort the population increasingly according to the tour's distance
		tours_dist = [ self._compute_tour_distance( tour = t ) for t in tours ]
		population = [ self.tour_info( dist, tour ) 
					   for dist, tour in zip( tours_dist, tours ) ]
		population = sorted(population)
		return population

	
	def _compute_tour_distance( self, tour ):
		"""
		1. compute the total distance for each tour,
		note that tour stores the name of the city, thus you need to map it
		with the city_dict to access the pairwise distance.
		2. initialize the distance with the last city to the first city's distance
		"""
		first_city = self.city_dict[ tour[0] ]
		last_index = self.city_num - 1
		last_city  = self.city_dict[ tour[last_index] ]	
		tour_dist  = self.pairwise_distance[last_city][first_city]
		
		for index, city_index in enumerate(tour):
			city = self.city_dict[city_index]
			next_index = index + 1
			next_city  = self.city_dict[ tour[next_index] ]
			tour_dist += self.pairwise_distance[city][next_city]
			
			if next_index == last_index:
				break

		return tour_dist


	def _evolve( self, population ):
		"""
		core method that does the crossover, mutation to generate
		the possibly best children for the next generation
		"""

		# retain a copy of the population, why it is needed is explained
		# in the try, except block below
		population_backup = population.copy()

		# retain the best tour (number determined by the retain_len)
		population = population[:self.retain_len]
		parent = [ p.tour for p in population ]

		# generate the children (tours) for the next generation
		children = []
		while len(children) < self.population_size:			
			child = self._crossover(parent)
			child = self._mutate(child) 
			children.append(child)

		# evaluate the children chromosome and retain the overall best,
		# overall simply means the best from the parent and the children, where
		# the size retained is determined by the population size
		children_dist = [ self._compute_tour_distance( tour = c ) for c in children ]
		population_children = [ self.tour_info( dist, tour ) 
								for dist, tour in zip( children_dist, children ) ]
		
		# if the generated two tours are the same or have the same length
		# (rarely occurs) then sorting will break; if this happens, simply
		# use the original population as it seems unreasonable to check 
		# every combination of the tour to spot the duplicate
		try:
			population.extend(population_children)
			population = sorted(population)
			population = population[:self.population_size]
		except ValueError:
			population = population_backup
		
		generation_best = population[0]
		return population, generation_best


	def _crossover( self, parent ):
		"""randomly select two parent and mate them"""
		index1, index2 = random.sample( range(self.retain_len), k = 2 )
		male, female = parent[index1], parent[index2]

		# generate the position to slice one of the parent,
		# then combine the two parent into one, during this process,
		# make sure every city is being considered
		pos_start, pos_end = random.sample( range(self.city_num), k = 2 )
		if pos_start > pos_end:
			pos_start, pos_end = pos_end, pos_start

		# remove the city that are already in female,
		# and concatenate the array together
		subset  = slice( pos_start, pos_end )
		boolean = np.in1d( female, male[subset], invert = True )
		not_in_male = female[boolean]
		child = np.r_[ not_in_male[:pos_start], male[subset], not_in_male[pos_start:] ]		
		return child


	def _mutate( self, child ):
		"""
		randomly swap the position of two cities if
		the the mutuate_rate's threshold is met
		"""
		if self.mutate_rate > random.random():
			swap1, swap2 = random.sample( range(self.city_num), k = 2 )
			child[swap1], child[swap2] = child[swap2], child[swap1]

		return child


	def convergence_plot(self):
		"""
		convergence plot showing the decrease of each generation's 
		best tour's distance
		"""
		if not self.is_fitted:
			ValueError('you have not fitted the algorithm using .fit')
		
		dist = [ g.dist for g in self.generation_history ]
		plt.plot( range( 1, len(dist) + 1 ), dist, '-' )
		plt.title( 'Distance Convergence Plot' )
		plt.xlabel('Iteration')
		plt.ylabel('Distance')
		plt.tight_layout()
		plt.show()

