import random
from collections import namedtuple


def population( pop_size, chromo_len, lower_bound, upper_bound ):
	"""
	creates a collection of chromosomes (i.e. a population)

	Parameters
	----------
	pop_size : int
		number of chromosomes in the population

	chromo_len : int
		number of possible values per chromosome

	lower_bound, upper_bound : int
		lower_bound and upper_bound possible value of the randomly generated chromosome

	Returns
	-------
	(list) each element is a chromosome
	"""
	def chromosome( chromo_len, lower_bound, upper_bound ):
		return [ random.randint( lower_bound, upper_bound ) for _ in range(chromo_len) ]

	return [ chromosome( chromo_len, lower_bound, upper_bound ) for _ in range(pop_size) ]


def calculate_cost( chromosome, target ):
	"""
	Determine the cost of an chromosome. lower is better

	Parameters
	----------
	chromosome : list
		each chromosome of the entire population

	target : int
		the sum that the chromosomes are aiming for
	
	Returns
	-------
	(int) absolute difference between the sum of the numbers 
	in the chromosome and the target
	"""
	summed = sum(chromosome)
	return abs(target - summed)


def evolve( pop, target, retain = 0.5, mutate = 0.1 ):
	"""
	evolution of the genetic algorithm

	Parameters
	----------
	pop : list
		the initial population for this generation

	target : int
		the targeted solution

	retain : float
		the fraction of the best chromosome to retain. used to mate
		the children for the next generation

	mutate : float
		the probability that each chromosome will mutate

	Returns
	-------
	children : list
		the crossovered and mutated population for the next generation

	generation_best : namedtuple( "cost", "chromo" )
		the current generation's best chromosome and its cost 
		(evaluated by the cost function)
	"""

	# evolution :
	# take the proportion of the best performing chromosomes
	# judged by the calculate_cost function and these high-performers 
	# will be the parents of the next generation
	graded = [ generation_info( calculate_cost( p, target ), p ) for p in pop ]
	graded = sorted(graded)
	parents = [ g.chromo for g in graded ]
	retain_len = int( len(parents) * retain )
	parents = parents[:retain_len]

	# the children_index set is used to
	# check for duplicate index1, index2. since
	# choosing chromosome ( a, b ) to crossover is the same
	# as choosing chromosome ( b, a )
	children = []
	children_index = set()
	desired_len = len(pop)
	parents_len = len(parents)
	
	# generate the the children (the parent for the next generation),
	# the children is mated by randomly choosing two parents and
	# mix the first half element of one parent with the later half 
	# element of the other
	while len(children) < desired_len:

		index1 = index2 = random.randint( 0, parents_len - 1 )
		while index1 == index2:
			index2 = random.randint( 0, parents_len - 1 )

		if ( index1, index2 ) not in children_index:
			male   = parents[index1]
			female = parents[index2]
			pivot  = len(male) // 2
			child1 = male[:pivot] + female[pivot:]
			child2 = female[:pivot] + male[pivot:]
			children.append(child1)
			children.append(child2)
			children_index.add( ( index1, index2 ) )
			children_index.add( ( index2, index1 ) )

	# mutation :
	# randomly change one element of the chromosome
	for chromosome in children:
		if mutate > random.random():
			index_to_mutate = random.randint( 0, chromo_len - 1 )
			chromosome[index_to_mutate] = random.randint( lower_bound, upper_bound )

	# evaluate the children chromosome and retain the overall best
	graded_childen = [ generation_info( calculate_cost( p, target ), p ) for p in children ]
	graded.extend(graded_childen)
	graded = sorted(graded)
	generation_best = graded[0]
	children = [ g.chromo for g in graded[:desired_len] ]
	return children, generation_best


# --------------------------------------------------------------------------
# example

pop_size = 100
chromo_len = 5
lower_bound = 0
upper_bound = 100
pop = population( pop_size, chromo_len, lower_bound, upper_bound )

retain = 0.5
target = 200
mutate = 0.1
generation = 10

# store the best chromosome and its cost for each generation,
# so we can get an idea of when the algorithm converged
generation_history = []
generation_info = namedtuple( "generation_info", [ "cost", "chromo" ] )

for i in range(generation):
	pop, generation_best = evolve( pop, target, retain, mutate )
	generation_history.append(generation_best)
	print( "iteration {}'s best generation: {}".format( i + 1, generation_best ) )

