import time
import logging
from functools import wraps
from importlib import reload

# http://stackoverflow.com/questions/18786912/get-output-from-the-logging-module-in-ipython-notebook
# ipython notebook already call basicConfig somewhere, thus reload the logging
reload(logging)

def logger(func):
	"""
	create logging for the function,
	re-define the format to add specific logging time
	"""
	@wraps(func)
	def wrapper( *args, **kwargs ):
        logging.basicConfig( filename = '{}.log'.format( func.__name__ ),
							 format   = '%(asctime)s -- %(levelname)s:%(name)s: %(message)s',
							 datefmt  = '%Y/%m/%d-%H:%M:%S',
							 level    = logging.INFO )
		
		# custom the logging information
		logging.info( 'Ran with args: {} and kwargs: {}'.format( args, kwargs ) )
		return func( *args, **kwargs )

	return wrapper


def timer(func):
	"""time the running time of the passed in function"""
	@wraps(func)
	def wrapper( *args, **kwargs ):
		t1 = time.time()
		result = func( *args, **kwargs )
		t2 = time.time() - t1
		print( '{} ran in: {} sec'.format( func.__name__, t2 ) )

	return wrapper


