import numpy as np
import pandas as pd

def remove_invalid_data(path):
	""" 
	Takes a path to a water pumps csv, loads in pandas, removes
	invalid columns and returns the dataframe.
	"""
	df = pd.read_csv( path, index_col = 0 )

	# Nested dictionaries, e.g., {'a': {'b': nan}}, are read as
	# follows: look in column 'a' for the value 'b' and replace it
	# with nan.
	invalid_values = {
		'amount_tsh': { 0: np.nan },
		'longitude': { 0: np.nan },
		'installer': { 0: np.nan },
		'construction_year': { 0: np.nan },
	}

	# drop rows with invalid values
	df = df.replace(invalid_values)
	df = df.dropna(how = 'any')
	return df
