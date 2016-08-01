import os
import pytest
import pandas as pd

@pytest.fixture()
def df():
	"""read in the raw data file and return the dataframe"""
	pump_data_path = os.path.join( 'data', 'raw', 'pumps_train_values.csv' )
	df = pd.read_csv(pump_data_path)
	return df


def test_df_fixture(df):
	assert df.shape == (59400, 40)

	useful_columns = [ 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'region',
					   'population', 'construction_year', 'extraction_type_class',
					   'management_group', 'quality_group', 'source_type',
					   'waterpoint_type', 'status_group' ]

	for column in useful_columns:
		assert column in df.columns
