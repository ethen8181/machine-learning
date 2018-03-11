import os
import re
import requests
import pandas as pd


def main():
    file_path = 'adult.csv'
    if not os.path.isfile(file_path):
        def chunks(input_list, n_chunk):
            """take a list and break it up into n-size chunks"""
            for i in range(0, len(input_list), n_chunk):
                yield input_list[i:i + n_chunk]

        columns = [
            'age', 'workclass', 'fnlwgt', 'education',
            'education_num', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'capital_gain',
            'capital_loss', 'hours_per_week', 'native_country', 'income']

        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        r = requests.get(url)
        raw_text = r.text.replace('\n', ',')
        splitted_text = re.split(r',\s*', raw_text)
        data = list(chunks(splitted_text, n_chunk = len(columns)))
        data = pd.DataFrame(data, columns = columns).dropna(axis = 0, how = 'any')
        data.to_csv(file_path, index = False)


if __name__ == '__main__':
    main()
