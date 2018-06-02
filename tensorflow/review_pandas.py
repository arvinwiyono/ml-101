# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

california_housing_df = pd.read_csv('https://storage.googleapis.com/mledu-datasets/california_housing_train.csv')
california_housing_df.head()
print(california_housing_df.hist('housing_median_age'))


cities_df = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities_df['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities_df['Population density'] = cities_df['Population'] / cities_df['Area square miles']
cities_df = cities_df.reindex(np.random.permutation(cities_df.index))

# excercise 1
starts_with_san = cities_df['City name'].apply( lambda name : name.startswith('San') )
greater_than_50_miles = cities_df['Area square miles'] > 50
cities_df['Condition met'] = starts_with_san & greater_than_50_miles;
print(cities_df)

# exercise 2
# when reindexing with index that does not exist, it creates a new row 
cities_df = cities_df.reindex([4, 2, 0, 1])
print(cities_df)