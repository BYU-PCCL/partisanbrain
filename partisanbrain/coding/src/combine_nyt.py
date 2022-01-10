import numpy as np
import pandas as pd
from nyt_categories import categories

data = pd.read_csv('data/nyt/nytimes.csv', encoding='unicode_escape')
data['category'] = data.topic_2digit.map(categories)


bodies = pd.read_pickle('data/nyt/bodies.pkl')
# drop all rows where body is empty
# bodies = bodies[bodies.body != '']

# merge on title
data = pd.merge(data, bodies, on='title', how='inner')

# count instances of unique values of each category
print(data.category.value_counts())