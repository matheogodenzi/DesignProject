# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:30:07 2024

@author: mimag
"""

import pandas as pd

# Sample data
data = {
    'coefficients': [1/11, 2/11, 3/11, 3/11, 1/11, 1/11],
    'figures': [13,64,78,95,19,70]
}

df = pd.DataFrame(data)

# Calculate the weighted sum
df['weighted'] = df['coefficients'] * df['figures']
weighted_sum = df['weighted'].sum()
weighted_average = weighted_sum / df['coefficients'].sum()
print(weighted_sum)
