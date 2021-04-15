import numpy as np
import pandas as pd

# Panda body sizes
pandas_bm = np.array([84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76])

# Bear body sizes
bears_bm = np.array([67.65, 92.13, 58.92, 87.64, 76.31, 88.86])

print(np.std(pandas_bm), np.mean(pandas_bm))

# Global land temperature from 1900 to 2020 (NASA)
climate = pd.read_csv('global_temperature_NASA.txt', sep='\t')

print(climate.year.tolist())
print(climate.temperature.tolist())