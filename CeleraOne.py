from sklearn.linear_model import RandomizedLogisticRegression
import pandas as pd
import warnings

# to avoid deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# to read the csv data using pandas
task_data = pd.read_csv('task_data.csv')

# to prepare the X and Y(labels)
colNames = task_data.columns[task_data.columns.str.contains('sensor')]
X = task_data.loc[:, colNames]
Y = task_data.class_label

# to implement randomized logistic regression. Here, the sample_fraction flag is set to 0.5,
# which defines the fraction of samples that is randomly selected and used in each iteration of
# this algorithm.
randLog = RandomizedLogisticRegression(sample_fraction=0.5)
randLog.fit(X, Y)

# to print the result
print("Features sorted by their score:")
print(sorted(zip(randLog.scores_, colNames), reverse=True))

# the output will be close to the following results
'''
Features sorted by their score:
[(0.96, 'sensor8'), (0.875, 'sensor4'), (0.69, 'sensor0'), 
(0.49, 'sensor3'), (0.04, 'sensor1'), 
(0.02, 'sensor7'), (0.005, 'sensor5'), 
(0.0, 'sensor9'), (0.0, 'sensor6'), (0.0, 'sensor2')]

'''
