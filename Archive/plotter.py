import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

y = 'Class'
x1 = 'SimVal (nsq)'
x2 = 'SimVal (lda)'
x3 = 'SimVal (lsa)'
data1 = pd.read_csv('../Code/datasets/ADA_Exercise_Questions_Relabelled_v7.csv', usecols=[2, 5], names=[y, x1])
#data2 = pd.read_csv('../Code/datasets/ADA_Exercise_Questions_Relabelled_v3.csv', usecols=[3, 1], names=[y, x2])
#data3 = pd.read_csv('../Code/datasets/ADA_Exercise_Questions_Relabelled_v3.csv', usecols=[4, 1], names=[y, x3])
sns.swarmplot(data=data1, x=y, y=x1)
#sns.swarmplot(data=data2, x=y, y=x2)
#sns.swarmplot(data=data3, x=y, y=x3)
plt.show()
'''
y1 = 'Class'
x1 = 'nsq'

data1 = pd.read_csv('datasets/OS_Exercise_Questions_Relabelled_v1.csv', usecols=[2, 5], names=[y1, x1])
print('data1', data1)
for x in data1:
    print(x)
sns.swarmplot(data=data1, x=y1, y=x1)
plt.show()
'''