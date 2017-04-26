import csv
import re

s = open('temp.txt').readlines()
s = [s2 for s2 in s if s2.strip() != '\n']

s = '\n'.join(s)
split_qs = re.split('[\n][\d]+\.', s, flags=re.M | re.DOTALL)

for i in range(len(split_qs)):
    t = split_qs[i] 
    t = re.sub('[\s]+', ' ', t)
    split_qs[i] = t

with open('temp2.csv', 'w') as f:
   r = csv.writer(f)
   r.writerows([re.sub('[\s]+', ' ', q.strip().replace('\n', ' '), flags=re.M | re.DOTALL) ] for q in split_qs)