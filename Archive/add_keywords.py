import pickle

s = '''suggest,1
paraphrase,1
specify,2
appraise,4
restate,1
give,4
recite,0
check,3
throw,0
finding,2
check,3
specify,2
recite,0
check,2
defend,4
discuss,3
relate,2
give,0
organize,5
relate,2
defend,4
restate,1
participate,2
critique,4
discuss,4
discuss,4
discuss,1
specify,4
assess,4
critique,4
run,2
enumerate,0
follow,1
do,2
follow,2'''

cog2label = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
label2cog = { v : k for k, v in cog2label.items()}

domain = pickle.load(open('../Code/resources/domain.pkl', 'rb'))
for line in s.split('\n'):
	word, label = line.split(',')
	label = int(label)
	if(word not in domain[label2cog[label]]):
		domain[label2cog[label]].add(word)

pickle.dump(domain, open('../Code/resources/domain.pkl', 'wb'))
