import os
import re
import csv
os.chdir('./resources/ADA')

exercise_content = []
for filename in sorted(os.listdir('.')):
	if '__' in filename or '.DS_Store' in filename:
		continue
	with open(filename) as f:
		contents = f.read()

		match = re.search(r'\n[\s]*Exercises[\s]+([\d]+\.[\d]+)[\s]*(.*)', contents, flags=re.M | re.DOTALL) 

		if match:
			exercise_content.append((int(match.group(1).replace('.', '')), '\n' + match.group(2).split('SUMMARY')[0]))

exercise_content = sorted(exercise_content, key=lambda x: x[0])

with open('__ADA_Questions.csv', 'w') as f:
	writer = csv.writer(f)
	for e in exercise_content:
		#print >> f, e[-1]
		for question in re.split('[\n][\s]*[\d]+\.', e[-1].strip(), flags=re.M | re.DOTALL):
			if len(question) > 0:
				writer.writerow([re.sub('1\.', '', question.strip())])