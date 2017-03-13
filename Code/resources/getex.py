import re
import sys
import subprocess
import csv

pdfname = 'ADA'

process = subprocess.Popen('pdftotext -table -fixed 0 -f 28 -clip %s -' %(pdfname + '.pdf'), shell=True, stdout=subprocess.PIPE)
text, _ = process.communicate() 

text = ('\n'.join(line for line in text.split('\n') if line != ''))
questions_text = map(lambda x: re.sub('\n[\s]+([\d]+\.)', '\n\1', x), \
		re.findall(r'Exercises [\d]+\.[\d]+[\s]*\n(.*?)(?:Summary|\f)', text, flags=re.M | re.DOTALL))

questions = []
for exercises in questions_text:
	questions.extend(re.findall(r'(?:\n|^)([\d]+\..*?)(?=(?:[\d]+\.))', exercises, flags=re.M | re.DOTALL))

with open(pdfname + '/__Exercise_Questions.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	for q in questions:
		writer.writerow([re.sub('[\s]{2,}', ' ', re.sub('\n +', '\n', q).replace('-\n', '').replace('\n', ' '))])
