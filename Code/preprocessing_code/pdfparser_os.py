# Invocation: python pdfparser_os.py
import csv
import os
import re
import subprocess

from rake import Rake

CHAPTER_MODE, SECTION_MODE = 0, 1
split_mode = SECTION_MODE

pdfname = 'OS'

process = subprocess.Popen('pdftotext -table -fixed 0 -clip ../resources/%s -' %(pdfname + '.pdf'), shell=True, stdout=subprocess.PIPE)
text, _ = process.communicate()

print('Parsing Done.')

toC = open('../resources/osToC.txt').read()
toC = re.search('PART 1 BACKGROUND       #29(.*?)APPENDICES      #735', toC, flags=re.M | re.DOTALL).group(1)

chapter_tree = []
for line in list(map(str.strip, toC.split('\n'))):
	if re.search('Chapter', line):
		match = re.search('Chapter[\s]*([\d]+)(.*?)#([\d]+)', line)
		chapter_tree.append({'sno' : match.group(1), 'level' : 1, 'title' : match.group(2).strip(), 'pno': int(match.group(3))})
	else: 
		match = re.search('([\d]+\.[\d]+)+[\s]+(.*?)#([\d]+)', line)
		if match:
			chapter_tree.append({'sno' : match.group(1), 'level' : 2, 'title' : match.group(2).strip(), 'pno': int(match.group(3))})

chapter_tree += [{'sno' : '[\d]*', 'level' : -1, 'title' : 'Exercises', 'pno' : chapter_tree[-1]['pno'] + 1}] # to ensure the last section is processed as per my current logic

print('Chapter tree built.')

if split_mode == CHAPTER_MODE:
	dirname = pdfname + '[chapters]'
else:
	dirname = pdfname	

try:
	os.stat('../resources/' + dirname)
except:
	os.mkdir('../resources/' + dirname)
finally:
	os.chdir('../resources/' + dirname)

pages = ('\n'.join(line for line in text.decode('latin-1').split('\n') if line != '')).split('\f')
pages = ['\n'.join(p.split('\n')[1:]) for p in pages]

unigram_rake = Rake('../stopwords.txt', 3, 1, 3)
bigram_rake = Rake('../stopwords.txt', 3, 2, 3)
trigram_rake = Rake('../stopwords.txt', 3, 3, 2)

keywords = set()
if split_mode == CHAPTER_MODE:
	chapter_tree = list(filter(lambda x: int(x['level']) in [1, -1], chapter_tree))
else:
	chapter_tree = list(filter(lambda x: int(x['level']) in [1, 2, -1], chapter_tree))


for i, (cur_topic, next_topic) in enumerate(zip(chapter_tree[:-1], chapter_tree[1:])):
	if next_topic['level'] != -1:
		section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : int(next_topic['pno'])])
	else:
		section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : ])

	if cur_topic['title'] in chapter_tree[i - 1]['title'] or next_topic['title'] in chapter_tree[i - 1]['title']:
			section_text = re.sub(chapter_tree[i - 1]['title'], '', section_text, flags=re.M | re.DOTALL)

	section_text2 = re.sub('[\s]+', ' ', re.sub('[^a-z\s.!?\']', ' ', re.sub('-\n', '', re.sub('\n[\s]*', '\n', section_text.lower(), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL)
	
	section_text = re.split(cur_topic['title'].upper().replace(' ', '[\s]*?'), section_text, flags=re.M | re.DOTALL)[1]
	section_text = re.split('([\d]+\.[\d]+[\s]+)?' + next_topic['title'].upper().replace(' ', '[\s]+'), section_text, flags=re.M | re.DOTALL)[0]

	with open(str(cur_topic['pno']) + '.txt', 'w') as f:
		f.write(cur_topic['title'] + '\n')
		f.write(section_text)

	#keywords = keywords.union(map(lambda x: x[0], unigram_rake.run(section_text2))).union(map(lambda x: x[0], bigram_rake.run(section_text2))).union(map(lambda x: x[0], trigram_rake.run(section_text2)))

#print >> open('__Keywords.txt', 'w'), '\n'.join(list(keywords))

with open('__Sections.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Section No.', 'Level', 'Section', 'Page No.'])
	writer.writerows([[chapter_tree[i]['sno'], chapter_tree[i]['level'], chapter_tree[i]['title'], chapter_tree[i]['pno']] for i in range(len(chapter_tree[:-1]))])
