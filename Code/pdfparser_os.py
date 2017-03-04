# Invocation: python pdfparser_os.py

import subprocess
import sys
import re
import os
import pprint
import csv
import itertools
from rake import Rake

CHAPTER_MODE, SECTION_MODE = 0, 1

CHAPTER_PATTERN = r'(?:^|\n)[ ]*([\d.]+)?[ ]*(%s)[ ]*([\d]+)[ ]*(?:$|\n)'

split_mode = SECTION_MODE

pdfname = 'OS'
first_chapter = 'Introduction'
last_chapter = 'Appendix A'

process = subprocess.Popen('pdftotext -table -fixed 0 -f 15 -clip resources/%s -' %(pdfname + '.pdf'), shell=True, stdout=subprocess.PIPE)
text, _ = process.communicate()

print 'Parsing Done.'

pages = ('\n'.join(line for line in text.split('\n') if line != '')).split('\f')

chapter_tree = []
index = {}

index_text = '\n'.join(pages[:6])
topics = re.findall('((?:Chapter[\s]+.*?)|(?:Appendix[\s]+.*?))\n', index_text, flags=re.M | re.DOTALL)


for start, end in zip(topics[:-1], topics[1:]):
	if last_chapter in re.sub(' +', ' ', start):
		break

	chapter_tree.append({ 	'sno' 	:	re.search('\d+', start, flags=re.M).group(),
							'level'	:	1,
							'title'	:	re.sub(' +', ' ', re.split('Chapter[\s]+[\d]+', start)[1].strip(), flags=re.M),
							'pno' 	:	None })


	cur_chapter_topics = ([{'sno' 	:	match[0],
							'level' : 	2,
							'title' : re.sub(r'([()])', r'\\\1', re.sub('[\s]+', ' ', match[1].strip())),
							'pno' 	: int(match[2])}
		for match in re.findall('([\d]+\.[\d]+)[\s]+([a-zA-Z -]+)[\s]+([\d]+)', index_text.split(start)[1].split(end)[0], flags=re.M) ])
	chapter_tree[-1]['pno'] = cur_chapter_topics[0]['pno']
	chapter_tree.extend(sorted(cur_chapter_topics, key=lambda x: x['pno']))

chapter_tree += [{'sno' : '[\d]*', 'level' : -1, 'title' : 'Exercises', 'pno' : chapter_tree[-1]['pno'] + 1}] # to ensure the last section is processed as per my current logic

try:
	os.stat('resources/' + pdfname)
except:
	os.mkdir('resources/' + pdfname)
finally:
	os.chdir('resources/' + pdfname)

unigram_rake = Rake('../stopwords.txt', 3, 1, 3)
bigram_rake = Rake('../stopwords.txt', 3, 2, 3)
trigram_rake = Rake('../stopwords.txt', 3, 3, 2)

keywords = set()
if split_mode == CHAPTER_MODE:
	chapter_tree = filter(lambda x: x['level'] == 1, chapter_tree)

pages = pages[40:]

skip = []

preprocessed_sections = []
for i, (cur_topic, next_topic) in enumerate(zip(chapter_tree[:-1], chapter_tree[1:])):
	if next_topic['level'] != -1:
		section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : int(next_topic['pno'])]).replace('CHAPTER', '')
	else:
		section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : ]).replace('CHAPTER', '')

	if cur_topic['title'] in chapter_tree[i - 1]['title'] or next_topic['title'] in chapter_tree[i - 1]['title']:
			section_text = re.sub(chapter_tree[i - 1]['title'], '', section_text, flags=re.M | re.DOTALL)

	try:
		section_text = re.search(r'[\s\d.]*%s(.*?)[\s\d.]*%s' %(r'(?:[\s-]+)'.join(re.split('[ -]', cur_topic['title'])), r'(?:[\s-]+)'.join(re.split('[ -]', next_topic['title']))), section_text, re.M | re.DOTALL).group(1)
	except:
		skip.append(i + 1)
		continue

	section_text = re.sub('[\s]+', ' ', re.sub('[^a-z\s.!?\']', ' ', re.sub('-\n', '', re.sub('\n[\s]*', '\n', section_text.lower(), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL)

	with open(str(cur_topic['pno']) + ' ' + cur_topic['title'] + '.txt', 'w') as f:
		f.write(section_text)

	preprocessed_sections.append(section_text)

	keywords = keywords.union(map(lambda x: x[0], unigram_rake.run(section_text))).union(map(lambda x: x[0], bigram_rake.run(section_text))).union(map(lambda x: x[0], trigram_rake.run(section_text)))

print >> open('__Keywords.txt', 'w'), '\n'.join(list(keywords))

with open('__Sections.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Id', 'Section No.', 'Level', 'Section', 'Page No.'])

	writer.writerows([[i, chapter_tree[i]['sno'], chapter_tree[i]['level'], chapter_tree[i]['title'], chapter_tree[i]['pno']] for i in range(len(chapter_tree[:-1])) if i not in skip ])

with open('__OS-train.txt', 'w') as f:
	f.write('\n'.join(preprocessed_sections))

print len(preprocessed_sections)
os.chdir('..')
