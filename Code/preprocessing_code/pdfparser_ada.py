# Invocation: python pdfparser_ada.py

import csv
import os
import re
import subprocess

from rake import Rake

CHAPTER_MODE, SECTION_MODE = 0, 1

CHAPTER_PATTERN = r'(?:^|\n)[ ]*([\d.]+)?[ ]*(%s)[ ]*([\d]+)[ ]*(?:$|\n)'

split_mode = SECTION_MODE
def get_chapters(text):
	return [{'sno' :match[0],
			'level' : 3 if match[0] == ''
				else 1 if '.' not in match[0]
				else 2 if match[0].index('.') == match[0].rindex('.')
				else 3,
			'title' : re.sub(r'([()])', r'\\\1', re.sub('[\s]+', ' ', match[1].strip())),
			'pno' : int(match[2])}
		for match in re.findall(CHAPTER_PATTERN %(r'[^\d]+?'), '\n'.join(text.split('\n')[1 : -1]), re.M|re.DOTALL) if match[1].strip()[0].isalnum() and 'Summary' not in match[1]]

pdfname = 'ADA'
first_chapter = 'Introduction'
last_chapter = 'Epilogue'

process = subprocess.Popen('pdftotext -table -fixed 0 -clip ../resources/%s -' %(pdfname + '.pdf'), shell=True, stdout=subprocess.PIPE)
text, _ = process.communicate()

pages = ('\n'.join(line for line in text.decode('latin-1').split('\n') if line != '')).split('\f')
for i, page in enumerate(pages):
	if 'brief contents' not in page.lower() and ('contents' in page.lower() or 'table of contents' in page.lower()):
		break

chapter_tree = []
index = {}
for j, page in enumerate(pages[i:]):

	if re.search(CHAPTER_PATTERN %(last_chapter), page, re.M):
		chapter_tree += get_chapters(re.split(CHAPTER_PATTERN %(last_chapter), page, re.M)[0])
		index = { c['title'] : c['pno'] for c in chapter_tree }

		while not (re.match('%d' %index[first_chapter], pages[j + i].split('\n')[0].strip()) or re.match('%d' %index[first_chapter], pages[j + i].split('\n')[-1].strip())):
			j += 1
		pages = map(lambda x: '\n'.join(x.split('\n')[1:-1]), pages[(j + i) : ])
		break

	else:
		chapter_tree += get_chapters(page)

chapter_tree += [{'sno' : '[\d]*', 'level' : -1, 'title' : 'Exercises', 'pno' : chapter_tree[-1]['pno'] + 1}] # to ensure the last section is processed as per my current logic

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

with open('__Sections.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Section No.', 'Level', 'Section', 'Page No.'])
	writer.writerows([[c['sno'], c['level'], c['title'], c['pno']] for c in chapter_tree])

unigram_rake = Rake('../stopwords.txt', 3, 1, 3)
bigram_rake = Rake('../stopwords.txt', 3, 2, 3)
trigram_rake = Rake('../stopwords.txt', 3, 3, 2)

keywords = set()
if split_mode == CHAPTER_MODE:
	chapter_tree = list(filter(lambda x: int(x['level']) in [1, -1], chapter_tree))
else:
	chapter_tree = list(filter(lambda x: int(x['level']) in [1, 2, -1], chapter_tree))

preprocessed_sections = []
for i, (cur_topic, next_topic) in enumerate(zip(chapter_tree[:-1], chapter_tree[1:])):
	if next_topic['level'] != -1:
		section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : int(next_topic['pno'])])
	else:
		section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : ])

	if cur_topic['title'] in chapter_tree[i - 1]['title'] or next_topic['title'] in chapter_tree[i - 1]['title']:
			section_text = re.sub(chapter_tree[i - 1]['title'], '', section_text, flags=re.M | re.DOTALL)

	section_text = re.search(r'(?:%s[ ]*)?%s[\s]*(.*?)(?:%s[ ]*)?%s[\s]*' %(cur_topic['sno'], r'(?:[\s]+)'.join(cur_topic['title'].split()), next_topic['sno'], r'(?:[\s]+)'.join(next_topic['title'].split())), section_text, re.M | re.DOTALL).group(1)

	section_text2 = re.sub('[\s]+', ' ', re.sub('[^a-z\s.!?\']', ' ', re.sub('-\n', '', re.sub('\n[\s]*', '\n', section_text.lower(), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL)

	with open(str(cur_topic['pno']) + '.txt', 'w') as f:
		f.write(cur_topic['title'] + '\n')
		f.write(section_text)
	print('Processed %s' %cur_topic['title'])

	#keywords = keywords.union(map(lambda x: x[0], unigram_rake.run(section_text2))).union(map(lambda x: x[0], bigram_rake.run(section_text2))).union(map(lambda x: x[0], trigram_rake.run(section_text2)))

#print >> open('__Keywords.txt', 'w'), '\n'.join(list(keywords))
