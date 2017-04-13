
import sys
import copy
import re
import requests
import threading
import itertools
import pprint
from nltk.stem.wordnet import WordNetLemmatizer
import signal

def handler(signum, frame):
    print('Signal handler called with signal', signum)
    os._exit(1)

signal.signal(signal.SIGINT, handler)

lock = threading.Lock()

class ValThread(threading.Thread):
	def __init__(self, tid, keywords):
		threading.Thread.__init__(self)
		self.tid = tid
		self.keywords = keywords
		self.valid_keywords = []
	
	def run(self):
		try:
			self.valid_keywords = list(keyword for keyword in self.keywords if is_valid(keyword))
		except:
			pass

		lock.acquire()
		with open(path + '/__ValidatedKeywords.txt', 'a') as f:
			print >> f, '\n'.join([ str((x, 0)) for x in self.valid_keywords])

		lock.release()

def is_valid(phrase):
	print('Validating', phrase)
	url = 'http://lookup.dbpedia.org/api/search.asmx/KeywordSearch?QueryString=%s'
	response = requests.get(url % phrase, 
			headers={'Accept': 'application/json'}).json()
	
	if len(response['results']) == 0:
		return False

	for result in response['results']:
		#if result['description'] != None and re.search('comput(?:ing|er science)|algo(?:rithm|rithmic)?|programming(?: language)?', result['description'].lower(), flags=re.M):
		if result['description'] != None and re.search(r'operating system(?:s)?|\bos\b', result['description'].lower(), flags=re.M):
		
				return True

	return False

if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit('Not enough arguments')

	num_threads = 16 if len(sys.argv) == 2 else min(64, max(16, int(sys.argv[2])))

	path = sys.argv[1]

	ngrams = { 'trigrams' : set(), 'bigrams' : set(), 'unigrams' : set() } 

	lemmatizer = WordNetLemmatizer()
	with open(path + '/__Keywords.txt') as f:
		for line in f:
			if not re.match('[a-z]', line):
				continue
			words = [w for w, g in itertools.groupby(line.rstrip('\n').split())]
			words.append(lemmatizer.lemmatize(words.pop()))
			line = ' '.join(words)

			if len(words) == 3:
				ngrams['trigrams'].add(line)

			elif len(words) == 2:
	 			ngrams['bigrams'].add(line)

			elif len(words) == 1:
				ngrams['unigrams'].add(line)

	# filter out duplicates and redundant keywords
	searchspace = ' '.join(list(ngrams['trigrams']))
	for bigram in copy.copy(ngrams['bigrams']):
		if bigram in searchspace:
			ngrams['bigrams'].discard(bigram)

	searchspace = ' '.join(list(ngrams['trigrams']) + list(ngrams['bigrams']))
	for unigram in copy.copy(ngrams['unigrams']):
		if unigram in searchspace:
			ngrams['unigrams'].discard(unigram)

	keywords = sorted(list(ngrams['unigrams']) + list(ngrams['bigrams']) + list(ngrams['trigrams']))

	valid_keywords = []
	threads = []

	with open(path + '/__ValidatedKeywords.txt', 'w') as f:
		pass

	for chunk in [keywords[i::num_threads] for i in xrange(num_threads)]:
		threads.append(ValThread(len(threads), chunk))
		print('Starting Thread-' + str(threads[-1].tid))
		threads[-1].start()

	for i, thread in enumerate(threads):
		thread.join()
		print('Thread-' + str(thread.tid), 'finished')
		valid_keywords.extend(thread.valid_keywords)
