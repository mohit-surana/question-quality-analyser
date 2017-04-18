import csv
import json
import logging
import os
import os.path
import random
import signal
import subprocess
import sys
import threading
import urllib
import zlib

import cStringIO
import pycurl


def handler(signum, frame):
    print('Signal handler called with signal', signum)
    os._exit(1)

logging.basicConfig()
signal.signal(signal.SIGINT, handler)

if not os.path.exists('proxies.txt'):
    with open('proxies.txt', 'w') as f:
        null = open(os.devnull, 'w')
        subprocess.Popen('python3.6 proxyfinder.py 1000 500', shell=True, stdout=f, stderr=null).communicate()

proxy_dict = json.loads(open('proxies.txt').read())
proxy_set = set()
for protocol in proxy_dict:
    proxy_set = proxy_set.union(set(proxy_dict[protocol]))

def discard_proxy(proxy):	
    proxy_set.discard(proxy)
    if (len(proxy_set))	% 50 == 0:
        print('====> Proxies left: %d' %(len(proxy_set)))

LOCK = threading.Lock()

COMMITTED_QIDS = set()

class ValThread(threading.Thread):
    def __init__(self, tid, keywords, thread_type='Normal'):
        threading.Thread.__init__(self)
        self.tid = tid
        self.keywords = { x[0] : x[1] for x in map(lambda x: eval(x), set(keywords)) }
        print(len(self.keywords))

        self.thread_type = thread_type
        self.proxy = None

        if thread_type == 'Tag':
            self.url = 'https://api.stackexchange.com/2.2/questions?%s'
            self.urlparams = { 'site'	: 'stackoverflow',
                            'page' 		: -1,
                            'pagesize'	: 100,
                            'order' 	: 'desc',
                            'sort' 		: 'creation',
                            'filter'	: '!-MOiNm40F1YLRluUDp7t.GuRzx*2eF9I0'}
        else:
            self.url = 'https://api.stackexchange.com/2.2/search?%s'
            self.urlparams = { 'site'	: 'stackoverflow',
                            'page' 		: -1,
                            'pagesize'	: 100,
                            'order' 	: 'desc',
                            'sort' 		: 'creation',
                            'filter'	: '!-MOiNm40F1YLRluUDp7t.GuRzx*2eF9I0'}

    def run(self):
        print('[Thread-%d]\t Starting' %self.tid)

        self.c = pycurl.Curl()
        self.c.setopt(pycurl.CONNECTTIMEOUT, 10)
        self.c.setopt(pycurl.TIMEOUT, 10)

        for i, keyword in enumerate(self.keywords):
            page_no = self.keywords[keyword]
            print('[Thread-%d]\t <%s (%d/%d)>' %(self.tid, keyword, i + 1, len(self.keywords)))
            questions = {}
            while True:
                page_no += 1
                self.urlparams['page'] = page_no

                if self.thread_type == 'Tag':
                    self.urlparams['tagged'] = '-'.join(keyword.split(' '))
                else:
                    self.urlparams['intitle'] = keyword

                self.c.setopt(pycurl.URL, self.url %(urllib.urlencode(self.urlparams)))

                t = self.__get_questions()

                if len(t) == 0:
                    break

                questions.update(t)

                if len(questions) > 400:
                    self.keywords[keyword] = page_no
                    self.__print_to_file(questions)
                    questions.clear()
                    continue

            if len(questions) > 0:
                self.keywords[keyword] = page_no
                self.__print_to_file(questions)

        print('[Thread-%d]\t Finished' %self.tid)

    def __print_to_file(self, questions):
        LOCK.acquire()
        print('[Thread-%d]\t Storing %d questions' %(self.tid, len(questions)))
        with open(path + '/__Questions.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            for qid, question in questions.items():
                if qid not in COMMITTED_QIDS:
                    writer.writerow([qid] + question)

        COMMITTED_QIDS.update(questions.keys())

        exhausted_keywords = json.loads(open(path + '/__ValidatedKeywords[Exhausted].txt').read())
        exhausted_keywords.update(self.keywords)
        with open(path + '/__ValidatedKeywords[Exhausted].txt', 'w') as f:
            json.dump(exhausted_keywords, f)
            LOCK.release()

    def __get_questions(self):
        questions = {}
        buf = cStringIO.StringIO()
        self.c.setopt(pycurl.WRITEFUNCTION, buf.write)
        while True:
            if self.proxy is None:
                try:
                    proxy = random.sample(proxy_set, 1)[0]
                except:
                    return {}
            else:
                proxy = self.proxy

            self.c.setopt(pycurl.PROXY, proxy)

            try:
                self.c.perform()
            except:

                error_msg = sys.exc_info()
                discard_proxy(proxy)
                continue

            try:
                response = json.loads(zlib.decompress(buf.getvalue(), 16 + zlib.MAX_WBITS))
            except:
                try:
                    response = json.loads(buf.getvalue())
                except:
                    discard_proxy(proxy)
                    continue

            if self.proxy is None:
                self.proxy = proxy

            if 'error_id' not in response.keys():
                questions.update({item['question_id'] : [item['title'].encode('utf-8'), item['score']] for item in response['items']})

            else:
                discard_proxy(proxy)
                self.proxy = None
                continue

            break

        buf.close()

        return questions

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Not enough arguments')

    num_threads = 64 if len(sys.argv) == 2 else max(5, int(sys.argv[2]))
    tag_threads = int(0.2 * num_threads)

    path = '' + sys.argv[1]


    with open(path + '/__ValidatedKeywords.txt', 'r') as f:
        KEYWORDS = list(set([k for k in f.read().split('\n') if k != '']))

    if not os.path.exists(path + '/__ValidatedKeywords[Exhausted].txt'):
        with open(path + '/__ValidatedKeywords[Exhausted].txt', 'w') as f:
            f.write(json.dumps({}))

    else:
        KEYWORDS = { x[0] : x[1] for x in map(lambda x: eval(x), KEYWORDS) }
        KEYWORDS.update(json.loads(open(path + '/__ValidatedKeywords[Exhausted].txt').read()))
        KEYWORDS = [str((k, v)) for k, v in KEYWORDS.items()]

    KEYWORDS = KEYWORDS[:]

    with open('Tags.csv', 'r') as f:
        reader = csv.reader(f)
        so_tags = sorted((line[0] for line in reader))

    TAGS = filter(lambda k: k in so_tags, KEYWORDS)
    KEYWORDS = filter(lambda k: k not in TAGS, KEYWORDS)

    threads = []
    if not os.path.exists(path + '/__Questions.csv'):
        with open(path + '/__Questions.csv', 'w') as f:
            csv.writer(f).writerow(['Question ID', 'Title', 'Score'])
    else:
        COMMITTED_QIDS = set(map(lambda x: x.split(',')[0], open(path + '/__Questions.csv', 'r').read().split('\r\n')))

    print('Initialised qbscraper')

    for chunk in [TAGS[i::tag_threads] for i in xrange(tag_threads)]: # for tags only
        threads.append(ValThread(len(threads), chunk, thread_type='Tag'))
        threads[-1].start()

    for chunk in [KEYWORDS[i::num_threads - tag_threads] for i in xrange(num_threads - tag_threads)]: # for the remaining keywords
        threads.append(ValThread(len(threads), chunk))
        threads[-1].start()

    for i, thread in enumerate(threads):
        thread.join()

