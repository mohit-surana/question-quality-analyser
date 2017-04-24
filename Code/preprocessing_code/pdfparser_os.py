# Invocation: python pdfparser_os.py
import csv
import os
import re
import subprocess
import sys

from rake import Rake

CHAPTER_MODE, SECTION_MODE = 0, 1
split_mode = SECTION_MODE

if len(sys.argv) < 2:
    pdfname = 'OS'
else:
    pdfname = sys.argv[1]

def get_output(str_command):
    process = subprocess.Popen(str_command, shell=True, stdout=subprocess.PIPE)
    text, _ = process.communicate()

    try:
        text = text.decode('utf-8')
    except:
        pass

    return text

    
text = get_output('pdftotext -table -fixed 0 -clip ../resources/%s -' %(pdfname + '.pdf'))
print('Parsing Done.')

toC = get_output('mutool show ../resources/%s outline' %(pdfname + '.pdf'))

if pdfname == 'OS':
    search_str = r'PART[\s]+1[\s]+BACKGROUND[\s]+#29(.*?)APPENDICES[\s]+#735'
elif pdfname == 'OS2':
    search_str = r'PREFACE[\s]+#24,16,753(.*?)13.2[\s]+ALPHABETICAL[\s]+BIBLIOGRAPHY[\s]+#1072'
elif pdfname == 'OS3':
    search_str = r'PART[\s]+ONE[\s]+OVERVIEW[\s]+#25,0,547(.*?)Credits[\s]+#933,0,559'
elif pdfname == 'OS4':
    search_str = r'Part[\s]+1:[\s]+Overview[\s]+#22(.+)'

toC = re.search(search_str, toC, flags=re.M | re.DOTALL).group(1)
#print(toC)
chapter_tree = []
for line in list(map(str.strip, toC.split('\n'))):
    if re.search('Chapter', line) or '.' not in line:
        if pdfname == 'OS' or pdfname == 'OS3':
            match = re.search('Chapter[\s]*([\d]+)(.*?)#([\d]+)', line)
        elif pdfname == 'OS2':
            match = re.search('([\d]+)(.*?)#([\d]+)', line)
        elif pdfname == 'OS4':
            match = re.search('Chapter[\s]*([\d]+):(.*?)#([\d]+)', line)

        if match:
            title = match.group(2).strip()
            chapter_tree.append({'sno' : match.group(1).strip(), 'level' : 1, 'title' : title, 'pno': int(match.group(3).strip())})
    else:
        if pdfname == 'OS2' and re.search('([\d]+\.[\d]+\.[\d]+)+[\s]+(.*?)#([\d]+)', line):
            continue

        match = re.search('([\d]+\.[\d]+)+[\s]+(.*?)#([\d]+)', line)

        if match:
            title = re.sub(r'[^a-zA-Z0-9\-,/.?!\\\s:;\'"&$#@()\[\]{}]', '',  match.group(2).strip())
            if pdfname == 'OS4':
                title = title.replace('Operatin of', 'Operation of') \
                             .replace('System Performance', 'System Performance,') \
                             .replace('Operating of', 'Operation of') \
                             .replace('Sturcture', 'Structure') \
                             .replace('Implenting', 'Implementing') \
                             .replace('Synchronizatin', 'Synchronization') \
                             .replace('Concurent', 'Concurrent') \
                             .replace('Polices', 'Policies') \
                             .replace('Ressource', 'Resource') \
                             .replace('Resource Deadlock by', 'Resource Deadlocks by') \
                             .replace('Operting', 'Operating') \
                             .replace('Polocies', 'Policies') \
                             .replace('Memory-Maped', 'Memory-Mapped') \
                             .replace('Files Organizations and Acces', 'File Organizations and Access') \
                             .replace('Realiability', 'Reliability') \
                             .replace('Imput-Output Control Slystem', 'Input-Output Control System') \
                             .replace('Devece', 'Device') \
                             .replace('Sisk', 'Disk') \
                             .replace('Authentical', 'Authentication') \
                             .replace('Design Issues of Distributed', 'Design Issues in Distributed') \
                             .replace('Notion of Time and Date', 'Notions of Time and State') \
                             .replace('Time, Clock, and Event Precedences', 'Time, Clocks, and Event Precedences') \
                             .replace('Electio Algorithms', 'Election Algorithms') \
                             .replace('Authentics', 'Authentication')
            elif pdfname == 'OS3':
                title = title.replace('Petersons Solution', 'Peterson\'s Solution')

            chapter_tree.append({'sno' : match.group(1).strip(), 'level' : 2, 'title' : title, 'pno': int(match.group(3).strip())})

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
pages = ['\n'.join(['\n'] + p.split('\n')[:]) for p in pages]

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
        section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : int(next_topic['pno']) - 1 ])
    else:
        section_text = '\n'.join(pages[int(cur_topic['pno']) - 1 : ])

    if cur_topic['title'] in chapter_tree[i - 1]['title'] or next_topic['title'] in chapter_tree[i - 1]['title']:
        section_text = re.sub(chapter_tree[i - 1]['title'], '', section_text, flags=re.M | re.DOTALL)

    section_text2 = re.sub('[\s]+', ' ', re.sub('[^a-z\s.!?\']', ' ', re.sub('-\n', '', re.sub('\n[\s]*', '\n', section_text.lower(), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL)

    if split_mode == SECTION_MODE:
        t1 = cur_topic['title']
        t2 = next_topic['title']

        if pdfname == 'OS' or pdfname == 'OS2':
            t1 = t1.upper()
            t2 = t2.upper()
            
        elif pdfname == 'OS4':
            t1 = cur_topic['title'] if cur_topic['level'] == 1 else cur_topic['title'].upper()
            t2 = next_topic['title'] if next_topic['level'] == 1 else next_topic['title'].upper()

        try:
            section_text = re.split(t1.replace(' ', '[\s]*?') + '.*?\n', section_text, flags=re.M | re.DOTALL)[1]
        except:
            if pdfname == 'OS3' and '3.6' == cur_topic['sno'] or cur_topic['level'] == 1:
                pass

        section_text = re.split('\n[\s]*([\d]+\.[\d]+[\s]+)?' + t2.replace(' ', '[\s]+'),  section_text, flags=re.M | re.DOTALL)[0]
            
    with open(str(cur_topic['pno']) + '.txt', 'w') as f:
        f.write(cur_topic['title'] + '\n')
        f.write(section_text)
        print(cur_topic)

    #keywords = keywords.union(map(lambda x: x[0], unigram_rake.run(section_text2))).union(map(lambda x: x[0], bigram_rake.run(section_text2))).union(map(lambda x: x[0], trigram_rake.run(section_text2)))

#print >> open('__Keywords.txt', 'w'), '\n'.join(list(keywords))

with open('__Sections.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Section No.', 'Level', 'Section', 'Page No.'])
    writer.writerows([[chapter_tree[i]['sno'], chapter_tree[i]['level'], chapter_tree[i]['title'], chapter_tree[i]['pno']] for i in range(len(chapter_tree[:-1]))])
