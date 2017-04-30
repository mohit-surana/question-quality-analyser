import csv

import openpyxl
from fuzzywuzzy import process

wb = openpyxl.load_workbook('ADA_Exercise_Questions_Labelled.xlsx', data_only=True)
sheet = wb["__Exercise_Questions_Labelled"]

no_of_rows = 205 # no_of_rows = sheet.max_row # doesn't work for some reason

labelled_rows = []
for row_no in range(1, no_of_rows):
    row = sheet[str(row_no)]
    question, cog_label, know_label = row[0].value, row[1].value, row[2].value
    if(row[1].fill.bgColor.value == 'FFFFFF00'):
        cog_label += '/'
    if(row[2].fill.bgColor.value == 'FFFFFF00'):
        know_label += '/'
    labelled_rows.append((question, cog_label, know_label))

unlabelled_questions = []
with open('ADA Questions.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        unlabelled_questions.append(row[0])

final_rows = []
labelled_questions = [x[0] for x in labelled_rows]
i = 1
for unlabelled_question in unlabelled_questions:
    print('Question number:', i)
    i += 1
    question, score = process.extract(unlabelled_question, labelled_questions, limit=1)[0]
    cog_label, know_label = '', ''
    if(score > 86):
        index = labelled_questions.index(question)
        cog_label, know_label = labelled_rows[index][1], labelled_rows[index][2]
    final_rows.append((unlabelled_question, cog_label, know_label))

with open('ADA Questions Labelled.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for x, y_cog, y_know in final_rows:
        csvwriter.writerow([x, y_know, y_cog])
