import csv
import os


def convert_2darray_to_list(array):
    l = []
    for x in array:
        l.append(convert_1darray_to_list(x))
    return l


def convert_1darray_to_list(array):
    l = []
    for x in array:
        l.append(round(x, 2))
    return l


def load_csv(subject):
    question_details = []
    performance_details = []
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', subject + '_Performance.csv'), 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            question_details.append(row[0:3])
            performance_details.append(row[3:])
    return question_details, performance_details
