from . import utils

cog2label = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
label2cog = { v : k for k, v in cog2label.items()}

know2label = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}
label2know = { v : k for k, v in know2label.items()}


def get_data(subject):
    data = dict()
    question_details, performance_details = utils.load_csv(subject)
    know_freq = [{'name': label2know[x], 'y': 0} for x in range(4)]
    cog_freq = [{'name': label2cog[x], 'y': 0} for x in range(6)]
    know_perf = [{'type': 'column', 'name': 'Average Correct', 'data': [0] * 4, 'yAxis': 0}, {'type': 'column', 'name': 'Total', 'data': [0] * 4, 'yAxis': 0}, {'type': 'spline', 'name': '% Correct', 'data': [0] * 4, 'yAxis': 1}]
    cog_perf = [{'type': 'column', 'name': 'Average Correct', 'data': [0] * 6, 'yAxis': 0}, {'type': 'column', 'name': 'Total', 'data': [0] * 6, 'yAxis': 0}, {'type': 'spline', 'name': '% Correct', 'data': [0] * 6, 'yAxis': 1}]

    for _, know_label, cog_label in question_details:
        know_label_mapping = label2know[int(know_label)]
        for i in range(4):
            if know_freq[i]['name'] == know_label_mapping:
                know_freq[i]['y'] += 1
        cog_label_mapping = label2cog[int(cog_label)]
        for i in range(6):
            if cog_freq[i]['name'] == cog_label_mapping:
                cog_freq[i]['y'] += 1

    for q_details, perf in zip(question_details, performance_details):
        _, know_label, cog_label = q_details
        know_label, cog_label =  int(know_label), int(cog_label)
        corr = perf.count('Correct')
        tot = len(perf)
        know_perf[0]['data'][know_label] += 0 if tot == 0 else corr/tot
        know_perf[1]['data'][know_label] += 1
        cog_perf[0]['data'][cog_label] += 0 if tot == 0 else corr/tot
        cog_perf[1]['data'][cog_label] += 1
    for i in range(4):
        know_perf[2]['data'][i] = 0 if know_perf[1]['data'][i] == 0 else (know_perf[0]['data'][i] / know_perf[1]['data'][i]) * 100
    for i in range(6):
        cog_perf[2]['data'][i] = 0 if cog_perf[1]['data'][i] == 0 else (cog_perf[0]['data'][i] / cog_perf[1]['data'][i]) * 100

    data['know_freq'] = know_freq
    data['cog_freq'] = cog_freq
    data['know_perf'] = know_perf
    data['cog_perf'] = cog_perf
    return data


'''
    { 'question_details_ada': question_details_ada, 'performance_details_ada': performance_details_ada,
    'question_details_os': question_details_os, 'performance_details_os': performance_details_os, }
'''