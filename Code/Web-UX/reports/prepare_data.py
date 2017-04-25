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

    for _, know_label, cog_label in question_details:
        know_label_mapping = label2know[int(know_label)]
        for i in range(4):
            if know_freq[i]['name'] == know_label_mapping:
                know_freq[i]['y'] += 1
        cog_label_mapping = label2cog[int(cog_label)]
        for i in range(6):
            if cog_freq[i]['name'] == cog_label_mapping:
                cog_freq[i]['y'] += 1

    data['know_freq'] = know_freq
    data['cog_freq'] = cog_freq
    return data


'''
    { 'question_details_ada': question_details_ada, 'performance_details_ada': performance_details_ada,
    'question_details_os': question_details_os, 'performance_details_os': performance_details_os, }
'''