import codecs
import csv
import re

from utils import clean
import pickle
import nsquared as Nsq
from nsquared import DocumentClassifier
'''
#Dont use them
import lda
import lsa
'''
mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}

relabelType = 'ADA'


X = []
Y_cog = []
Y_know = []
if relabelType == 'ADA':
    with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[1:])
        # NOTE: This is used to skip the first line containing the headers
        for row in csvreader:
            sentence, label_cog, label_know = row
            m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
            sentence = m.groups()[2]
            label_cog = label_cog.split('/')[0]
            label_know = label_know.split('/')[0]
            # sentence = clean(sentence)
            # NOTE: Commented the above line because the cleaning mechanism is different for knowledge and cognitive dimensions
            X.append(sentence)
            Y_cog.append(mapping_cog[label_cog])
            Y_know.append(mapping_know[label_know])


    count = 0

    #classifier = pickle.load(open('models/Nsquared/%s/nsquared_66_new.pkl' % ('ADA', ), 'rb'))
    y_pred = [0.92736833655244499, 0.91424978943320601, 0.78569820773207288, 0.67431296570148858, 0.73805242676846394, 0.68854087843729517, 0.44322363803494536, 0.48970977528788373, 0.78858364806760339, 0.63742545959589691, 0.46413279293619514, 0.70537808267142399, 0.70531800938384681, 0.82365200901107716, 0.53868358976718045, 0.26304618448705352, 0.57727320693933015, 0.78859577238805512, 0.78340123574952691, 0.77344286483000968, 0.79936197643507323, 0.68521890106375127, 0.89583793692293168, 0.76114324439529268, 0.58805628736020232, 0.71919271172649424, 0.90984384985227074, 0.68854803730881486, 0.65740146352608519, 0.7804434592829258, 0.65170572420278716, 0.76452769770549467, 0.765673141278296, 0.53280671332649787, 0.71217680991290089, 0.47563468008412807, 0.74561887778412372, 0.78631666751585727, 0.77816278227811175, 0.66973004043128803, 0.74393697461861708, 0.6323515047191155, 0.41964890118924864, 0.69905526939617812, 0.49740125704529486, 0.92429770125276756, 0.90202520926550422, 0.81532458354681059, 0.92449663725057474, 0.91753345983100865, 0.20254525634547105, 0.92022810636377728, 0.4364372538249941, 0.86506018092851089, 0.84576106135168794, 0.89215252773572618, 0.74408627563546437, 0.81245835618914453, 0.80947383280982232, 0.79242433540143498, 0.77155290798742993, 0.65430059633827187, 0.77740248977013926, 0.79155525735214927, 0.91241529517792663, 0.93347969524765284, 0.66339977595592237, 0.89265010519319821, 0.86633959673515892, 0.86651503702891508, 0.8894311215025883, 0.75993138089162215, 0.91865298560016684, 0.55012136707506765, 0.83660537032681292, 0.71104016294748806, 0.82576573283886834, 0.78341858122597907, 0.43006705281416047, 0.3807648891034906, 0.76692220350040807, 0.94008848271707424, 0.060355916547893268, 0.23808206966769424, 0.38275854725098518, 0.23518617363483249, 0.53998576049559077, 0.36170762376899085, 0.076340669808929049, 0.43486924362451834, 0.13136456878880928, 0.050802357918451738, 0.53025328877854805, 0.27712991382522967, 0.23024464059739264, 0.075478566237803621, 0.22981688826928623, 0.41876709896102299, 0.59130975506808836, 0.75256720472727801, 0.58474269543190505, 0.76417127485441128, 0.61528775920738699, 0.68306231577671406, 0.78016113306205592, 0.85023979885001844, 0.76501751713678989, 0.70899568473718411, 0.46974048575125654, 0.17514253032889512, 0.36737264899576522, 0.0, 0.45242437868815477, 0.72882390513879569, 0.49038399362612284, 0.78096974937943686, 0.42932776093817132, 0.32673809372761947, 0.42413407882499943, 0.8152690194065666, 0.14382180394727132, 0.52413314222573371, 0.95122537045193822, 0.063891010390768235, 0.12211394722098209, 0.073072960484893604, 0.69992350848087881, 0.78889359278734983, 0.32259016725706674, 0.22710005283478046, 0.71718596229831866, 0.62484593045782, 0.24755930998328798, 0.65517286820347731, 0.11795354703708634, 0.038340783350991899, 0.1365606699470126, 0.063124501859725679, 0.70220495519273407, 0.12590403880266421, 0.096025726957890453, 0.19427095458542087, 0.41876776324488196, 0.35454360845843913, 0.18614676508998293, 0.28358210441675719, 0.54336112372830847, 0.048688615087368554, 0.81634415640617508, 0.15704444161291878, 0.6932242375495028, 0.35035146006730783, 0.32159547044480952, 0.083122787416125318, 0.76667277193161631, 0.65265435150946449, 0.44749033253105308, 0.32015841245428189, 0.41452200400985634, 0.28409532400500231, 0.4344823429428562, 0.50609302553883517, 0.46093649725912561, 0.40266266475754303, 0.1700058593022036, 0.94742377991826199, 0.96053800619393337, 0.13265826714259293, 0.14960815686896353, 0.32630076331352581, 0.82373676210446833, 0.54143657988194482, 0.13709451497289515, 0.93760703993770789, 0.44302633813328235, 0.16675601798179779, 0.30434660844577088, 0.13370794738322869, 0.76324991929827912, 0.67977196405790152, 0.228415375969897, 0.60548474076067682, 0.12292023343866708, 0.8097371083715027, 0.8006196500587609, 0.20572634595454184, 0.38177280866045571, 0.66526841963629368, 0.12192268273320389, 0.91000069444806764, 0.84089599010864402, 0.57022960397327382, 0.62828720202069965, 0.11047402437628755, 0.10422988799773356, 0.043486504737561577, 0.86473303040988769, 0.11247405684667335, 0.83746627751878944, 0.97042630816967956, 0.81900961227332936, 0.782252111390071, 0.97306973853326217]
    with codecs.open('datasets/ADA_Exercise_Questions_Relabelled_v7.csv', 'w', encoding="utf-8") as csvfile:

        csvwriter = csv.writer(csvfile)
        #csvwriter.writerow(['Questions', 'Manual Label', 'NSQ', 'LDA', 'LSA', 'Knowledge', 'Cognitive'])
        for i, t in enumerate(zip(X, Y_cog, Y_know)):
            (x, y_cog, y_know) = t
            #print(x)

            nsq = y_pred[i]
            #lda_label = max(lda.get_vector('n', x, 'tfidf', subject_param = 'ADA')[1])
            lda_label = 1
            #lsa_label = lsa.get_values(x, subject_param = 'ADA')
            lsa_label = 1
            csvwriter.writerow([x, y_cog + 6 * y_know, nsq, lda_label, lsa_label, y_know, y_cog])
            count += 1
            if(count % 10 == 0):
                print(count)

elif relabelType == 'OS':
    with codecs.open('datasets/OS_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[5:])
        # NOTE: This is used to skip the first line containing the headers
        for row in csvreader:
            #sentence, label_cog, label_know = row
            #m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
            #sentence = m.groups()[2]
            #label_cog = label_cog.split('/')[0]
            #label_know = label_know.split('/')[0]
            # sentence = clean(sentence)
            # NOTE: Commented the above line because the cleaning mechanism is different for knowledge and cognitive dimensions
            
            X.append(row[0])
            if(row[6] == '' and row[4] == ''):      #Following Mohit > Shiva > Shrey
                label_cog = row[2].split('/')[0]
                Y_cog.append(mapping_cog[label_cog.strip()])
            elif(row[6] == '' and row[4] != ''):
                label_cog = row[4].split('/')[0]
                Y_cog.append(mapping_cog[label_cog.strip()])
            else:
                label_cog = row[6].split('/')[0]
                Y_cog.append(mapping_cog[label_cog.strip()])
            
            if(row[5] == '' and row[3] == ''):
                label_know = row[1].split('/')[0]
                Y_know.append(mapping_know[label_know.strip()])
            elif(row[5] == '' and row[3] != ''):
                label_know = row[3].split('/')[0]
                Y_know.append(mapping_know[label_know.strip()])
            else:
                label_know = row[5].split('/')[0]
                Y_know.append(mapping_know[label_know.strip()])
            
    count = 0      

    classifier = pickle.load(open('models/Nsquared/%s/nsquared.pkl' % ('OS', ), 'rb'))  

    with codecs.open('datasets/OS_Exercise_Questions_Relabelled_v1.csv', 'w', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        #csvwriter.writerow(['Questions', 'Manual Label', 'NSQ', 'LDA', 'LSA', 'Knowledge', 'Cognitive'])
        for x, y_cog, y_know in zip(X, Y_cog, Y_know):
            #print(x)
            nsq = max(classifier.classify(x)[0])
            #lda_label = max(lda.get_vector('n', x, 'tfidf', subject_param = 'OS')[1])
            lda_label = 1
            #lsa_label = lsa.get_values(x, subject_param = 'OS')
            lsa_label = 1
            csvwriter.writerow([x, y_cog + 6 * y_know, nsq, lda_label, lsa_label, y_know, y_cog])
            count += 1
            if(count % 10 == 0):
                print(count)