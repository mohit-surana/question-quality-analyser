import numpy as np

from tkinter import *

from brnn        import BiDirectionalRNN, sent_to_glove, clip
from svm_glove   import TfidfEmbeddingVectorizer
from maxent      import features
from cogvoter    import predict_cog_label, get_cog_models

from nsquared    import DocumentClassifier
from nsquared_v2 import predict_know_label, get_know_models

from utils       import get_modified_prob_dist

subject = 'ADA'
    
class Visualize:

    def __init__(self):
        self.array1 = []
        self.array2 = []
        self.know_models = get_know_models(subject)
        print('[Visualize] Knowledge models loaded')

        self.cog_models = get_cog_models()
        print('[Visualize] Cognitive models loaded')

    def get_probabilities(self, root, question):  # make a call to Shiva and Mohit to get required arrays: send question
        question = question.get()

        level_know, prob_know = predict_know_label(question, self.know_models)
        array_know = get_modified_prob_dist(prob_know)

        level_cog, prob_cog = predict_cog_label(question, self.cog_models, subject)
        array_cog = get_modified_prob_dist(prob_cog)

        

        print(question)
        print(array_know)
        print(array_cog)
        self.populate_table(root, array_know, array_cog)

    def populate_table(self, root, array_know, array_cog):
        nmarray = np.dot(np.array(array_know).reshape(-1, 1), np.array(array_cog).reshape(1, -1))
        print(nmarray)
        frame = Frame(root)
        Grid.rowconfigure(root, 0, weight=1)
        Grid.columnconfigure(root, 0, weight=1)
        frame.grid(row=0, column=0, sticky=N+S+E+W)
        grid = Frame(frame)
        grid.grid(sticky=N+S+E+W, column=7, row=7)
        Grid.rowconfigure(frame, 0, weight=1)
        Grid.columnconfigure(frame, 0, weight=1)
        cog_values = ['', '', 'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
        know_values = ['', '', 'Factual Knowledge', 'Conceptual Knowledge', 'Procedural Knowledge', 'Metacognitive Knowledge']

        myWhite = '#%02x%02x%02x' % (255, 255, 255)  # set your favourite rgb color
        myTitleColor = '#%02x%02x%02x' % (255, 230, 190)  # set your favourite rgb color
        
        maxVal = np.max(nmarray)

        for x in range(1,6):
            for y in range(1,8):
                if x == 1 or y ==1:
                    if x ==1 and y ==1:
                        l = Label(frame, text=know_values[x], relief=RIDGE, bg = myTitleColor)
                    elif y == 1:
                        l = Label(frame, text=know_values[x], relief=RIDGE, bg= myTitleColor)
                    else:
                        l = Label(frame, text=cog_values[y], relief=RIDGE, bg= myTitleColor)
                else:
                    var = nmarray[x-2, y-2] * 255
                    if(var != 0):
                        if nmarray[x - 2, y - 2] == maxVal:
                            myCellColor = '#%02x%02x%02x' % (255 - int(var)//2, 0, 0)  # set your heatmap color
                        else:
                            myCellColor = myWhite
                        l = Label(frame, text='{:.2f}'.format(nmarray[x - 2][y - 2]), relief=RIDGE, bg= myCellColor)
                        
                    else:
                        myCellColor = '#%02x%02x%02x' % (255, 255,255)  # set it white
                        l = Label(frame, text='', relief=RIDGE, bg= myCellColor)
                l.grid(row=x, column=y, sticky=N+S+E+W)



        for x in range(1,6):
            # Grid.columnconfigure(frame, x, weight=1, uniform=1)
            Grid.rowconfigure(frame, x, weight=1, uniform=1)

        for y in range(1,8):
            # Grid.rowconfigure(frame, y, weight=1, uniform=1)
            Grid.columnconfigure(frame, y, weight=1, uniform=1)


        button = Button(frame, text='Refresh', command= lambda: self.create_table(root)).grid(row=6, column=4,pady=20, sticky=N+S+E+W)
        knowledge_label = Label(frame, text='Knowledge Dimension\n[[{:.2f} {:.2f}]\n[{:.2f} {:.2f}]]'.format(array_know[0], array_know[1], array_know[2], array_know[3])).grid(row=3,column=0, rowspan=3, sticky=N+S+E+W)
        cognitive_label = Label(frame, text='Cognitive Dimension\n[' + str(array_cog) + ']').grid(row=0,column=4, columnspan=2, sticky=N+S+E+W)

        root.mainloop()

    def create_table(self, root):
        frame = Frame(root)
        Grid.rowconfigure(root, 0, weight=1)
        Grid.columnconfigure(root, 0, weight=1)
        frame.grid(row=0, column=0, sticky=N+S+E+W)
        grid = Frame(frame)
        grid.grid(sticky=N+S+E+W, column=7, row=7)
        Grid.rowconfigure(frame, 0, weight=1)
        Grid.columnconfigure(frame, 0, weight=1)
        cog_values = ['','', 'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
        know_values = ['', '', 'Factual Knowledge', 'Conceptual Knowledge', 'Procedural Knowledge', 'Metacognitive Knowledge']

        myWhite = '#%02x%02x%02x' % (255, 255, 255)  # set your favourite rgb color
        myTitleColor = '#%02x%02x%02x' % (255, 230, 190)  # set your favourite rgb color

        for x in range(1,6):
            for y in range(1,8):
                if x == 1 or y ==1:
                    if x ==1 and y ==1:
                        l = Label(frame, text=know_values[x], relief=RIDGE, bg = myTitleColor)
                    elif y == 1:
                        l = Label(frame, text=know_values[x], relief=RIDGE, bg= myTitleColor)
                    else:
                        l = Label(frame, text=cog_values[y], relief=RIDGE, bg= myTitleColor)
                else:
                    l = Label(frame, text='', relief=RIDGE, bg= myWhite)
                l.grid(row=x, column=y, sticky=N+S+E+W)



        for x in range(1,6):
            # Grid.columnconfigure(frame, x, weight=1, uniform=1)
            Grid.rowconfigure(frame, x, weight=1, uniform=1)

        for y in range(1,8):
            # Grid.rowconfigure(frame, y, weight=1, uniform=1)
            Grid.columnconfigure(frame, y, weight=1, uniform=1)

        v = StringVar()
        question_label = Label(frame, text='Enter Question').grid(row=6,column=2,pady=10, sticky=N+S+E+W)

        e = Entry(frame, textvariable = v).grid(row=6, column=3,columnspan=3,pady=10, sticky=N+S+E+W)
        button = Button(frame, text='Submit', command= lambda: self.get_probabilities(root, v), relief=RIDGE).grid(row=7, column=4,pady=20, sticky=N+S+E+W)
        knowledge_label = Label(frame, text='Knowledge Dimension').grid(row=3,column=0, sticky=N+S+E+W)
        cognitive_label = Label(frame, text='Cognitive Dimension').grid(row=0,column=4, sticky=N+S+E+W)


        root.mainloop()


def main():
    obj = Visualize()
    root = Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()

    # root.overrideredirect(True)
    root.geometry("%dx%d+0+0" % (w, h))
    root.focus_set() # <-- move focus to this widget
    root.bind("<Escape>", lambda e: e.widget.quit())
    root.wm_title = "BLOOMS TAXONOMY"

    obj.create_table(root)


# if __name__ == "__main__":
main()
