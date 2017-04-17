from tkinter import *

import cogvoter
import nsquared_imp
import utils

subject = 'ADA'
    
class Visualize:

    def __init__(self):
        self.array1 = []
        self.array2 = []

    def getProbabilities(self, root, question):
        #Make a call to Shiva and Mohit to get required arrays: send question
        question = question.get()
        array1 = [0.2, 0.3, 0.4, 0]
        array2 = [0.1, 0.1, 0.3, 0.3, 0, 0]
        know_level, know_prob = nsquared_imp.get_know_label(question, subject)
        print(know_level, know_prob[know_level])
        array1 = utils.get_knowledge_probs_level(know_level, know_prob[know_level])
        cog_level, cog_prob = cogvoter.predict_cog_label(question)
        print(cog_level, cog_prob[cog_level])
        array2 = utils.get_cognitive_probs_level(cog_level, cog_prob[cog_level])
        nmarray = np.outer(np.array(array1), np.array(array2))
        nmarray = nmarray / np.max(np.max(nmarray))
        print(question)
        print(array1)
        print(array2)
        print(nmarray)
        self.populateTable(root, nmarray)

        #return nmarray

    def populateTable(self, root, nmarray):
        frame=Frame(root)
        Grid.rowconfigure(root, 0, weight=1)
        Grid.columnconfigure(root, 0, weight=1)
        frame.grid(row=0, column=0, sticky=N+S+E+W)
        grid=Frame(frame)
        grid.grid(sticky=N+S+E+W, column=7, row=7)
        Grid.rowconfigure(frame, 0, weight=1)
        Grid.columnconfigure(frame, 0, weight=1)
        cog_values = ['', '', 'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
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
                    var = nmarray[x-2,y-2]*255
                    if(var != 0):
                        myCellColor = '#%02x%02x%02x' % (255 - int(var)//2, 0, 0)  # set your heatmap color
                    else:
                        myCellColor = '#%02x%02x%02x' % (255, 255,255)  # set it white
                    l = Label(frame, text= '', relief=RIDGE, bg= myCellColor)
                l.grid(row=x, column=y, sticky=N+S+E+W)



        for x in range(1,6):
            #Grid.columnconfigure(frame, x, weight=1, uniform=1)
            Grid.rowconfigure(frame, x, weight=1, uniform=1)

        for y in range(1,8):
            #Grid.rowconfigure(frame, y, weight=1, uniform=1)
            Grid.columnconfigure(frame, y, weight=1, uniform=1)


        button = Button(frame, text='Refresh', command= lambda: self.createTable(root)).grid(row=6, column=4,pady=20, sticky=N+S+E+W)
        knowledgeLabel = Label(frame, text= 'Knowledge Dimension').grid(row=3,column=0, sticky=N+S+E+W)
        cognitiveLabel = Label(frame, text= 'Cognitive Dimension').grid(row=0,column=4,sticky=N+S+E+W)
        root.mainloop()

    def createTable(self, root):
        frame=Frame(root)
        Grid.rowconfigure(root, 0, weight=1)
        Grid.columnconfigure(root, 0, weight=1)
        frame.grid(row=0, column=0, sticky=N+S+E+W)
        grid=Frame(frame)
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
            #Grid.columnconfigure(frame, x, weight=1, uniform=1)
            Grid.rowconfigure(frame, x, weight=1, uniform=1)

        for y in range(1,8):
            #Grid.rowconfigure(frame, y, weight=1, uniform=1)
            Grid.columnconfigure(frame, y, weight=1, uniform=1)

        v = StringVar()
        questionLabel = Label(frame, text= 'Enter Question').grid(row=6,column=2,pady=10, sticky=N+S+E+W)

        e = Entry(frame, textvariable = v).grid(row=6, column=3,columnspan=3,pady=10, sticky=N+S+E+W)
        button = Button(frame, text='Submit', command= lambda: self.getProbabilities(root, v), relief=RIDGE).grid(row=7, column=4,pady=20, sticky=N+S+E+W)
        knowledgeLabel = Label(frame, text= 'Knowledge Dimension').grid(row=3,column=0, sticky=N+S+E+W)
        cognitiveLabel = Label(frame, text= 'Cognitive Dimension').grid(row=0,column=4, sticky=N+S+E+W)


        root.mainloop()


def main():
    obj = Visualize()
    root = Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()

    #root.overrideredirect(True)
    root.geometry("%dx%d+0+0" % (w, h))
    root.focus_set() # <-- move focus to this widget
    root.bind("<Escape>", lambda e: e.widget.quit())
    root.wm_title = "BLOOMS TAXONOMY"

    obj.createTable(root)


#if __name__ == "__main__":
main()
