from tkinter import *
from tkinter import ttk
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import pyscreenshot as ImageGrab
class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 30
        self.model = tf.keras.models.load_model('mnist.h5')
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None      

    def changeW(self,e): #change Width of pen through slider
        self.penwidth = e
           

    def clear(self):
        self.c.delete(ALL)
        self.label.configure(text='Draw something',font=("Courier", 30))

    def predict(self):
        #get the image
        x = root.winfo_rootx()+self.c.winfo_x()
        y = root.winfo_rooty()+self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        img = ImageGrab.grab().crop((x,y,x1,y1))
        img = img_to_array(img)

        #preprocess the image
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img,(28,28))
        img = img.reshape(1,28,28,1)
        img = img/255.0

        #peform predictions
        prediction, acc = self.model.predict(img).argmax(axis=1),self.model.predict(img).max(axis=1)
        self.label.configure(text=str(prediction[0])+',confidence: '+str(acc*100))

        pass
    def drawWidgets(self):
        self.c = Canvas(self.master,width=500,height=500,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)
        self.label = Label(self.master, text="Draw something", font=("Courier", 20))
        self.label.pack(side=BOTTOM)
        self.b = Button(self.master, text = 'Predict!', bd = '5',command=self.predict)
        self.b.pack(side=BOTTOM)
        self.bc = Button(self.master,text = 'Clear',bd='5',command=self.clear)  
        self.bc.pack(side=BOTTOM)

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Paint App')
    root.mainloop()