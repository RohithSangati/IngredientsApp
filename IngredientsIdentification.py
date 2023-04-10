from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from Recipe import *
import os
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import ast
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split

main = tkinter.Tk()
main.title("Ingredients Identification from the Food Image")
main.geometry("1300x1200")

global filename
global classifier
recipe_list = []
global dataset

def uploadDataset():
    textarea.delete('1.0', END)
    global filename
    global dataset
    recipe_list.clear()
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=os.path.basename(filename))    
    textarea.insert(END,'Dataset loaded\n\n')

    dataset = pd.read_csv(filename,nrows=1000)
    for i in range(len(dataset)):
        r_id = dataset.get_value(i, 'recipe_id')
        r_name = dataset.get_value(i, 'recipe_name')
        ingredients = dataset.get_value(i, 'ingredients')
        nutritions = dataset.get_value(i, 'nutritions')
        cooking = ast.literal_eval(dataset.get_value(i, 'cooking_directions')).get('directions')
        r_name = r_name.strip().lower()
        obj = Recipe()
        obj.setRecipeID(r_id)
        obj.setName(r_name)
        obj.setIngredients(ingredients)
        obj.setNutritions(nutritions)
        obj.setCooking(cooking)
        recipe_list.append(obj)
    indian = np.load('model/index.txt.npy',allow_pickle=True)
    for i in range(len(indian)):
        recipe_list.append(indian[i])
    obj = recipe_list[len(recipe_list)-1]
    print(obj.getName())
    data = ""
    textarea.insert(END,"Recipes data loaded\n")            
        
def buildCNNModel():
    textarea.delete('1.0', END)
    global classifier
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    print(y_train.shape)
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = classifier.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        classifier.load_weights("model/model_weights.hdf5")
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    textarea.insert(END,"CNN training process completed with Accuracy = "+str(accuracy))
    

def predict():
    textarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)
    if predict > 0:
        predict = predict - 1
    print(predict)
    obj = recipe_list[predict]
    textarea.insert(END,"Recipe Name\n")
    textarea.insert(END,obj.getName()+"\n\n")
    textarea.insert(END,"Ingredients Details\n")
    textarea.insert(END,obj.getIngredients()+"\n\n")
    textarea.insert(END,"Cooking Details\n")
    textarea.insert(END,obj.getCooking()+"\n\n")
    textarea.insert(END,"Nutritions Details\n")
    textarea.insert(END,obj.getNutritions()+"\n\n")

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, 'Receipe Name : '+obj.getName(), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Receipe Name : '+obj.getName(), img)
    cv2.waitKey(0)

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    #plt.plot(loss, 'ro-', color = 'red')
    #plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.plot(accuracy, 'ro-', color = 'orange')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Recipe CNN Accuracy & Loss Graph')
    plt.show()

    
def close():
    main.destroy()
    
font = ('times', 14, 'bold')
title = Label(main, text='Ingredients Identification from the Food Image')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Food Image Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='mint cream', fg='olive drab')  
pathlabel.config(font=font1)           
pathlabel.place(x=320,y=100)

cnnButton = Button(main, text="Build CNN Model", command=buildCNNModel)
cnnButton.place(x=50,y=150)
cnnButton.config(font=font1) 

predictButton = Button(main, text="Upload Image & Identify Ingredients", command=predict)
predictButton.place(x=320,y=150)
predictButton.config(font=font1) 

graphButton = Button(main, text="CNN Model Accuracy/Loss Graph", command=graph)
graphButton.place(x=650,y=150)
graphButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=50,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=20,width=150)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=250)
textarea.config(font=font1)

main.config(bg='gainsboro')
main.mainloop()
