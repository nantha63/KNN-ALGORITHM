 
import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv("F:\MACHINE LEARNING\Data Set\Iris_new.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)

  

  
# Define the GUI
root = tk.Tk()
root.title("KNN Classifier")
root.geometry("1500x1500")
root.config(bg="#2ecc71")

# Define the input fields

label1 = tk.Label(root, text="SEPAL LENGTH:",font=("comic sans Ms",10,"bold"),bg="#c0392b")
label1.place(x=500,y=200)
entry1 = tk.Entry(root,width=30)
entry1.place(x=500,y=250)

title_label = tk.Label(root, text="PREDICT  SPECIES",font=("comic sans Ms",20,"bold"),bg="#c0392b")
title_label.place(x=650,y=100)


label2 = tk.Label(root, text="SEPAL WIDTH:",font=("comic sans Ms",10,"bold"),bg="#c0392b")
label2.place(x=600,y=300)
entry2 = tk.Entry(root,width=30)
entry2.place(x=600,y=350)

#define the input

label3 = tk.Label(root, text="PEATAL LENGTH:",font=("comic sans Ms",10,"bold"),bg="#c0392b")
label3.place(x=650,y=400)
entry3 = tk.Entry(root)
entry3.place(x=650,y=450)



label3 = tk.Label(root, text="PETAL WIDTH:",font=("comic sans Ms",10,"bold"),bg="#c0392b")
label3.place(x=700,y=500)
entry3 = tk.Entry(root)
entry3.place(x=700,y=550)

# Define the prediction function
def predict():
    sepal_length = float(entry1.get())
    sepal_width = float(entry2.get())
     
    test_data = sc.transform([[sepal_length, sepal_width]])
    prediction = classifier.predict(test_data)
    result_label.config(font=("comic sans Ms",20,"bold"),bg="#c0392b",text="Predicted Species: " + prediction[0])
    result_label.place(x=900,y=200)

def predict():
    petal_length=float(entry3.get())
    petal_width=float(entry3.get())
    test_data = sc.transform([[petal_length,petal_width]])
    prediction = classifier.predict(test_data)
    result_label.config(font=("comic sans Ms",20,"bold"),bg="#c0392b",text="Predicted Species: " + prediction[0])
    result_label.place(x=900,y=300)

# Define the prediction button
button = tk.Button(root, text="PREDICT",font=("comic sans Ms",20,"bold"),bg="#c0392b", command=predict)
button.place(x=1000,y=400)

# Define the result label
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()






















