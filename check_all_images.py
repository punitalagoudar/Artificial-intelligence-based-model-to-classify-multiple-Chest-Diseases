from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import tkinter as tk
from tkinter import filedialog
import glob
from tkinter import messagebox
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sklearn import metrics
from Fn_Prediction import Pred2

def_label=["COVID", "Normal","Pneumonia","TB"]
 
def rocall(fpr,tpr):
    lw = 1
    n_classes=12
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Plot all ROC curves
    

    colors = cycle(['blue','red','green','yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.figure(i)
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(def_label[i], roc_auc[i]))


        plt.plot([0, 1], [0, 1], color='navy') #, lw=lw, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC '+def_label[i])


print("Complete Accuracy Analysis")

tot=0
match=0
itype=0

target=[1]*100+[0]*100
output=[]

tot_target=[]
tot_output=[]
tot_output2=[]

fpr=[0]*4
tpr=[0]*4
roc_auc=[0]*4

for foldername in glob.glob("test/*"):
        foldername=foldername.upper()
         
        foldername=foldername+"\*.*"
        
        count=0
        for imgfile in glob.glob(foldername):
                filename=imgfile[-24:]

                maxindex=Pred2(imgfile)
                
                Alabel=def_label[itype]
                Plabel=def_label[maxindex]

                print(Alabel,'--',Plabel);
                
                tot_target.append(itype)
                tot_output.append(maxindex)
                
                if itype==maxindex:
                        output.append(1)
                        tot_output2.append(1)
                        match=match+1
                        count=count+1
                else:
                        output.append(0)
                        tot_output2.append(0)

                       ### print(imgfile)

                tot=tot+1                        

        gacc=count/100*100
        print('='*60)
        print(Alabel,' Classification Accuracy ',round(gacc,2),' %')
        ##print('='*60)
        

        output=output+[0]*100
        fpr[itype], tpr[itype], _ = roc_curve(target,output)
        roc_auc[itype] = auc(fpr[itype], tpr[itype])
        output=[]
        itype=itype+1


acc=match/tot*100
print('\n Total Accuracy ',round(acc,2),' %')

           
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(tot_target, tot_output)
print('\nConfusion Matrix')
print(confusion)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro Precision: {:.2f}'.format(precision_score(tot_target, tot_output, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(tot_target, tot_output, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(tot_target, tot_output, average='micro')))


from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(tot_target, tot_output))

def_class=['01','02','03','04']


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = def_class)
cm_display.plot()
plt.title('Confusion matrix')
plt.show()


tot_target=[1]*400+[0]*400
tot_output2=tot_output2+[0]*400
fpr, tpr, _ = roc_curve(tot_target,tot_output2)
roc_auc = auc(fpr, tpr)
plt.figure(12)
plt.plot(fpr, tpr, color='blue') 
plt.plot([0, 1], [0, 1], color='red') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of CNN model')


mes='CNN Accuracy '+str(round(acc,2))+'%'
plt.text(0,1,mes)
plt.show()



 


                
                



