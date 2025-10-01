from flask import Flask
from flask import Flask, jsonify, request, render_template
from flask_session import Session

import json
import sys

from flask import Flask, flash, redirect, render_template, \
     request, url_for ,session

from datetime import datetime
##import constant as cnst
import sqlite3 as sql

from tkinter import * 
from tkinter import messagebox 
import tkinter as tk

from Fn_Prediction import makePrediction

node = Flask(__name__)

node.config['SESSION_TYPE'] = 'medical'
node.config['SECRET_KEY'] = 'super'

sess = Session()


@node.route('/')
def  first():
    return  render_template('index.html')

@node.route('/menu')
def  menu():
    return  render_template('menu.html')


@node.route('/login', methods=['POST'])
def  login():
    
    error = None
    if  request.method  ==  'POST':
        username=request.form['username']
        password=request.form['password']
        con = sql.connect("chest")
        cur = con.cursor()
        ##statement = f"SELECT * from users WHERE username='{username}' AND Password = '{password}';"
        statement = "select * from Users where username='"+username+"' and password='"+password+"'"
        cur.execute(statement)
        if not cur.fetchone():  # An empty result evaluates to False.
            print("Login failed")
            error  =  'Invalid  username  or  password.  Please  try  again  !'
        else:
            return  render_template('menu.html')
            
    return  render_template('index.html',  error  =  error)

@node.route('/div2checkup')
def  div2checkup():
    return  render_template('checkup.html')
          
        
@node.route('/newcheckup', methods=['POST'])
def newcheckup():

    if  request.method  ==  'POST':

        fullname =request.form['fullname']
        age = request.form['age']
        gender = request.form['gender']

        res=makePrediction()
  
        con = sql.connect("chest")
        cur = con.cursor()
        records = [(fullname,age, gender,res)]
        cur.executemany('INSERT INTO checkup VALUES(?,?,?,?);',records);
        con.commit()

        Report="Predicted as : "+res
##        root = tk.Tk()
##        root.withdraw()
##        root.call('wm', 'attributes', '.', '-topmost', True)
##        messagebox.showinfo("Info", Report) 
##        root.destroy()

    return render_template('output.html',Report=Report) 


@node.route('/div2perf')
def  div2perfs():
    return render_template('result2.html') 

@node.route('/div2bak')
def  div2bak():
    return render_template('menu.html') 


@node.route('/div2checkuplist')
def  div2checkuplist():
        statement = "select * from checkup"
        con = sql.connect("chest")
        cur = con.cursor()
        cur.execute(statement)
        data = cur.fetchall()

        print(data)
        return  render_template('checklist.html',  data  =  data)
    

if __name__== '__main__':

    if len(sys.argv) >= 2:
        port = sys.argv[1]
    else:
        port = 8000

    node.run(host='127.0.0.1', port=port)
