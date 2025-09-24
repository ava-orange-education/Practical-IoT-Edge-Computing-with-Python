#importing libraries
import numpy as np
import flask
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/first')
def index():
    return flask.render_template('first.html')
    
#prediction function
def ValuePredictor(to_predict_list):
    result=7
    to_predict = np.array(to_predict_list).reshape(1,5)
    # If it is Dark and People are more than 3 
    if(to_predict[0][0]>3 & to_predict[0][1]==2):
        #if both lights off
        if(to_predict[0][2] ==0 & to_predict[0][3] ==0):
           result=3
           #If light 2 off
        elif(to_predict[0][2] ==1 & to_predict[0][3] ==0):
           result=1
           #If light 1 off
        elif(to_predict[0][2] ==0 & to_predict[0][3] ==1):
            result=2
    # If it is Dark and People are less than 3
    if(to_predict[0][0]<3 & to_predict[0][1]==2):
        #If Both lights are off
        if(to_predict[0][2] ==0 & to_predict[0][3] ==0):
            result=2
        #If both lights are on
        elif(to_predict[0][2] ==1 & to_predict[0][3] ==1):
            result=5
        elif(to_predict[0][2] ==1 & to_predict[0][3] ==0):
           result=1
         
        elif(to_predict[0][2] ==0 & to_predict[0][3] ==1):
            result=2
    if(to_predict[0][1] ==0):
        result = 4
    if(to_predict[0][1] ==1):
        if(to_predict[0][2] ==0 & to_predict[0][3] ==0):
            result=2
        elif(to_predict[0][2] ==1 & to_predict[0][3] ==1):
            result=5
        elif(to_predict[0][2] ==1 & to_predict[0][3] ==0):
           result=1
         
        elif(to_predict[0][2] ==0 & to_predict[0][3] ==1):
            result=2
    if(to_predict[0][1] == 2):
        #Reading
        if(to_predict[0][4] ==0):
            result=3
        #Eating
        elif(to_predict[0][4] ==1):
            result=3
        #Watching TV
        elif(to_predict[0][4] ==2):
            result=4
        #Sleeping
        elif(to_predict[0][4] == 3):
            result=4
  
    return result


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)==3:
            prediction='Both Lights ON'
        elif int(result)==2:
            prediction='Light 1 ON'
        elif int(result)==1:
            prediction='Light 2 ON'
        elif int(result)==5:
            prediction='Light 2 OFF'
        elif int(result)==4:
            prediction='Both Light OFF'
        else:
            prediction='No change'
            
        return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
	app.run(debug=True)