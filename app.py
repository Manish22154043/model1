from flask import Flask,request,jsonify

import pickle
import numpy as np

model=pickle.load(open('placement.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict',methods=['POST'])
def predict():
    CGPA=request.form.get('CGPA')
    IQ=request.form.get('IQ')

    input_query=np.array([[CGPA,IQ]])

    result=model.predict(input_query)[0]

    return ({'Placement':str(result)})



if __name__=="__main__":
    app.run(debug=True)
