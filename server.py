from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)

filename = './mdoels/GaussianNB_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return 'Welcome to the home page served by Raspberry Pi'


## 4.1
import math
def FFT(data):
    col = data.columns.values
    df_fft = np.fft.fft(data, axis = 0)
    abs_df = pd.DataFrame(abs(df_fft), columns= col)
    
    return abs_df


## 4.2
def Concate_features(x):
    index_list = ["S" + str(n) for n in np.arange(1,5)]
    dff = pd.DataFrame({})
    for index in index_list:

        xx = x.loc[[index]]
        col = [index + "_" + "F" + str(n) for n in np.arange(1,16)]
        xx.columns = col 
        xx = xx.reset_index().drop(["index"], axis = 1)
        dff = pd.concat([dff, xx], axis = 1)
    return dff


## 4.3
def RMS(data):
    
    col = ["F"+ str(n) for n in np.arange(1,16)]
    
    x = data[0:10]
    x = x.applymap(lambda a : a**2)
    x = pd.DataFrame((x.sum()/10),columns=["F1"]).applymap(lambda a : math.sqrt(a))

    for n in np.arange(1,15):
        y = data[n*0:(n+1)*10]
        y = y.applymap(lambda a : a**2)
        y = pd.DataFrame((y.sum()/10) , columns= [col[n]]).applymap(lambda a : math.sqrt(a))
        x = pd.concat([x, y], axis = 1)
    
    df = Concate_features(x)
    return df


## 4.0
def Feature_create(data):
    df = pd.DataFrame({})
    X = data[['S1', 'S2', 'S3', 'S4']]
    available_range = X.shape[0]//300
    if available_range > 0:
        for n in range(available_range):
            time_domain = X[n*300:(n+1)*300]
            freq_domain = FFT(time_domain)
            f = freq_domain[0:150]
            f = RMS(f)
            df = pd.concat([df, f], axis = 0 )
    
    else:
        freq_domain = FFT(X)
        f = freq_domain
        df = RMS(f)


    return df 


@app.route('/predict', methods=['POST'])
def predict():
    # get the data from the POST request.
    data = request.get_json(force=True)
    dataF = {
        "S1": data['a1'],
        "S2": data['a2'],
        "S3": data['a3'],
        "S4": data['a4'],
    }
    df = pd.DataFrame.from_dict(dataF, orient='index').T
    df.columns = ['S1', 'S2', 'S3', 'S4']
    # print(df)
    df_data = Feature_create(df)
    df_data["Load"] = data.get("load")
    # print(df_data)
    # make prediction using model loaded from disk as per the data.
    prediction = loaded_model.predict(df_data)
    classes  = ['Healthy', 'Broken']
    return jsonify(classes[prediction[0]])


if __name__ == '__main__':
    print('Starting Python Flask Server For Load Prediction')
    app.run(port=5000, debug=True, host='0.0.0.0')
