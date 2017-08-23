import pandas as pd
import numpy as np
import matplotlib
import autotime
import requests
import pickle
from sklearn.externals import joblib

# Get pickle file
model = joblib.load('02740002.pkl')

file = open('frame.pkl','rb')
frame = pickle.load(file)
file.close()

time = model.predict(frame)[0]
print(time)
