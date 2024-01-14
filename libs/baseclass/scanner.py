from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivy.lang.builder import Builder
from kivy.clock import Clock
from kivy.properties import ObjectProperty

from graph_generator import GraphGenerator
import numpy as np
import sqlite3
import os
import random
import pandas as pd

import tensorflow as tf
import tflite_runtime.interpreter as tflite
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

import RPi.GPIO as GPIO


Builder.load_file('./libs/kv/scanner.kv')


class Scanner(Screen):
    label_OM = ObjectProperty()
    label_N = ObjectProperty()
    label_P = ObjectProperty()
    label_K = ObjectProperty()
    figure_wgt4 = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        self.data = pd.read_csv("/home/stardust/NPK-Identifier/assets/datasets/TrainingSets.csv")
        features = self.data.iloc[:, 4:].values
        first_der_features = self.calculate_first_derivative(features)
        der_features = np.concatenate((features, first_der_features), axis=1)
        self.scaler = StandardScaler().fit(der_features)



    def on_enter(self, *args):
        

        self.label_OM.text = "OM: - %"
        self.label_N.text = "N: - ppm"
        self.label_P.text = "P: - ppm"
        self.label_K.text = "K: - ppm"

        # set the lights to high (turn on)
        GPIO.output(12, GPIO.HIGH)

        # initialize database
        self.conn = sqlite3.connect('spectral_calib.db')
        self.cursor = self.conn.cursor()
        # Create a table to store spectral data
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS SpectralData (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                data BLOB NOT NULL
            )
        ''')

        # access the NIR
        self.spec = MDApp.get_running_app().spec

        #access the GraphGenerator() to create the initial graph
        mygraph = GraphGenerator()
        
        #initialize the state of the button upon entering the screen
        self.ids['rescan_button'].disabled = True
        self.ids['capture_button'].disabled = False

        self.figure_wgt4.figure = mygraph.fig
        self.figure_wgt4.axes = mygraph.ax1

        # get initial spectral data and update the graph
        self.figure_wgt4.xmin= np.min(self.spec.wavelengths())
        self.figure_wgt4.xmax = np.max(self.spec.wavelengths())
        self.figure_wgt4.ymin=np.min(self.spec.intensities(False,True))
        self.figure_wgt4.ymax = np.max(self.spec.intensities(False,True))
        self.figure_wgt4.line1=mygraph.line1
        mygraph.line1.set_color('red')
        self.home()
        self.figure_wgt4.home()
        Clock.schedule_interval(self.update_graph,.1)

    # get the data stored in the database
    def get_data(self, data_type):
        self.cursor.execute('''
            SELECT data FROM SpectralData WHERE type = ?
        ''', (data_type,))
        result = self.cursor.fetchone()
        if result:
            # Convert bytes back to NumPy array when retrieving from the database
            return np.frombuffer(result[0], dtype=np.float32)  # Adjust dtype based on your data type
        else:
            return None
        
    
    # rough estimation of available nitrogen
    def cal_nitrogen(self, organic_matter):
        return (((organic_matter/100) * 0.03) * 0.2) * 10000
        
    # calculation to convert or calculate the reflectance from intensity
    def reflectance_cal(self, sample_intensities):
        dark_data_retrieved = self.get_data('dark')
        light_data_retrieved = self.get_data('light')
        background_data_retrieved = self.get_data('background')

        ref_sub_dark = np.subtract(dark_data_retrieved, background_data_retrieved)
        corrected_ref = np.subtract(light_data_retrieved, ref_sub_dark)  

        sample_dark = np.subtract(dark_data_retrieved, background_data_retrieved)
        corrected_sample = np.subtract(sample_intensities, sample_dark)

        reflectance = np.divide(corrected_sample, corrected_ref)
        reflectance_mult = np.multiply(reflectance, 100)

        # Apply Savitzky-Golay filter
        window_length = 5  # Adjust for desired smoothing level
        polyorder = 2  # Polynomial order (often 2 or 3 for spectroscopy)
        
        return savgol_filter(reflectance_mult, window_length, polyorder)
    
    def set_touch_mode(self,mode):
        self.figure_wgt4.touch_mode=mode

    def home(self):
        self.figure_wgt4.home()

    #update the content of the graph   
    def update_graph(self,_):
        xdata= self.spec.wavelengths()
        intensities = self.reflectance_cal(np.array(self.spec.intensities(False,True), dtype=np.float32))
        self.figure_wgt4.line1.set_data(xdata,intensities)
        self.figure_wgt4.ymax = np.max(intensities)
        self.figure_wgt4.ymin = np.min(intensities)
        self.figure_wgt4.xmax = np.max(xdata)
        self.figure_wgt4.xmin = np.min(xdata)
        self.home()
        self.figure_wgt4.figure.canvas.draw_idle()
        self.figure_wgt4.figure.canvas.flush_events() 
    
    # if a button is press it will reverse the state of the buttons
    def activate_button(self):
        self.ids['rescan_button'].disabled = not self.ids['rescan_button'].disabled
        self.ids['capture_button'].disabled = not self.ids['capture_button'].disabled

    # this will stop the scheduled task and get the final reflectance and update the labels
    def disable_clock(self):
        output_data_OM, output_data_P, output_data_K = self.capture_model(self.reflectance_cal(np.array(self.spec.intensities(False,True))))
        
        # update the text labels of OM, N, P, K
        self.label_OM.text = f"OM: {round(float(output_data_OM[0][0]),2)} %"
        self.label_N.text = f"N: {round(float(self.cal_nitrogen(output_data_OM)),2)} ppm"
        self.label_P.text = f"P: {round(float(output_data_P[0][0]), 2)} ppm"
        self.label_K.text = f"K: {round(float(output_data_K[0][0]), 2)} ppm"

        Clock.unschedule(self.update_graph)

    # this will load the tflite saved model
    def loading_model(self, reflectance_scaled, model_path, model_shape):
        tf.keras.backend.clear_session()

        # load lite model of OM
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_data = reflectance_scaled.astype(np.float32).reshape(1, model_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]['index'])
    
    def standardize_column(self, features):
        ## We train the scaler on the full train set and apply it to the other datasets
        features_scaled = self.scaler.transform(features)
        return features_scaled

    def calculate_first_derivative(self, features):
        # Compute the first derivative of the spectra along the wavelength axis
        first_derivative = np.diff(features, n=1, axis=1)
        
        # Pad the result with zeros to match the original number of features
        first_derivative_padded = np.pad(first_derivative, ((0, 0), (0, 1)), 'constant', constant_values=0)
        
        return first_derivative_padded

    def capture_with_derivatives(self, reflectance, model_path, model_shape):
        # x_cal, x_tuning = train_test_split(np.array(self.data.iloc[:, 4:]).astype(np.float32), test_size=0.33, random_state=42)
        
        first_der_reflectance = self.calculate_first_derivative(reflectance)

        
        der_reflectance = np.concatenate((reflectance, first_der_reflectance), axis=1)

        reflectance_scaled = self.standardize_column(der_reflectance)

        return self.loading_model(reflectance_scaled, model_path, model_shape)

    def capture_model(self, final_reflectance):
        # X_train = np.array(self.data.iloc[:, 4:96]).astype(np.float32) 
        # reflectance_scaled = self.standardize_column(X_train , np.array(final_reflectance[:92]).astype(np.float32).reshape(1, -1))
        
        # the code is being run by root the reason for this hardcoded directory
        output_data_OM = self.capture_with_derivatives(final_reflectance.reshape(1, -1), "/home/stardust/NPK-Identifier/assets/models/final_regression_model_OM.tflite", 256)

        # load lite model of P
        output_data_P = self.capture_with_derivatives(final_reflectance.reshape(1, -1), "/home/stardust/NPK-Identifier/assets/models/final_regression_model_P.tflite", 256)

        # load lite model of K
        output_data_K  = self.capture_with_derivatives(final_reflectance.reshape(1, -1), "/home/stardust/NPK-Identifier/assets/models/final_regression_model_K.tflite", 256)

        return output_data_OM, output_data_P, output_data_K
        # return output_data_OM, output_data_P, None

    # set of process upon leaving the screen
    def on_leave(self, *args):
        self.ids['rescan_button'].disabled = True
        self.ids['capture_button'].disabled = False
        GPIO.output(12, GPIO.LOW)
        self.conn.close()
        return super().on_leave(*args)
