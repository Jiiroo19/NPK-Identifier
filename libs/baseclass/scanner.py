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

import tensorflow as tf
import tflite_runtime.interpreter as tflite
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

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


    def on_enter(self, *args):
        self.label_OM.text = "OM: - %"
        self.label_N.text = "N: - ppm"
        self.label_P.text = "P: - ppm"
        self.label_K.text = "K: - ppm"

        # set the lights to high
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

        mygraph = GraphGenerator()
        
        self.ids['rescan_button'].disabled = True
        self.ids['capture_button'].disabled = False

        self.figure_wgt4.figure = mygraph.fig
        self.figure_wgt4.axes = mygraph.ax1

        # get initial spectral data
        self.figure_wgt4.xmin= np.min(self.spec.wavelengths())
        self.figure_wgt4.xmax = np.max(self.spec.wavelengths())
        self.figure_wgt4.ymin=np.min(self.spec.intensities(False,True))
        self.figure_wgt4.ymax = np.max(self.spec.intensities(False,True))
        self.figure_wgt4.line1=mygraph.line1
        mygraph.line1.set_color('red')
        self.home()
        self.figure_wgt4.home()
       
        Clock.schedule_interval(self.update_graph,.1)

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
        window_length = 6  # Adjust for desired smoothing level
        polyorder = 3  # Polynomial order (often 2 or 3 for spectroscopy)

        return savgol_filter(reflectance_mult, window_length, polyorder)
    
    def set_touch_mode(self,mode):
        self.figure_wgt4.touch_mode=mode

    def home(self):
        self.figure_wgt4.home()
        
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
    
    def activate_button(self):
        self.ids['rescan_button'].disabled = not self.ids['rescan_button'].disabled
        self.ids['capture_button'].disabled = not self.ids['capture_button'].disabled

    def disable_clock(self):
        output_data_OM, output_data_P, output_data_K = self.capture_model(self.reflectance_cal(np.array(self.spec.intensities(False,True), dtype=np.float32)))
        
        # update the text labels of OM, N, P, K
        self.label_OM.text = f"OM: {round(float(output_data_OM[0][0]),2)} %"
        self.label_N.text = f"N: {round(float(self.cal_nitrogen(output_data_OM)),2)} ppm"
        self.label_P.text = f"P: {round(float(output_data_P[0][0]), 2)} ppm"
        self.label_K.text = f"K: {round(float(output_data_K[0][0]), 2)} ppm"

        Clock.unschedule(self.update_graph)

    def loading_model(self, reflectance_scaled, model_path):
        tf.keras.backend.clear_session()

        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        # load lite model of OM
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_data = reflectance_scaled.astype(np.float32).reshape(1, 92)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]['index'])

    def capture_model(self, final_reflectance):
        final_reflectance = final_reflectance[:92]
        print(final_reflectance)

        scaler = StandardScaler()
        # input_data = np.array((final_reflectance), dtype = np.float32).reshape(1, 128)
        # Reshape to 2D for StandardScaler
        reshaped_input_data = final_reflectance.reshape(-1, 1)
        reflectance_scaled = scaler.fit_transform(reshaped_input_data)

        output_data_OM = self.loading_model(reflectance_scaled, "/home/stardust/NPK-Identifier/assets/models/final_regression_model_OM.tflite")

        # load lite model of P
        output_data_P = self.loading_model(reflectance_scaled, "/home/stardust/NPK-Identifier/assets/models/final_regression_model_P.tflite")

        # load lite model of K
        output_data_K  = self.loading_model(reflectance_scaled, "/home/stardust/NPK-Identifier/assets/models/final_regression_model_K.tflite")

        return output_data_OM, output_data_P, output_data_K
        # return output_data_OM, output_data_P, None


    def on_leave(self, *args):
        self.ids['rescan_button'].disabled = True
        self.ids['capture_button'].disabled = False
        GPIO.output(12, GPIO.LOW)
        self.conn.close()
        return super().on_leave(*args)
