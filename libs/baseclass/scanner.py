from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivy.lang.builder import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.utils import platform

import matplotlib.pyplot as plt
from graph_generator import GraphGenerator
import numpy as np
import pandas as pd
import sqlite3
import os
import random

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

import RPi.GPIO as GPIO

Builder.load_file('./libs/kv/scanner.kv')


class Scanner(Screen):
    label_OM = ObjectProperty()
    label_P = ObjectProperty()
    label_K = ObjectProperty()
    figure_wgt4 = ObjectProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def on_enter(self, *args):
        # set the lights to high
        GPIO.output(12, GPIO.HIGH)
        # initial text label for NPK
        self.label_OM.text = "N: - ppm"
        self.label_P.text = "P: - ppm"
        self.label_K.text = "K: - ppm"

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
        self.spec.open()

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
            return np.frombuffer(result[0], dtype=np.float64)  # Adjust dtype based on your data type
        else:
            return None

    def reflectance_cal(self, sample_intensities):
        dark_data_retrieved = self.get_data('dark')
        light_data_retrieved = self.get_data('light')
        background_data_retrieved = self.get_data('background')

        ref_sub_dark = np.subtract(dark_data_retrieved, background_data_retrieved)
        corrected_ref = np.subtract(light_data_retrieved, ref_sub_dark)  

        sample_dark = np.subtract(dark_data_retrieved, background_data_retrieved)
        corrected_sample = np.subtract(sample_intensities, sample_dark)

        reflectance = np.divide(corrected_sample, corrected_ref)
        return np.multiply(reflectance, 100)

    def set_touch_mode(self,mode):
        self.figure_wgt4.touch_mode=mode

    def home(self):
        self.figure_wgt4.home()
        
    def update_graph(self,_):
        xdata= self.spec.wavelengths()
        intensities = self.reflectance_cal(np.array(self.spec.intensities(False,True), dtype=np.float64))
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
        self.capture_model(self.reflectance_cal(np.array(self.spec.intensities(False,True), dtype=np.float64)))
        Clock.unschedule(self.update_graph)

    def capture_model(self, final_reflectance):
        # os.environ['PYTHONHASHSEED'] = '0'
        # np.random.seed(42)
        # random.seed(42)
        # tf.random.set_seed(42)

        scaler = StandardScaler()
        input_data = np.array(final_reflectance).reshape(1, 128, 1)
        # Reshape to 2D for StandardScaler
        reshaped_input_data = input_data.reshape(-1, 1)

        # reflectance_scaled = scaler.fit_transform(reshaped_input_data)
        reflectance_scaled = scaler.fit_transform(reshaped_input_data).reshape(1, 128, 1)

        tf.keras.backend.clear_session()
        model_OM = tf.keras.models.load_model("./assets/models/final_regression_model_OM.h5")
        device_pred_OM = model_OM.predict(reflectance_scaled)
        self.label_OM.text = f"N: {round(float(device_pred_OM[0][0]),2)} ppm"

        tf.keras.backend.clear_session()
        model_P = tf.keras.models.load_model("./assets/models/final_regression_model_P.h5")
        device_pred_P = model_P.predict(reflectance_scaled)
        self.label_P.text = f"P: {round(float(device_pred_P[0][0]), 2)} ppm"

        tf.keras.backend.clear_session()
        model_K = tf.keras.models.load_model("./assets/models/final_regression_model_K.h5")
        device_pred_K = model_K.predict(reflectance_scaled)
        self.label_K.text = f"K: {round(float(device_pred_K[0][0]), 2)} ppm"


    def on_leave(self, *args):
        self.ids['rescan_button'].disabled = True
        self.ids['capture_button'].disabled = False
        self.conn.close()
        return super().on_leave(*args)