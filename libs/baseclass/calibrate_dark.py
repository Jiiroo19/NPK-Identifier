from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivy.lang.builder import Builder
from kivy.clock import Clock
from kivy.properties import ObjectProperty

from graph_generator import GraphGenerator
import numpy as np
import sqlite3

import RPi.GPIO as GPIO


Builder.load_file('./libs/kv/calibrate_dark.kv')


class CalibrateDark(Screen):
    figure_wgt1 = ObjectProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_enter(self, *args):

        # set the lights to low
        GPIO.output(12, GPIO.LOW)

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
        
        self.figure_wgt1.figure = mygraph.fig
        self.figure_wgt1.axes = mygraph.ax1

        # get initial spectral data
        self.figure_wgt1.xmin= np.min(self.spec.wavelengths())
        self.figure_wgt1.xmax = np.max(self.spec.wavelengths())
        self.figure_wgt1.ymin=np.min(self.spec.intensities(False,True))
        self.figure_wgt1.ymax = np.max(self.spec.intensities(False,True))
        self.figure_wgt1.line1=mygraph.line1
        mygraph.line1.set_color('red')
        self.home()
        self.figure_wgt1.home()

        
       
        Clock.schedule_interval(self.update_graph,.1)
    
    # Function to delete existing data of a certain type
    def delete_data(self):
        self.cursor.execute('''
            DELETE FROM SpectralData WHERE type = ?
        ''', ("dark",))
        self.conn.commit()

    # Function to insert data into the table
    def insert_data(self, data_type, spectral_data):
        self.delete_data()
        self.cursor.execute('''
            INSERT INTO SpectralData (type, data) VALUES (?, ?)
        ''', (data_type, spectral_data))
        self.conn.commit()

    def set_touch_mode(self,mode):
        self.figure_wgt1.touch_mode=mode

    def home(self):
        self.figure_wgt1.home()
        
    def update_graph(self,_):
        xdata= self.spec.wavelengths()
        intensities = self.spec.intensities(False,True)
        self.figure_wgt1.line1.set_data(xdata,intensities)
        self.figure_wgt1.ymax = np.max(intensities)
        self.figure_wgt1.ymin = np.min(intensities)
        self.figure_wgt1.xmax = np.max(xdata)
        self.figure_wgt1.xmin = np.min(xdata)
        self.home()
        self.figure_wgt1.figure.canvas.draw_idle()
        self.figure_wgt1.figure.canvas.flush_events()
    
    def activate_capture(self):
        self.ids['capture_dark'].disabled = not self.ids['capture_dark'].disabled
        self.ids['next_dark'].disabled = not self.ids['next_dark'].disabled
        self.ids['rescan_dark'].disabled = not self.ids['rescan_dark'].disabled
    
    def activate_rescan(self):
        self.ids['rescan_dark'].disabled = not self.ids['rescan_dark'].disabled
        self.ids['next_dark'].disabled = not self.ids['next_dark'].disabled
        self.ids['capture_dark'].disabled = not self.ids['capture_dark'].disabled

    def disable_clock(self):
        self.insert_data('dark', np.array(self.spec.intensities(False,True), dtype=np.float32).reshape(-1, 1))
        Clock.unschedule(self.update_graph)
    
    def on_leave(self, *args):
        self.ids['next_dark'].disabled = True
        self.ids['capture_dark'].disabled = False
        self.ids['rescan_dark'].disabled = True
        self.conn.close()
        return super().on_leave(*args)

