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

import RPi.GPIO as GPIO

Builder.load_file('./libs/kv/scanner.kv')


class Scanner(Screen):
    figure_wgt4 = ObjectProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def on_enter(self, *args):
        # set the lights to high
        GPIO.output(12, GPIO.HIGH)

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
        self.home()
        self.figure_wgt4.home()
       
        Clock.schedule_interval(self.update_graph,1/60)

    def set_touch_mode(self,mode):
        self.figure_wgt4.touch_mode=mode

    def home(self):
        self.figure_wgt4.home()
        
    def update_graph(self,_):
        xdata= self.spec.wavelengths()
        intensities = self.spec.intensities(False,True)
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
        Clock.unschedule(self.update_graph)

    def on_leave(self, *args):
        self.ids['rescan_button'].disabled = True
        self.ids['capture_button'].disabled = False
        GPIO.output(12, GPIO.HIGH)
        return super().on_leave(*args)