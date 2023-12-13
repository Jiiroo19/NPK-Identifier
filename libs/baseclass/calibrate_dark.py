from kivy.uix.screenmanager import Screen
from kivy.lang.builder import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.properties import ObjectProperty

from kivy.utils import platform

#avoid conflict between mouse provider and touch (very important with touch device)
#no need for android platform
if platform != 'android':
    from kivy.config import Config
    Config.set('input', 'mouse', 'mouse,disable_on_activity')

# from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from graph_generator import GraphGenerator

import numpy as np
import pandas as pd

# from libs.baseclass import graph_widget

Builder.load_file('./libs/kv/calibrate_dark.kv')


class CalibrateDark(Screen):
    figure_wgt = ObjectProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.i=0
        self.csv_file = pd.read_csv('./all_results - Copy.csv')
        self.wave = pd.read_csv('wave.csv').iloc[:,:].values
        self.get_all_spec = self.csv_file.iloc[:, 4:].values
        print(self.get_all_spec[0])
        print(self.wave[0])


    def on_enter(self, *args):
        mygraph = GraphGenerator()
        
        self.figure_wgt.figure = mygraph.fig
        self.figure_wgt.axes = mygraph.ax1
        self.figure_wgt.xmin= np.min(self.wave[0])
        self.figure_wgt.xmax = np.max(self.wave[0])
        self.figure_wgt.ymin=np.min(self.get_all_spec[0])
        self.figure_wgt.ymax = np.max(self.get_all_spec[0])
        self.figure_wgt.line1=mygraph.line1
        self.home()
        self.figure_wgt.home()
       
        Clock.schedule_interval(self.update_graph,1/60)

    def set_touch_mode(self,mode):
        self.figure_wgt.touch_mode=mode

    def home(self):
        self.figure_wgt.home()
        
    def update_graph(self,_):
        xdata= self.wave[0]
        self.figure_wgt.line1.set_data(xdata,self.get_all_spec[self.i])
        self.figure_wgt.ymax = np.max(self.get_all_spec[self.i])
        self.figure_wgt.ymin = np.min(self.get_all_spec[self.i])
        self.figure_wgt.xmax = np.max(xdata)
        self.figure_wgt.xmin = np.min(xdata)
        self.home()
        self.figure_wgt.figure.canvas.draw_idle()
        self.figure_wgt.figure.canvas.flush_events() 
        self.i+=1
    
    def activate_button(self):
        self.ids['next_but'].disabled = False

    def disable_clock(self):
        Clock.unschedule(self.update_graph)