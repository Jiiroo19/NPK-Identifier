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

Builder.load_file('./libs/kv/calibrate_bg.kv')


class CalibrateBG(Screen):
    figure_wgt2 = ObjectProperty()
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

        self.ids['next_but'].disabled = True
        
        self.figure_wgt2.figure = mygraph.fig
        self.figure_wgt2.axes = mygraph.ax1
        self.figure_wgt2.xmin= np.min(self.wave[0])
        self.figure_wgt2.xmax = np.max(self.wave[0])
        self.figure_wgt2.ymin=np.min(self.get_all_spec[0])
        self.figure_wgt2.ymax = np.max(self.get_all_spec[0])
        self.figure_wgt2.line1=mygraph.line1
        self.home()
        self.figure_wgt2.home()
       
        Clock.schedule_interval(self.update_graph,1/60)

    def set_touch_mode(self,mode):
        self.figure_wgt2.touch_mode=mode

    def home(self):
        self.figure_wgt2.home()
        
    def update_graph(self,_):
        xdata= self.wave[0]
        self.figure_wgt2.line1.set_data(xdata,self.get_all_spec[self.i])
        self.figure_wgt2.ymax = np.max(self.get_all_spec[self.i])
        self.figure_wgt2.ymin = np.min(self.get_all_spec[self.i])
        self.figure_wgt2.xmax = np.max(xdata)
        self.figure_wgt2.xmin = np.min(xdata)
        self.home()
        self.figure_wgt2.figure.canvas.draw_idle()
        self.figure_wgt2.figure.canvas.flush_events() 
        self.i+=1
    
    def activate_next(self):
        self.ids['next_but'].disabled = not self.ids['next_but'].disabled
    
    def activate_capture(self):
        self.ids['capture'].disabled = not self.ids['capture'].disabled
    
    def activate_rescan(self):
        self.ids['rescan'].disabled = not self.ids['rescan'].disabled

    def disable_clock(self):
        Clock.unschedule(self.update_graph)