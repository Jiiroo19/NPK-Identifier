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

Builder.load_file('./libs/kv/scanner.kv')


class Scanner(Screen):
    figure_wgt4 = ObjectProperty()
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
        
        self.ids['rescan_button'].disabled = True
        self.ids['capture_button'].disabled = False

        self.figure_wgt4.figure = mygraph.fig
        self.figure_wgt4.axes = mygraph.ax1
        self.figure_wgt4.xmin= np.min(self.wave[0])
        self.figure_wgt4.xmax = np.max(self.wave[0])
        self.figure_wgt4.ymin=np.min(self.get_all_spec[0])
        self.figure_wgt4.ymax = np.max(self.get_all_spec[0])
        self.figure_wgt4.line1=mygraph.line1
        mygraph.line1.set_color('red')
        self.home()
        self.figure_wgt4.home()
       
        Clock.schedule_interval(self.update_graph,1/60)

    def set_touch_mode(self,mode):
        self.figure_wgt4.touch_mode=mode

    def home(self):
        self.figure_wgt4.home()
        
    def update_graph(self,_):
        xdata= self.wave[0]
        self.figure_wgt4.line1.set_data(xdata,self.get_all_spec[self.i])
        self.figure_wgt4.ymax = np.max(self.get_all_spec[self.i])
        self.figure_wgt4.ymin = np.min(self.get_all_spec[self.i])
        self.figure_wgt4.xmax = np.max(xdata)
        self.figure_wgt4.xmin = np.min(xdata)
        self.home()
        self.figure_wgt4.figure.canvas.draw_idle()
        self.figure_wgt4.figure.canvas.flush_events() 
        self.i+=1
    
    def activate_button(self):
        if self.ids['rescan_button'].disabled:
            self.ids['rescan_button'].disabled = False
            self.ids['capture_button'].disabled = True
        else:
            self.ids['rescan_button'].disabled = True
            self.ids['capture_button'].disabled = False

    def disable_clock(self):
        Clock.unschedule(self.update_graph)