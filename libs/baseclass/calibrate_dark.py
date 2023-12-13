from kivy.uix.screenmanager import Screen
from kivy.lang.builder import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt

Builder.load_file('./libs/kv/calibrate_dark.kv')

# Define what we want to graph
x = [1,2,3,4,5]
y = [5, 12, 6, 9, 15]

plt.plot(x,y)
plt.ylabel("This is MY Y Axis")
plt.xlabel("X Axis")

class PlotDark(FloatLayout):
    def __init__(self, **kwargs):
        super(PlotDark).__init__(**kwargs)
        
    
   


class CalibrateDark(Screen):
    pass