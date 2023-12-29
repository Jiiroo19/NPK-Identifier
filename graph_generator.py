import matplotlib as mpl
import matplotlib.pyplot as plt
from kivy.metrics import dp

#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
mpl.rcParams['font.family'] = 'Verdana'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=dp(13)
font_size_axis_tick=dp(12)        

class GraphGenerator(object):
    """class that generate Matplotlib graph."""

    def __init__(self):
        """Create empty structure plot. 
        
        """       
        super().__init__()

        self.fig, self.ax1 = plt.subplots(1, 1)

        self.line1, = self.ax1.plot([], [],label='line1')       

        self.xmin,self.xmax = self.ax1.get_xlim()
        self.ymin,self.ymax = self.ax1.get_ylim()
        
        self.fig.subplots_adjust(left=0.13,top=0.96,right=0.93,bottom=0.2)
 
        self.ax1.set_xlim(self.xmin, self.xmax)
        self.ax1.set_ylim(self.ymin, self.ymax)
        # Set x-axis and y-axis label font styles
        # font_size_axis_title = 13  # Font size
        font_properties = {
            'family': 'IMPACT',  # Font family (change to your desired font)
            'weight': 'normal',   # Font weight: 'normal', 'bold', 'light', etc.
            'style': 'normal',   # Font style: 'normal', 'italic', 'oblique'
        }

        self.ax1.set_xlabel("Wavelengths", fontsize=font_size_axis_title, fontdict=font_properties)
        self.ax1.set_ylabel("Reflectance", fontsize=font_size_axis_title, fontdict=font_properties)
        
            
        self.ax1.set_xlabel("Wavelengths",fontsize=font_size_axis_title)
        self.ax1.xaxis.set_label_coords(0.5,-.12)
        self.ax1.set_ylabel("Reflectance",fontsize=font_size_axis_title)
        self.ax1.grid(True)
        self.ax1.patch.set_edgecolor('black')  
        self.ax1.patch.set_linewidth(1) 
        
                