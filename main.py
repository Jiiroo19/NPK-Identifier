from kivy.clock import Clock
from kivy.config import Config
from kivy.utils import rgba
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.properties import StringProperty, NumericProperty
from libs.baseclass import lobby, calibrate_light, calibrate_bg, calibrate_dark, scanner
from kivy.core.window import Window

from kivy.config import Config

# import RPi.GPIO as GPIO
from seabreeze.spectrometers import Spectrometer
import atexit



class MyApp(MDApp):
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(12, GPIO.OUT)
    spec = Spectrometer.from_first_available()
    spec.integration_time_micros(100000)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = 'NPK Identifier'
        self.theme_cls.primary_palette = "Gray"

    def build(self):
        kv_run = Builder.load_file("main.kv")
        atexit.register(self.on_exit)
        Config.set('graphics', 'fullscreen', 'auto')
        Config.write()
        return kv_run
        
    def on_exit(self):
        self.spec.close()
        # GPIO.cleanup()

    def colors(self, color_code):
        if color_code == 0:
            color_rgba = '#35353f'
        elif color_code == 1:
            color_rgba = '#09AF79'
        elif color_code == 2:
            color_rgba = '#ffffff'
        return rgba(color_rgba)

    def show_screen(self, name):
        self.root.current = 'lobby'
        self.root.get_screen('lobby').ids.manage.current = name
        return True


if __name__ == "__main__":
    MyApp().run()