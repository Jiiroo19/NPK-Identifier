from kivy.config import Config
from kivy.utils import rgba
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from libs.baseclass import lobby, calibrate_light, calibrate_bg, calibrate_dark, scanner

from kivy.config import Config

import RPi.GPIO as GPIO
from seabreeze.spectrometers import Spectrometer
import atexit
import os



class MyApp(MDApp):
    spec = Spectrometer.from_first_available()
    spec.integration_time_micros(100000)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = 'NPK Identifier'
        self.theme_cls.primary_palette = "Gray"
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(12, GPIO.OUT)
        GPIO.output(12, GPIO.LOW)

    def build(self):
        kv_run = Builder.load_file("main.kv")
        atexit.register(self.on_exit)
        Config.set('graphics', 'fullscreen', 'auto')
        Config.write()
        return kv_run
        
    def on_exit(self):
        self.spec.close()

    def show_screen(self, name):
        self.root.current = 'lobby'
        self.root.get_screen('lobby').ids.manage.current = name
        return True


if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    MyApp().run()
