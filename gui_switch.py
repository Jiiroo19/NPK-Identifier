import kivy
from kivy.app import App
from kivy.uix.switch import Switch
from kivy.lang import Builder
from kivy.config import Config
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
import RPi.GPIO as GPIO

# Set the application to fullscreen
#Config.set('graphics', 'fullscreen', 'auto')

# Set a fixed resolution (might not be necessary for fullscreen)
#Config.set('graphics', 'width', '480')
#Config.set('graphics', 'height', '320')

# Apply the configuration changes
#Config.write()

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)

# KivyMD GUI design
KV = '''
BoxLayout:
    orientation: 'vertical'
    padding: '10dp'
    MDLabel:
        text: 'GPIO Control'
        halign: 'center'
        theme_text_color: 'Secondary'
    MDSwitch:
        id: gpio_switch
        pos_hint: {'center_x': 0.5, 'center_y': 0.5}
        on_active: app.switch_callback(*args)
    MDFlatButton:
        text: 'Exit'
        pos_hint: {'center_x': 0.5}
        on_release: app.stop()
'''

class GPIOApp(MDApp):
    def build(self):
        return Builder.load_string(KV)

    def switch_callback(self, switch_object, switch_value):
        if switch_value:
            GPIO.output(12, GPIO.HIGH)
        else:
            GPIO.output(12, GPIO.LOW)

if __name__ == '__main__':
    GPIOApp().run()

# Clean up GPIO on exit
GPIO.cleanup()
