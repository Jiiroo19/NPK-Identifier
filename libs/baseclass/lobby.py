from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivy.lang.builder import Builder
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
import subprocess

Builder.load_file('./libs/kv/lobby.kv')




class Lobby(Screen):
    dialog = None

    def dialog_close(self, *args):
        self.dialog.dismiss(force=True)

    def shutdown(self):
        if not self.dialog:
            self.dialog = MDDialog(
                text="Are you sure you want to shut down?",
                buttons=[
                    MDFlatButton(
                        text="No",
                        theme_text_color="Custom",
                        font_name = './assets/fonts/IMPACT.TTF',
                        text_color=[0.7215, 0.451, 0.2, 1],
                        on_release= self.dialog_close
                        
                    ),
                    MDFlatButton(
                        text="Yes",
                        font_name= './assets/fonts/IMPACT.TTF',
                        theme_text_color="Custom",
                        text_color= [0.7215, 0.451, 0.2, 1],
                        on_press= self.handle_answer
                    ),
                ],
            )
        self.dialog.open()

    def handle_answer(self, *args):
        self.dialog_close()
        spec = MDApp.get_running_app().spec
        spec.close()

        # this will shutdown the device
        subprocess.call(['sudo', 'shutdown', '-h', 'now'])

