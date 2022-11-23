import sys
import os
from entries import Pokedex

from playsound import playsound

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import QSize
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QMainWindow, QWidget, QPushButton

POKEDEX_DISPLAY_WIDTH  = 320
POKEDEX_DISPLAY_HEIGHT = 240

class DexWindow(QMainWindow):
    """ Constructor
    """
    def __init__(self):
        QMainWindow.__init__(self)

        self.entry_current = 0

        self.window_size_max = QSize(POKEDEX_DISPLAY_WIDTH, POKEDEX_DISPLAY_HEIGHT)
        self.window_pokemon_image = QSize(240, 240)

        self.setMaximumSize(self.window_size_max)
        self.setWindowTitle("Pokedex")

        self.image_file = "images/bulbasaur.png"
        self.audio_file = "audio/bulbasaur.wav"

        # Buttons
        self.button_cycle_left = QPushButton("<")
        self.button_cycle_left.clicked.connect(self.decrementEntry)
        self.button_cycle_right = QPushButton(">")
        self.button_cycle_right.clicked.connect(self.incrementEntry)
        self.button_read_entry = QPushButton("Read Entry", self)
        self.button_read_entry.clicked.connect(self.read_entry)

        # Image display layout
        self.entry_image_layout = QtWidgets.QHBoxLayout()
        self.entry_image = QtGui.QPixmap(self.image_file).scaled(self.window_pokemon_image)
        self.entry_image_label = QtWidgets.QLabel()
        self.entry_image_label.setPixmap(self.entry_image)
        self.entry_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.entry_image_layout.addWidget(self.entry_image_label)

        # Entry control layout
        self.entry_control_layout = QtWidgets.QHBoxLayout()
        self.entry_control_widget = QtWidgets.QWidget()
        self.entry_control_widget_layout = QtWidgets.QHBoxLayout(self.entry_control_widget)
        self.entry_control_widget_layout.addWidget(self.button_cycle_left)
        self.entry_control_widget_layout.addWidget(self.button_read_entry)
        self.entry_control_widget_layout.addWidget(self.button_cycle_right)
        self.entry_control_layout.addWidget(self.entry_control_widget)
        
        # Overall layout 
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.entry_image_layout)
        self.layout.addLayout(self.entry_control_layout)
        
        # Central widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    """ Updates the current entry after a change occurs
    """
    def updateEntry(self):
        print("Entry: {}".format(self.entry_current))
        entry = Pokedex.getEntry(self.entry_current)

        # Update audio file 
        self.audio_file = "audio/{}.wav".format(entry.name).lower()
        if(os.path.exists(self.audio_file)):
            print("Audio file found: {}".format(self.audio_file))
        else:
            print("Audio file NOT found: {}".format(self.audio_file))

        # Update image file
        self.image_file = "images/{}.png".format(entry.name).lower()
        if(os.path.exists(self.image_file)):
            print("Image file found: {}".format(self.image_file))
        else:
            print("Image file NOT found: {}".format(self.image_file))
        self.entry_image = QtGui.QPixmap(self.image_file).scaled(self.window_pokemon_image)
        self.entry_image_label.setPixmap(self.entry_image)

    """ Plays the audio for a pokedex entry
    """
    def read_entry(self):
        self.entry_control_widget.setEnabled(False)
        playsound(self.audio_file)
        self.entry_control_widget.setEnabled(True)

    """ Decrements the current selected entry 
    """
    def decrementEntry(self):
        if(self.entry_current > 0):
            # Valid entries only
            self.entry_current = self.entry_current - 1
            self.updateEntry()

    """ Increments the current selected entry
    """
    def incrementEntry(self):
        if(self.entry_current < (Pokedex.numEntries() - 1)):
            # Valid entries only
            self.entry_current = self.entry_current + 1
            self.updateEntry()
