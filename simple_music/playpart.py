'''
Author:     Eli 
Project:    simplemusic
'''
from __future__ import print_function
import sys

from music21 import *
import numpy as np

''' Runs generate() -- generating, playing, then storing a musical sequence --
    with the default Metheny file. '''
def main(args):
    # i/o settings
    data_fn = str(args[1])
    melody = int(args[2])

    # Parse the MIDI data for separate melody and accompaniment parts
    midi_data = converter.parse(data_fn)
    
    # Get melody part, compress into single voice
    melody_stream = midi_data[melody]
    melody_stream.show('midi')


''' If run as script, execute main '''
if __name__ == '__main__':
    import sys
    main(sys.argv)
