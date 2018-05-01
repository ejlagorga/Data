'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Generate jazz using a deep learning model (LSTM in deepjazz).

Some code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml 
with express permission.

Code was built while significantly referencing public examples from the
Keras documentation on GitHub:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generator.py [# of epochs]

    Note: running Keras/Theano on GPU is formally supported for only NVIDIA cards (CUDA backend).
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
