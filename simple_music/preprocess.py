'''
Author:     John LaGorga 
Project:    JAZZ 
Purpose:    Parse, cleanup and process data.

Code adapted from Ji-Sung Kim's deepjazz, https://github.com/evancchow/jazzml with
express permission.
'''

from __future__ import print_function

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest
from grammar import *

#----------------------------HELPER FUNCTIONS----------------------------------#

def __flatten_stream(s):
    voices = s.getElementsByClass(stream.Voice)
    if len(voices) != 0:
        for voice in voices[1:]:
            for n in voice:
                voices[0].insert(n.offset, n)
        s = voices[0]

    for i in s:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    return s

def __cut_stream(s, offset):
    part = stream.Part()
    part.append(s.getElementsByClass(instrument.Instrument))
    part.append(s.getElementsByClass(tempo.MetronomeMark))
    part.append(s.getElementsByClass(key.KeySignature))
    part.append(s.getElementsByClass(meter.TimeSignature))
    part.append(s.getElementsByOffset(offset[0], offset[1], includeEndBoundary=True))
    return part.flat


''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(data_fn, solo, soffset, accomp, aoffset):

    # Parse the MIDI data for separate melody and accompaniment parts
    midi_data = converter.parse(data_fn)

    # Extract solo stream, from positions ..ByOffset(i, j)
    solo_stream = midi_data[solo]
    solo_stream.show('midi')
    solo_stream = __flatten_stream(solo_stream)
    solo_stream = __cut_stream(solo_stream, soffset)
    
    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    measures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in solo_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get accompaniment, compress into chords
    accomp_stream = midi_data[accomp]
    accomp_stream.show('midi')
    accomp_stream = __flatten_stream(accomp_stream)
    accomp_stream = __cut_stream(accomp_stream, aoffset)

    # Remove everything by chords
    accomp_stream.removeByClass(note.Rest)
    accomp_stream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in accomp_stream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1
    

    # Ensure measures and chords are the same size
    print(min(len(measures), len(chords)))

    while len(chords) > len(measures):
        del chords[len(chords) - 1]

    while len(chords) < len(measures):
        del measures[len(measures) - 1]

    print("\n\n\n CHORDS \n")
    print(chords)

    print("\n\n\n MEASURES \n")
    print(measures)

    assert len(chords) == len(measures)
    return measures, chords


''' Helper function to get the grammatical data from given musical data. '''
def __get_abstract_grammars(measures, chords):
    # extract grammars
    abstract_grammars = []
    for ix in range(1, len(measures)):
        m = stream.Voice()
        for i in measures[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in chords[ix]:
            c.insert(j.offset, j)
        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)

    return abstract_grammars

#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Get musical data from a MIDI file '''
def get_musical_data(data_fn, solo, soffset, accomp, aoffset):
    measures, chords = __parse_midi(data_fn, solo, soffset, accomp, aoffset)
    abstract_grammars = __get_abstract_grammars(measures, chords)

    return chords, abstract_grammars

''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    return corpus, values, val_indices, indices_val
