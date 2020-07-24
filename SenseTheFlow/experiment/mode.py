from enum import Enum


class Mode(Enum):
    TRAIN = 'train'
    EVAL = 'eval'
    TEST = 'test'

class Hookpoint(Enum):
    GRADIENT = 0
    LOOP = 1
