from enum import Enum


class Mode(Enum):
    TRAIN = 'train'
    EVAL = 'eval'
    TEST = 'test'
    ANY = 'any'

class Hookpoint(Enum):
    GRADIENT = 0
    POST_INITIALIZATION = 1
    LOOP = 2
    END = 3
