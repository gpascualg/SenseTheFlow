from enum import Enum


class Mode(Enum):
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'