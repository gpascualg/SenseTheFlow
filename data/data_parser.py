import tensorflow as tf
import multiprocessing

from ..helper import DefaultNamespace


class DataParser(object):
    def __init__(self):
        self.__input_fn = dict([
            (tf.estimator.ModeKeys.TRAIN, []),
            (tf.estimator.ModeKeys.PREDICT, []),
            (tf.estimator.ModeKeys.EVAL, [])
        ])

    def has(self, mode):
        return bool(self.__input_fn[mode])

    def num(self, mode):
        return len(self.__input_fn[mode])

    def from_generator(self, generator, output_types, output_shapes=None, parser_fn=None,
        pre_shuffle=False, post_shuffle=False, flatten=False, 
        skip=None, num_samples=None, batch_size=1,
        mode=tf.estimator.ModeKeys.TRAIN, **kwargs):

        generator = {
            'generator': generator, 
            'output_types': output_types,
            'output_shapes': output_shapes
        }

        input_fn = lambda num_epochs: self.generator_input_fn(
            generator,  parser_fn=parser_fn,
            pre_shuffle=pre_shuffle, post_shuffle=post_shuffle, flatten=flatten, 
            skip=skip, num_samples=num_samples, batch_size=batch_size,
            mode=mode, num_epochs=num_epochs
        )

        self.__input_fn[mode].append((input_fn, DefaultNamespace(**kwargs)))
        
        return self

    def train_from_generator(self, *args, **kwargs):
        self.from_generator(*args, mode=tf.estimator.ModeKeys.TRAIN, **kwargs)
        return self
    
    def eval_from_generator(self, *args, **kwargs):
        self.from_generator(*args, mode=tf.estimator.ModeKeys.EVAL, **kwargs)
        return self
        
    def predict_from_generator(self, *args, **kwargs):
        self.from_generator(*args, mode=tf.estimator.ModeKeys.PREDICT, **kwargs)
        return self

    def generator_input_fn(self, generator, mode, parser_fn=None,
        pre_shuffle=False, post_shuffle=False, flatten=False, 
        skip=None, num_samples=None, batch_size=1, num_epochs=1):

        dataset = tf.data.Dataset.from_generator(**generator)

        # Skip initial X
        if skip is not None:
            dataset = dataset.skip(skip)

        # Take only next X
        if num_samples is not None:
            dataset = dataset.take(num_samples)

        # Shuffling is tricky with repeated datasets
        if not post_shuffle:
            if pre_shuffle:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
                    pre_shuffle, num_epochs))
            else:
                dataset = dataset.repeat(num_epochs)
        elif pre_shuffle:
            dataset.shuffle(pre_shuffle)
        
        num_cpu = multiprocessing.cpu_count()

        # Unless some specific actions must be performed,
        # parallelize everything
        if batch_size > 0 and \
            parser_fn is not None and \
            not flatten and \
            num_samples is None:

            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                map_func=lambda *args: parser_fn(*args, mode=mode), 
                batch_size=batch_size,
                num_parallel_batches=num_cpu))
        
        else:
            if parser_fn is not None:
                dataset = dataset.map(lambda *args: parser_fn(*args, mode=mode), 
                    num_parallel_calls=num_cpu)

            if flatten:
                dataset = dataset.flat_map(lambda *args: tf.data.Dataset.from_tensor_slices((*args,)))

            # Post-parsing shuffle + repeat
            if post_shuffle:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
                    post_shuffle, num_epochs))
        
            # Create batches
            if batch_size > 0:
                dataset = dataset.batch(batch_size)

        # Prefetch to avoid IDLE time
        if batch_size > 0:
            dataset = dataset.prefetch(batch_size)

        # We are done
        iterator = dataset.make_one_shot_iterator()

        if mode == tf.estimator.ModeKeys.PREDICT:
            features = iterator.get_next()
            return features
        
        features, labels = iterator.get_next()
        return features, labels

    def input_fn(self, mode, num_epochs):
        assert bool(self.__input_fn[mode])
        for input_fn, args in self.__input_fn[mode]:
            yield (lambda: input_fn(num_epochs), args)

