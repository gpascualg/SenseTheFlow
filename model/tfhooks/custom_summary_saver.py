import tensorflow as tf


class CustomSummarySaverHook(tf.compat.v1.train.SummarySaverHook):
    def __init__(self,
               save_steps=None,
               save_secs=None,
               output_dir=None,
               summary_writer=None,
               scaffold=None,
               summary_op=None):

        tf.compat.v1.train.SummarySaverHook.__init__(self, 
            save_steps=save_steps,
            save_secs=save_secs,
            output_dir=output_dir,
            summary_writer=summary_writer,
            scaffold=scaffold,
            summary_op=summary_op)

    def _get_summary_op(self):
        tensors = [x for x in tf.get_collection(tf.GraphKeys.SUMMARIES) if x.op.name in self._summary_op]

        if len(tensors) != len(self._summary_op):
            tf.logging.error('Some tensors where not found')
            tnames = [x.op.name for x in tensors]
            tf.logging.error(set(self._summary_op) - set(tnames))

        return tensors
