import tensorflow as tf

from .utils import transpose_batch_time


def _wrap_loop_fn(loop_fn, dynaState):
    def _wrapped_loop_fn(time, cell_output, cell_state, loop_state):
        if cell_output is not None:
            dynaState.decode(loop_state)
        
        elements_finished, next_input, cell_state, emit_output = \
            loop_fn(time, cell_output, cell_state, dynaState)

        return (
            elements_finished,
            next_input,
            cell_state,
            emit_output, 
            dynaState.encode()
        )

    return _wrapped_loop_fn

def raw_rnn(cell, loop_fn, dynaState):
    outputs_ta, final_state, loop_state = tf.nn.raw_rnn(cell, _wrap_loop_fn(loop_fn, dynaState))
    outputs = transpose_batch_time(outputs_ta.stack())
    dynaState.decode(loop_state)

    return outputs, final_state, dynaState
