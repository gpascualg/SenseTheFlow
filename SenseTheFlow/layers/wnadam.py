import tensorflow as tf

class WNAdam(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate, global_step, beta1=0.9, use_locking=False, name='WNAdam'):
        super(WNAdam, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._global_step = global_step
        
        self._learning_rate_tensor =  None
        self._beta1_tensor = None
        self._global_step_on_worker = None
        
    def _create_slots(self, var_list):
        for v in var_list:
            with tf.colocate_with(v):
                self._zeros_slot(v, "momentum", self._name)
                self._get_or_make_slot(
                    v, 
                    tf.constant(1.0, shape=v.get_shape(), dtype=v.dtype), 
                    "bias", 
                    self._name
                )

    def _prepare(self):        
        learning_rate = self._learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate()
        self._learning_rate_tensor = tf.convert_to_tensor(learning_rate, name="learning_rate")
        
        beta1 = self._beta1
        if callable(beta1):
            beta1 = beta1()
        self._beta1_tensor = tf.convert_to_tensor(beta1, name="beta1")
        
        with tf.colocate_with(self._learning_rate_tensor):
            self._global_step_on_worker = tf.identity(self._global_step) + 1

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._learning_rate_tensor, var.dtype)
        beta1_t = tf.cast(self._beta1_tensor, var.dtype)
        
        with tf.device(var.device):
            step_t = tf.identity(self._global_step_on_worker)
            step_t = tf.cast(step_t, var.dtype)
        
        m = self.get_slot(var, "momentum")
        b = self.get_slot(var, "bias")

        m_t = m.assign((beta1_t * m) + (1.0 - beta1_t) * grad)
        b_t = b.assign(b + tf.square(lr_t) * tf.reduce_sum(tf.square(grad)) / b)
            
        var_update = tf.assign_sub(var, (lr_t / b_t) * m_t / (1.0 - tf.pow(beta1_t, step_t)))

        return tf.group(*[var_update, m_t, b_t])

    def _resource_apply_dense(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")


