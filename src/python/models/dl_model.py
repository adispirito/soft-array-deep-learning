import tensorflow as tf
import copy


class DL_Model(tf.keras.Model):
    def __init__(self, 
                 dl_model_func,
                 model_name="DL_Deconv_Model",
                 loss_kwargs={},
                 **kwargs):
        super(DL_Model, self).__init__(name=model_name)
        self.loss_kwargs = {key: tf.cast(val, tf.float32) 
                            for key, val in loss_kwargs.items()}
        
        self.model = dl_model_func(**kwargs)      
    @tf.function
    def call(self, x, **kwargs):
        x = self.model(x, **kwargs)
        return x
    
    
    def get_config(self):
        config = super(DL_Model, self).get_config()
        return config

    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
        
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.loss_func(y, y_pred, **self.loss_kwargs)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Clip Gradients by Norm
        gradients = [(tf.clip_by_norm(grad, clip_norm=5.0)) for grad in gradients]
        # gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

    
    @tf.function
    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        y_pred = self(x, training=False)  # Forward pass
        loss = self.loss_func(y, y_pred, **self.loss_kwargs)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

    
    @tf.function
    def loss_func(self, y, y_pred):
        loss = self.compiled_loss(y, y_pred)
        return loss
    
    @staticmethod
    @tf.function(jit_compile=True)
    def fft_2d(img, **kwargs):
        out = tf.transpose(img, perm=[0, 3, 1, 2])
        out = tf.signal.rfft2d(out, **kwargs)
        out = tf.transpose(out, perm=[0, 2, 3, 1])
        return out

    
    @staticmethod
    @tf.function(jit_compile=True)
    def ifft_2d(img, **kwargs):
        out = tf.transpose(img, perm=[0, 3, 1, 2])
        out = tf.signal.irfft2d(out, **kwargs)
        out = tf.transpose(out, perm=[0, 2, 3, 1])
        return out

    
    @staticmethod
    @tf.function(jit_compile=True)
    def rfft_loss(y_true, y_pred):
        from python.utils.conv_utils import rfft_2d
        loss = tf.math.reduce_mean(
                tf.math.abs(
                    tf.math.squared_difference(rfft_2d(y_true), 
                                               rfft_2d(y_pred))
                )
            )
        return loss
