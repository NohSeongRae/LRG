from spektral.layers import GeneralConv
from spektral.layers import GCNConv
from spektral.models.general_gnn import MLP
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Concatenate


class Autoencoder(Model):
    def __init__(self, n_features, pool, lift, batch_norm=False, post_procesing=True):
        super().__init__()
        #encoder
        self.pre = MLP(256, activation="relu", batch_norm=batch_norm)
        self.gnn1 = GeneralConv(activation="relu", batch_norm=batch_norm)
        self.gnn_shared=GeneralConv(activation="relu", batch_norm=batch_norm)
        self.skip = Concatenate()
        self.pool = pool

        #decoder
        self.lift = lift

        self.post_processing = post_procesing
        if post_procesing:
            self.gnn2 = GeneralConv(activation="relu", batch_norm=batch_norm)
            self.post = MLP(
                n_features,
                activation="relu",
                final_activation="linear",
                batch_norm=batch_norm,
            )

    def call(self, inputs):
        #encoder
        if len(inputs) == 2:
            x, a = inputs
            s = None
        elif len(inputs) == 3:
            x, a, s = inputs
        else:
            raise ValueError("Input must be [x, a] or [x, a, s].")

        x = self.pre(x)
        x = self.skip([self.gnn1([x, a]), x])

        pool_inputs = [x, a]
        if s is not None:
            pool_inputs.append(s)
        pool_outputs = list(self.pool(pool_inputs))
        if s is not None:
            pool_outputs.append(s)
        else:
            s = pool_outputs[2]
        x_pool, a_pool = pool_outputs[:2]
        z_mean=self.skip([self.gnn_shared([x_pool, a_pool]), x_pool])
        z_log_std = self.skip([self.gnn_shared([x_pool, a_pool]), x_pool])
        z=z_mean+tf.random.normal([len(x_pool), 256])*tf.exp(z_log_std)
        #decoder

        a_pool=tf.matmul(z, tf.transpose(z))
        pool_outputs[1]=a_pool
        x_lift, a_lift = self.lift(pool_outputs)

        if self.post_processing:
            x_lift = self.skip([self.gnn2([x_lift, a]), x_lift])
            x_lift = self.post(x_lift)

        return x_lift, a_lift, s, x_pool, a_pool, z_mean, z_log_std