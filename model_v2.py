from spektral.layers import GeneralConv
from spektral.utils import sp_matrix_to_sp_tensor
from spektral.layers import GCNConv
from spektral.models.general_gnn import MLP
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Concatenate


class VAE(Model):
    def __init__(self, n_features, batch_norm=False, post_procesing=True):
        super().__init__()
        # encoder
        self.pre = MLP(256, activation="relu", batch_norm=batch_norm)
        self.gnn1 = GeneralConv(activation="relu", batch_norm=batch_norm)
        self.gnn_shared = GeneralConv(activation="relu", batch_norm=batch_norm)
        self.skip = Concatenate()


        # decoder


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
        # encoder
        if len(inputs) == 2:
            x, a = inputs
        else:
            raise ValueError("Input must be [x, a] ")
        x = self.pre(x)
        x = self.skip([self.gnn1([x, a]), x])



        z_mean = self.skip([self.gnn_shared([x, a]), x])
        z_log_std = self.skip([self.gnn_shared([x, a]), x])
        z = z_mean + tf.random.normal([len(x), 768]) * tf.exp(z_log_std)
        # decoder

        # a_pool = tf.matmul(z, tf.transpose(z))
        # pool_outputs[1] = a_pool
        # pool_outputs[0]=z
        # x_lift, a_lift = self.lift(pool_outputs)
        A_pred=tf.matmul(z, tf.transpose(z))


        if self.post_processing:
            # a_lift = tf.sparse.from_dense(a_lift)
            X_pred = self.skip([self.gnn2([x, a]), x])
            X_pred = self.post(X_pred)

        return X_pred, A_pred,  z_mean, z_log_std