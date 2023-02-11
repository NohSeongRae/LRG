import time
import os
import sys

import numpy as np
import tensorflow as tf
from spektral.layers import ops
from spektral.utils import sparse
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from model import Autoencoder
from simple_data import make_simple_dataset
from etc.utils import logdir, to_numpy
from upsampling import upsampling_with_pinv

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)


def loss_fn(X, X_pred):  # please modify this later: custom loss
    """
    compute loss
    :param X: ground truth
    :param X_pred: model output
    :return: loss
    """
    loss = tf.keras.losses.mean_squared_error(X, X_pred)
    loss = tf.reduce_mean(loss)
    return loss


def downsampling(inputs):
    """
    downsampling method that perfroms in learning-time
    :param inputs: (X: node feature matrix), (A: adjacency matrix), (S: selection matrix obtained by NDP)
    :return: (X_pool: pooled node feature matrix), (A_pool: pooled adj matrix)
    """
    X, A, S = inputs
    return ops.modal_dot(S, X, transpose_a=True), ops.matmul_at_b_a(S, A)


def create_model(F):
    """
    initialize model
    :param F: input feature dimension (F = X.shape[-1])
    :return: initialized model (LRG)
    """
    pool = Lambda(downsampling)
    lift = Lambda(upsampling_with_pinv)
    model = Autoencoder(F, pool, lift)
    return model


def main(X, A, S, learning_rate, es_patience, es_tol, pos_weight, norm,
         A_label):  # please modify this later: multiple S for hierarchical pooling
    """
    buildi model and set up training
    :param X: node feature matrix
    :param A: adj matrix
    :param S: selection matrix obtained by NDP
    :param learning_rate: learning rate
    :param es_patience: patience parameter for early stopping
    :param es_tol: tolerance parameter for early stopping
    :return: model, training times
    """
    K.clear_session()
    F = X.shape[-1]
    model = create_model(F)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(model, optimizer, X, A, S):
        with tf.GradientTape() as tape:
            X_pred, A_pred, _, _, _, model_z_mean, model_z_log_std, L_A_pred = model([X, A, S], training=True)
            X_loss = loss_fn(X, X_pred)
            # L_A_pred=tf.reshape(A_pred, [-1])
            # L_A_label=tf.reshape(A_label, [-1])
            # rec_loss=norm*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=L_A_pred, labels=A_label.astype("f4"), pos_weight=pos_weight))
            # rec_loss = norm * tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=L_A_pred, labels=A_label.astype("f4")))
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            rec_loss = norm * bce(A_label.astype("f4"), L_A_pred)
            kl_loss = (0.5 / len(X)) * tf.reduce_mean(
                tf.reduce_sum(1 + 2 * model_z_log_std - tf.square(model_z_mean) - tf.square(tf.exp(model_z_log_std)),
                              1))
            # please modify this later: custom loss function
            total_loss = 10 * X_loss + 2 * rec_loss + kl_loss + sum(model.losses)
            loss_dic = {"X_loss": X_loss, "rec_loss": rec_loss, "kl_loss": kl_loss, "total_loss": total_loss}
        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return total_loss, loss_dic

    # fit model, early stopping
    patience = es_patience
    best_loss = np.inf
    best_weights = None
    training_times = []
    ep = 0

    while True:
        ep += 1
        timer = time.time()
        loss_out, loss_dic = train_step(model, optimizer, X, A, S)
        training_times.append(time.time() - timer)
        if loss_out + es_tol < best_loss:
            best_loss = loss_out
            patience = es_patience
            best_weights = model.get_weights()
            print("Epoch {} - New best loss: {:.4e}, X_loss : {:.4e}, rec_loss:{:.4e}, kl_loss:{:.4e}".format(ep,
                                                                                                              best_loss,
                                                                                                              loss_dic[
                                                                                                                  "X_loss"],
                                                                                                              loss_dic[
                                                                                                                  "rec_loss"],
                                                                                                              loss_dic[
                                                                                                                  "kl_loss"]))
        else:
            patience -= 1
            if patience == 0:
                break

    model.set_weights(best_weights)
    return model, training_times


def run_experiment(name, method, pooling, learning_rate, es_patience, es_tol, runs):
    """
    initialize the total experiment
    :param name: dataset name
    :param method: (pooling method name) * please modify here later
    :param pooling: wrapper function
    :param learning_rate: lr_rate
    :param es_patience: patience parameter for early stopping
    :param es_tol:tolerance parameter for early stopping
    :param runs: experiment repeat time
    :return: avg_results, std_results
    """

    # save dir
    log_dir = logdir(name)

    # data load
    A, X, _ = make_simple_dataset(name)
    A_label = A.toarray().reshape([-1])
    A_label.astype("f4")

    # pooling
    A, X, A_pool, S = pooling(X, A)
    # A_label = A.toarray().reshape([-1])
    X = np.array(X)
    S = to_numpy(S)

    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight.astype("f4")
    norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
    # A_label = sparse.sp_matrix_to_sp_tensor(A)
    A = sparse.sp_matrix_to_sp_tensor(A.astype("f4"))

    # summary writer
    # writer = tf.summary.create_file_writer("summaries")

    # #checkpoint setting
    # checkpoint_path="saved_checkpoints/{}_{}_matrices.ckpt".format(method, name)
    # checkpoint_dir=os.path.dirname(checkpoint_path)
    # Run main
    results = []
    for r in range(runs):
        print(f"{r + 1} of {runs}")
        model, training_times = main(
            X=X,
            A=A,
            S=S,
            learning_rate=learning_rate,
            es_patience=es_patience,
            es_tol=es_tol,
            pos_weight=pos_weight,
            norm=norm,
            A_label=A_label,
        )
        # evaluation
        X_pred, A_pred, _, _, _, model_z_mean, model_z_log_std, L_A_pred = model([X, A, S], training=True)
        X_loss = loss_fn(X, X_pred)
        # L_A_pred = tf.reshape(A_pred, [-1])
        # L_A_label = tf.reshape(A_label, [-1])
        # rec_loss=norm*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=L_A_pred, labels=A_label.astype("f4"), pos_weight=pos_weight))
        # rec_loss = norm * tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=L_A_pred, labels=A_label.astype("f4")))
        bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        rec_loss=norm*bce(A_label.astype("f4"),L_A_pred)
        kl_loss = (0.5 / len(X)) * tf.reduce_mean(
            tf.reduce_sum(1 + 2 * model_z_log_std - tf.square(model_z_mean) - tf.square(tf.exp(model_z_log_std)), 1))
        # please modify this later: custom loss function
        total_loss = 10 * X_loss + 2 * rec_loss + kl_loss + sum(
            model.losses)  # please modify this later: custom loss function
        loss_dic = {"X_loss": X_loss, "rec_loss": rec_loss, "kl_loss": kl_loss, "total_loss": total_loss}
        # with writer.as_default():
        #     tf.summary.scalar('loss', total_loss, step=r)
        results.append(total_loss)
        # model.save_weights('./saved_checkpoints')
        print("Final Loss: {:.4e}".format(total_loss))  # please modify this later: custom loss function

    avg_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    # run trained model to get pooled graph
    X_pred, A_pred, _, _, _, _, _, _ = model([X, A, S], training=False)

    # if there is selection mask, convert selection mask to selection matrix
    # S = to_numpy(S)
    # if S.dim == 1:
    #     S = np.eye(S.shape[0])[:, S.astype(bool)]

    # save data for plotting
    np.savez(
        log_dir + "{}_{}_matrices.npz".format(method, name),
        X=to_numpy(X),
        A=to_numpy(A),
        X_pred=to_numpy(X_pred),
        A_pred=to_numpy(A_pred),
        S=to_numpy(S),
        loss=total_loss,
        training_times=training_times,
    )

    return avg_results, std_results, loss_dic


def results_to_file(dataset, method, avg_results, std_results, loss_dic):
    filename = "{}_result.csv".format(dataset)
    with open(filename, "a") as f:
        line = "{}, {} +- {},X_loss, {}, rec_loss, {}, kl_loss, {}, total_loss, {} \n".format(method, avg_results,
                                                                                              std_results,
                                                                                              loss_dic["X_loss"],
                                                                                              loss_dic["rec_loss"],
                                                                                              loss_dic["kl_loss"],
                                                                                              loss_dic["total_loss"])
        f.write(line)
