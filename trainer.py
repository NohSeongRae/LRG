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
from model_v2 import VAE
from simple_data import make_simple_dataset
from etc.utils import logdir, to_numpy
from upsampling import upsampling_with_pinv


physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

from Losses import loss_fn, loss_rec_bce, loss_KL, loss_rec_weight



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
    model = VAE(F)
    return model


def main(X, A,  learning_rate, es_patience, es_tol, pos_weight, norm,
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
    def train_step(model, optimizer, X, A):
        with tf.GradientTape() as tape:
            X_pred, A_pred, model_z_mean, model_z_log_std = model([X, A], training=True)
            # X_loss = loss_fn(X, X_pred)
            rec_loss = norm * loss_rec_weight(A_pred, A_label,pos_weight)
            kl_loss = loss_KL(X, model_z_mean, model_z_log_std)
            # total_loss = 10 * X_loss + 2 * rec_loss + kl_loss + sum(model.losses)
            total_loss =   2 * rec_loss + kl_loss + sum(model.losses)
            # loss_dic = {"X_loss": X_loss, "rec_loss": rec_loss, "kl_loss": kl_loss,
            #             "total_loss": total_loss}
            loss_dic = { "rec_loss": rec_loss, "kl_loss": kl_loss,
                        "total_loss": total_loss}
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
        loss_out, loss_dic = train_step(model, optimizer, X, A)
        training_times.append(time.time() - timer)
        if loss_out + es_tol < best_loss:
            best_loss = loss_out
            patience = es_patience
            best_weights = model.get_weights()
            # print("Epoch {} - New best loss: {:.4e}, X_loss : {:.4e}, rec_loss:{:.4e}, kl_loss:{:.4e}".format(ep,
            #                                                                                                   best_loss,
            #                                                                                                   loss_dic[
            #                                                                                                       "X_loss"],
            #                                                                                                   loss_dic[
            #                                                                                                       "rec_loss"],
            #                                                                                                   loss_dic[
            #                                                                                                       "kl_loss"]))
            print("Epoch {} - New best loss: {:.4e}, rec_loss:{:.4e}, kl_loss:{:.4e}".format(ep,
                                                                                                              best_loss,

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


    X = np.array(X)


    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight.astype("f4")
    norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)

    A = sparse.sp_matrix_to_sp_tensor(A.astype("f4"))


    # Run main
    results = []
    for r in range(runs):
        print(f"{r + 1} of {runs}")
        model, training_times = main(
            X=X,
            A=A,
            learning_rate=learning_rate,
            es_patience=es_patience,
            es_tol=es_tol,
            pos_weight=pos_weight,
            norm=norm,
            A_label=A_label,
        )
        # evaluation
        X_pred, A_pred, model_z_mean, model_z_log_std = model([X, A], training=True)
        # X_loss = loss_fn(X, X_pred)
        rec_loss = norm * loss_rec_weight(A_pred, A_label, pos_weight)
        kl_loss = loss_KL(X, model_z_mean, model_z_log_std)
        # total_loss = 10 * X_loss + 2 * rec_loss + kl_loss + sum(model.losses)
        total_loss = 2 * rec_loss + kl_loss + sum(model.losses)
        # loss_dic = {"X_loss": X_loss, "rec_loss": rec_loss, "kl_loss": kl_loss,
        #             "total_loss": total_loss}
        loss_dic = {"rec_loss": rec_loss, "kl_loss": kl_loss,
                    "total_loss": total_loss}
        results.append(total_loss)

        print("Final Loss: {:.4e}".format(total_loss))  # please modify this later: custom loss function

    avg_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    # run trained model to get pooled graph
    X_pred, A_pred, _, _ = model([X, A], training=False)



    # save data for plotting
    np.savez(
        log_dir + "{}_{}_matrices.npz".format(method, name),
        X=to_numpy(X),
        A=to_numpy(A),
        X_pred=to_numpy(X_pred),
        A_pred=to_numpy(A_pred),
        loss=total_loss,
        training_times=training_times,
    )

    return avg_results, std_results, loss_dic


def results_to_file(dataset, method, avg_results, std_results, loss_dic):
    filename = "{}_result.csv".format(dataset)
    with open(filename, "a") as f:
        line = "{}, {} +- {}, rec_loss, {},  kl_loss, {}, total_loss, {} \n".format(method, avg_results,
                                                                                              std_results,
                                                                                              loss_dic["rec_loss"],

                                                                                              loss_dic["kl_loss"],
                                                                                              loss_dic["total_loss"])
        f.write(line)
