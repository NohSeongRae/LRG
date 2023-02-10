import random
import numpy as np
import os
import tensorflow as tf
import scipy.sparse as sp
def logdir(name):
    path = "./{}/".format(name)
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def set_seed(seed):
    """
    random seed for generation
    :param seed: manual seed from config.yaml
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # torch.cuda.manual_seed(seed)

def to_numpy(x):
    if sp.issparse(x):
        return x.A
    elif hasattr(x, "numpy"):
        return x.numpy()
    elif tf.keras.backend.is_sparse(x):
        return tf.sparse.to_dense(x).numpy()
    else:
        return np.array(x)


def ensure_dir(path):
    """
    check the existence of path, before creating it
    :param path: path
    :return:
    """
    if not os.path.exists(path):
        print("creating folder", path)
        os.makedirs(path)
    else:
        print(path, "path already exist")



def ensure_dirs(paths):
    """
    multiple-path version of ensuer_dir
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)



def gen_model_list(dirname, key):
    """
    generate model list from given directory
    :param dirname: directory that includes .pt
    :param key:
    :return: joined path (path + filename)
    """
    if os.path.exists(dirname) is False:
        print(f'expected directory name {dirname} is not exist!')
        return None
    gen_models=[os.path.join(dirname, f) for f in os.listdir(dirname) if
                os.path.isfile(os.path.join(dirname, f)) and
                key in f and ".pt" in f]
    if gen_models is None or len(gen_models) == 0:
        print(f'gen_model fails!. please check {dirname} includes .pt files')
        return None
    gen_models.sort()
    last_model_name=gen_models[-1]
    return last_model_name


