from sklearn.model_selection import train_test_split as tts
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense
import sklearn.metrics as sklm
from keras.utils.layer_utils import count_params as c_params
import datetime as dt
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt


rng = np.random.default_rng()


class LogPredsOverDataSLABWVG(tf.keras.callbacks.Callback):
    def __init__(self, data, scaler, file="datalog.csv", frequency=1, compress=False):
        datax, datay = get_raw_data_scaled(data, scaler)
        self.datatotest = datax
        self.path = file
        self.freq = frequency
        self.counter = 0
        self.pd_log = pd.DataFrame(data)
        self.pd_log = self.pd_log.rename(
            columns={0: "wvl", 1: "ncore", 2: "nclad", 3: "width", 4: "result"}
        )
        self.compress = compress

    def on_epoch_end(self, epoch, logs=None):
        if not (epoch + 1) % self.freq and epoch:
            predicts = self.model(self.datatotest)
            #             print(predicts[:10])
            loga = pd.DataFrame(predicts.numpy())
            #             print(loga.head())
            #             if self.pd_log.empty:
            #                 self.pd_log = self.pd_log.append(loga, ignore_index=True)
            #                 self.pd_log = self.pd_log.rename(columns={0: f"{epoch+1}"})
            #             else:
            self.pd_log[f"{epoch+1}"] = loga

    def on_train_end(self, epoch, logs=None):
        if self.compress:
            self.pd_log.to_csv(f"{self.path}", index_label="Index", compression="zip")
        else:
            self.pd_log.to_csv(f"{self.path}", index_label="Index")



# log predictions over given dataset
class LogPredsOverData(tf.keras.callbacks.Callback):
    def __init__(self, data, file="datalog.csv", frequency=1, compress=False):
        self.datatotest = data
        self.path = file
        self.freq = frequency
        self.counter = 0
        self.pd_log = pd.DataFrame([])

    def on_epoch_end(self, epoch, logs=None):
        if not (epoch + 1) % self.freq and epoch:
            predicts = self.model(self.datatotest)
            #             print(predicts[:10])
            loga = pd.DataFrame(predicts.numpy())
            #             print(loga.head())
            if self.pd_log.empty:
                self.pd_log = self.pd_log.append(loga, ignore_index=True)
                self.pd_log = self.pd_log.rename(columns={0: f"{epoch+1}"})
            else:
                self.pd_log[f"{epoch+1}"] = loga

    def on_train_end(self, epoch, logs=None):
        if compress:
            self.pd_log.to_csv(f"{self.path}", index_label="Index", compression="zip")
        else:
            self.pd_log.to_csv(f"{self.path}", index_label="Index")




# trainingslogger
class TrainingToFileLogger(tf.keras.callbacks.Callback):
    def __init__(self, file="training.log", force=False):
        self.path = file
        if os.path.isfile(file):
            if force:
                os.remove(file)
            else:
                print(
                    f"{file} already exists will append, "
                    "use force if this is not desired"
                )
        self.pd_log = pd.DataFrame([])

    def on_epoch_end(self, epoch, logs=None):
        # this seems to be quite slow and takes around 1.5ms
        # when pd_log gets large
        loga = pd.DataFrame(logs, index=[0])
        self.pd_log = self.pd_log.append(loga, ignore_index=True)

    def on_train_end(self, logs=None):
        # this might not be optimal but it ensures the preservation of the loss
        # print(self.pd_log)
        if os.path.isfile(self.path):
            prog_df = pd.read_csv(self.path, index_col=0)
            prog_df = prog_df.append(self.pd_log, ignore_index=True, sort=False)
            prog_df.to_csv(self.path, index_label="Index")
        else:
            prog_df = self.pd_log
            prog_df.to_csv(self.path, index_label="Index")


class TrainingToFileLogger_history(tf.keras.callbacks.Callback):
    def __init__(self, file="training_history.log", force=False):
        self.path = file
        if os.path.isfile(file):
            if force:
                os.remove(file)
            else:
                print(
                    f"{file} already exists will append, "
                    "use force if this is not desired"
                )

    def on_epoch_end(self, epoch, logs=None):
        # print(self.model.history.history)
        pass

    def on_train_end(self, logs=None):
        # this might not be optimal but it ensures the preservation of the loss
        # print(self.pd_log)
        if os.path.isfile(self.path):
            prog_df = pd.read_csv(self.path, index_col=0)
            prog_df = prog_df.append(
                pd.DataFrame(self.model.history.history), ignore_index=True, sort=False
            )
            prog_df.to_csv(self.path, index_label="Index")
        else:
            prog_df = pd.DataFrame(self.model.history.history)
            prog_df.to_csv(self.path, index_label="Index")


def print_training_log(file, onlyloss=True):
    df = pd.read_csv(file, index_col=0)
    fig, ax = plt.subplots(1)
    if onlyloss:

        plt.plot(df["loss"])
        plt.plot(df["val_loss"])
        ax.set_yscale("log")
    else:
        df.plot(logy=True, ax=ax)
    # plt.title("model loss")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.legend(["train", "val"], loc="upper left")
    plt.savefig(
        f"{file}_plot.png",
        dpi=150,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.5,
    )


def count_trainable_params(model):
    return c_params(model.trainable_weights)


def train_test_split(X, y, **kwargs):
    return tts(X, y, **kwargs)


def train_val_split(X, y, **kwargs):
    return tts(X, y, **kwargs)


def train_val_test_split(X, y, train_size, val_size, test_size, random_state=None):
    # due to precision
    if np.absolute(train_size + val_size + test_size - 1) > 0.0005:
        raise ValueError("The ratios must add up to 1")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_val_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_np_sample(data, fraction=-1, sample=-1):
    if fraction == -1 and sample == -1:
        raise NameError("Specify fraction or sample")
    if sample > 0:
        if sample > data.shape[0]:
            raise ValueError("sample size too big")
        return data[rng.choice(data.shape[0], sample, replace=False), :]
    if fraction > 1:
        raise ValueError("No sample size and fraction is bigger than 1")
    if fraction <= 0:
        raise ValueError("fraction smaller or equal to 0")
    return data[
        rng.choice(data.shape[0], int(data.shape[0] * fraction), replace=False), :
    ]


def np_hist(data, column="A", bins=50):
    P1 = pd.DataFrame(data)
    if column == "A":
        return P1.hist(bins=bins)
    else:
        return P1[column].hist(bins=bins)


def get_np_where(data_arr, value, search_column=0, remove_column=False):
    mask = np.where(np.abs(data_arr[:, search_column] - value) <= np.finfo(float).eps)
    if remove_column:
        return np.delete(data_arr[mask], search_column, 1)
    return data_arr[mask]


def n2n1diff_search(data_arr, diff, n2idx, n1idx):
    # mask = np.where(np.abs(data_arr[:, n2idx] - data_arr[:, n1idx]) <= diff+np.finfo(float).eps)
    # return data_arr[mask]
    return data_arr[np.abs((data_arr[:, n2idx] - data_arr[:, n1idx]) - diff) <= 2 * np.finfo(float).eps]

def scale_train_val_test(train, val, test):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    return train, val, test, scaler


def get_easy_model(
    neuron_list,
    input_shape,
    output_shape,
    layer_activation="relu",
    result_activation="linear",
    kernel_regu=None,
    optimizer="sgd",
    loss="mse",
    compile=False,
):
    if not isinstance(input_shape, tuple):
        raise TypeError("Input shape needs to be a tuple")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for n in neuron_list:
        model.add(Dense(n, activation=layer_activation, kernel_regularizer=kernel_regu))
    model.add(Dense(output_shape, activation=result_activation))
    if compile:
        model.compile(optimizer=optimizer, loss=loss)
    return model


def get_metric_scores(y_true, y_pred):
    output = {
        "exv": sklm.explained_variance_score(y_true=y_true, y_pred=y_pred),
        "maxerr": sklm.max_error(y_true=y_true, y_pred=y_pred),
        "mae": sklm.mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "mse": sklm.mean_squared_error(y_true=y_true, y_pred=y_pred),
        "rmse": sklm.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
        "medae": sklm.median_absolute_error(y_true=y_true, y_pred=y_pred),
        "r2": sklm.r2_score(y_true=y_true, y_pred=y_pred),
    }
    if (y_true > 0).all() and (y_pred > 0).all():
        output["msle"]= sklm.mean_squared_log_error(y_true=y_true, y_pred=y_pred),
        output["mpd"] = sklm.mean_poisson_deviance(y_true=y_true, y_pred=y_pred)
        output["mgd"] = sklm.mean_gamma_deviance(y_true=y_true, y_pred=y_pred)
    return output


def printtime(offset=1):
    import datetime as dt

    today = dt.datetime.now()
    offset = dt.timedelta(hours=offset)
    print((today + offset).strftime("%Y-%m-%d %H:%M:%S"))


def get_sample_data_to_train(raw_data, sample_size, train_size=0.8, val_size=0.1, test_size=0.1):
    t_sam = get_np_sample(raw_data, sample=sample_size)
    X = t_sam[:, :-1]
    y = t_sam[:, -1]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=train_size, val_size=val_size, test_size=test_size
    )
    #     print(X.shape)
    print(X_train.shape)
    #     print(X_val.shape)
    #     print(X_test.shape)
    X_train, X_val, X_test, scaler = scale_train_val_test(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def get_raw_data_scaled(raw_data, scaler):
    X = raw_data[:, :-1]
    y = raw_data[:, -1]
    X = scaler.transform(X)
    return X, y


def save_predictions(predictions, y_true, savepath):
    metrics = get_metric_scores(y_true, predictions)
    with open(f"{savepath}_metrics.txt", "w", encoding="utf-8") as wr:
        counterstr = ""
        for k, v in metrics.items():
            counterstr += f"{k}, {v}\n"
        wr.write(counterstr)
    header = "Y_true, Predictions, Error\n"
    with open(f"{savepath}_pred.txt", "w", encoding="utf-8") as wr:
        predstr = ""
        for idx in range(len(predictions)):
            predstr += f"{y_true[idx]:.10f}, {predictions[idx]:.10f}, {(y_true[idx]-predictions[idx]):.10f}\n"
        wr.write(header)
        wr.write(predstr)


def colab_path_creation(spath, dpath):
    spath = spath.replace("[", "-").replace("]", "+")
    if not Path(spath).is_dir():
        Path(spath).mkdir()
    if not Path(f"{dpath}{spath}").is_dir():
        print("creating drivepath")
        Path(f"{dpath}{spath}").mkdir()


def archive_name_creation(arcname):
    return arcname.lower().replace("/", "%")


def save_data_ckpt(savedir, drivedir):
    savedir = savedir.replace("[", "-").replace("]", "+")
    fname = archive_name_creation(savedir)
    # print("saving tar.xz")
    # basedir = os.path.dirname(savedir)
    # shutil.make_archive(f"{basedir}/{fname}", "zip", savedir)
    shutil.make_archive(f"{fname}", "zip", savedir)
    shutil.move(f"{fname}.zip", f"{drivedir}{savedir}/{fname}.zip")


def check_if_already_trained(trained_dir, drivedir):
    trained_dir = trained_dir.replace("[", "-").replace("]", "+")
    fname = archive_name_creation(trained_dir)
    if Path(f"{drivedir}{trained_dir}/{fname}.zip").is_file():
        print(f"{drivedir}{trained_dir}/{fname}.zip exists: continue")
        return True
    print(f"{drivedir}{trained_dir}/{fname}.zip")
    return False


def get_save_name(
    input_size,
    network_size,
    output_size,
    sample_size,
    batch_size=None,
    trainable_count=None,
):
    if trainable_count is None:
        if batch_size is None:
            temp = f"{input_size[0]}${network_size}${output_size}@{sample_size}"
            temp = temp.replace("]", "+").replace("[", "-")
            return temp
        temp = (
            f"{input_size[0]}${network_size}${output_size}@{sample_size}@{batch_size}"
        )
        temp = temp.replace("]", "+").replace("[", "-")
        return temp
    if batch_size is None:
        temp = f"{trainable_count}_{input_size[0]}${network_size}${output_size}@{sample_size}"
        temp = temp.replace("]", "+").replace("[", "-")
        return temp
    temp = f"{trainable_count}_{input_size[0]}${network_size}${output_size}@{sample_size}@{batch_size}"
    temp = temp.replace("]", "+").replace("[", "-")
    return temp


def get_data_path(
    curdir,
    input_size,
    network_size,
    output_size,
    sample_size,
    batch_size=None,
    trainable_count=None,
):
    temp = get_save_name(
        input_size=input_size,
        network_size=network_size,
        output_size=output_size,
        sample_size=sample_size,
        batch_size=batch_size,
        trainable_count=trainable_count,
    )
    return f"{curdir}/{temp}".replace("]", "+").replace("[", "-")

def get_rnd_unif(a,b,n=1000):
    if a>=b:
        print("the interval is [a,b) 'a' can't be bigger than 'b'")
        raise ValueError
    return (b - a) * rng.random((n,1)) + a
# model.fit
# losses = pd.DataFrame
# predictions = model.predictions
# losses = losses.append()
# predictions.extend/append


# EPOCHS = 10
# checkpoint_filepath = '/tmp/checkpoint'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_acc',
#     mode='max',
#     save_best_only=True)

# # Model weights are saved at the end of every epoch, if it's the best seen
# # so far.
# model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# # The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)
# string or PathLike, path to save the model file. filepath can contain named
# formatting options, which will be filled the value of epoch and keys in
#  logs(passed in on_epoch_end). For example: if filepath is
#   weights.{epoch: 02d} - {val_loss: .2f}.hdf5, then the model checkpoints
# will be saved with the epoch number and the validation loss in the filename.
# history.model.get_config()

# model(x) ist 10mal so schnell wie model.predict(x) aber Diff ist 1E-6

# for batchsize in 32 64 128 256 512 do
#     model = getmodel
#     model train
#     save losses
#     save model
#     test prediction?
#     save prediction
#
