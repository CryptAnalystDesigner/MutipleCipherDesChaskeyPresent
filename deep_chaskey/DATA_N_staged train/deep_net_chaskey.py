import numpy as np
from pickle import dump

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, concatenate, GlobalAveragePooling2D, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

import chaskey as chk
import tensorflow as tf
import multiprocessing as mp
from tensorflow.nn import dropout

bs = 1000
# wdir = './DATA_Nm_good_trained_nets/'
wdir = './'


def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


# make residual tower of convolutional blocks
def make_resnet(group_size=2, num_blocks=4, num_filters=32, num_outputs=1,  word_size=32, ks=3, depth=5, reg_param=0.001, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(group_size * 2 * num_blocks * word_size, ))         
    rs = Reshape((group_size, 2 * num_blocks, word_size) )(inp)      
    perm = Permute((1, 3, 2))(rs)  

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=5, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv03 = Conv1D(num_filters, kernel_size=8, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)

    c2 = concatenate([conv01, conv02, conv03], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters*3, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*3, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2
    # add prediction head
    # 展开，全连接层
    dense0 = GlobalAveragePooling2D()(shortcut)
    dense0 = dropout(dense0, 0.8)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense0)
    model = Model(inputs=inp, outputs=out)
    return(model)


def train_chaskey_distinguisher(num_epochs, num_rounds=7, diff=(0x8400, 0x0400, 0, 0), group_size=2, depth=1):
    # create the network
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync

    with strategy.scope():
        net = make_resnet(group_size=group_size, depth=depth, reg_param=10**-5)
        # net.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net.compile(optimizer=Adam(learning_rate =0.004), loss='mse', metrics=['acc'])
        
    # generate training and validation data
    # X, Y, X_eval, Y_eval = chk.create_train_test_dataset(n1=10**7, n2=10**6, x=num_rounds, y=0,
    #                                                      head=0, group_size=group_size, diff=diff)

    process_number = 50
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(chk.create_train_test_dataset, [(int(10**7/process_number), int(
            10**6/process_number), num_rounds, 0, 0, group_size, diff,) for i in range(process_number)])
    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    X_eval = accept_XY[0][2]
    Y_eval = accept_XY[0][3]

    for i in range(process_number-1):
        X = np.concatenate((X, accept_XY[i+1][0]))
        Y = np.concatenate((Y, accept_XY[i+1][1]))
        X_eval = np.concatenate((X, accept_XY[i+1][2]))
        Y_eval = np.concatenate((Y, accept_XY[i+1][3]))
    print("multiple processing end ......")

    # set up model checkpoint
    check = make_checkpoint(
        wdir+'chaskey_best_'+str(num_rounds)+'r_pairs'+str(group_size)+"_diff_"+str(diff)+'_distinguisher.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    #train and evaluate
    # h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
    #             validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                validation_data=(X_eval, Y_eval), callbacks=[check])
    np.save(wdir+'h'+str(num_rounds)+'r_pairs' +
            str(group_size)+"diff_"+str(diff)+'.npy', h.history['val_acc'])
    np.save(wdir+'h'+str(num_rounds)+'r_pairs' +
            str(group_size)+"diff_"+str(diff)+'.npy', h.history['val_loss'])
    dump(h.history, open(wdir+'hist'+str(num_rounds)+'r_pairs'+str(group_size)+"_diff_"+str(diff)+'.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))

    net.save(wdir + "chaskey_"+str(num_rounds) + '_pairs' +
             str(group_size) + "_diff_"+str(diff)+'_distinguisher.h5')

    return(net, h)


if __name__ == "__main__":

    for r in [3]:
        for gs in [16]:

            print("round = {} and group_size = {} ".format(r,gs))
            # train_chaskey_distinguisher(20, num_rounds=r, diff=(
            #     0x8400, 0x0400, 0, 0), group_size=gs, depth=5)
            train_chaskey_distinguisher(20, num_rounds=r, diff=(
                0x80000000, 0x00000000, 0x00000000, 0x80000000), group_size=gs, depth=5)
