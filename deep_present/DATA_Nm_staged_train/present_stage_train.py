
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import present
import numpy as np
import multiprocessing as mp
import sys 
sys.path.append("..") 
wdir = "./temp_distinguisher/"
n = 10**7
test_n = int(n/10)
num_rounds = 8
pairs = 2
bs = 1000


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


def first_stage(n, num_rounds=7, pairs=8):
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():
        net = load_model("../DATA_Nm_good_trained_nets/present_best_"+str(num_rounds-1) + 'r_pairs' +
                         str(pairs) + '_distinguisher.h5')
        net_json = net.to_json()
        net_first = model_from_json(net_json)
        net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_first.load_weights("../DATA_Nm_good_trained_nets/present_best_"+str(num_rounds-1) + 'r_pairs' +
                               str(pairs) + '_distinguisher.h5')

    process_number = 50
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(present.make_dataset_with_group_size, [(int(
            n*pairs/process_number), num_rounds-2, 0x0000000001000100, pairs,) for i in range(process_number)])

    X = accept_XY[0][0]
    Y = accept_XY[0][1]

    for i in range(process_number-1):
        X = np.concatenate((X, accept_XY[i+1][0]))
        Y = np.concatenate((Y, accept_XY[i+1][1]))

    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(present.make_dataset_with_group_size, [(int(
            test_n*pairs/process_number), num_rounds-2, 0x0000000001000100, pairs,) for i in range(process_number)])
    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]

    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval, accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval, accept_XY_eval[i+1][1]))
    print("multiple processing end ......")


    check = make_checkpoint(
        wdir+'present_first_best_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')
    net_first.fit(X, Y, epochs=10, batch_size=batch_size,
                  validation_data=(X_eval, Y_eval), callbacks=[check])

    net_first.save(wdir+'present_first_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')


def second_stage(n, num_rounds=9, pairs=8):

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    process_number = 50
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(present.make_dataset_with_group_size, [(int(
            n*pairs/process_number), num_rounds, 0x9, pairs,) for i in range(process_number)])
    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    for i in range(process_number-1):
        X = np.concatenate((X, accept_XY[i+1][0]))
        Y = np.concatenate((Y, accept_XY[i+1][1]))

    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(present.make_dataset_with_group_size, [(int(
            test_n*pairs/process_number), num_rounds, 0x9, pairs,) for i in range(process_number)])
    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]

    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval, accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval, accept_XY_eval[i+1][1]))
    print("multiple processing end ......")

    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model(wdir+'present_first_best_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')
        net_json = net.to_json()
        net_second = model_from_json(net_json)
        net_second.compile(optimizer=Adam(
            learning_rate=10**-4), loss='mse', metrics=['acc'])
        net_second.load_weights(
            wdir+'present_first_best_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')

    check = make_checkpoint(
        wdir+'present_second_best_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')
    net_second.fit(X, Y, epochs=4, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval), callbacks=[check])

    net_second.save(wdir+'present_second_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')


def stage_train(n, num_rounds=9, pairs=8):
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    process_number = 50
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(present.make_dataset_with_group_size, [(int(
            n*pairs/process_number), num_rounds, 0x9, pairs,) for i in range(process_number)])

    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    for i in range(process_number-1):
        X = np.concatenate((X, accept_XY[i+1][0]))
        Y = np.concatenate((Y, accept_XY[i+1][1]))

    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(present.make_dataset_with_group_size, [(int(
            test_n*pairs/process_number), num_rounds, 0x9, pairs,) for i in range(process_number)])
    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]

    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval, accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval, accept_XY_eval[i+1][1]))
    print("multiple processing end ......")
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model(wdir+'present_second_best_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')
        net_json = net.to_json()
        net_third = model_from_json(net_json)
        net_third.compile(optimizer=Adam(learning_rate=10**-5),
                          loss='mse', metrics=['acc'])
        net_third.load_weights(
           wdir+'present_second_best_'+str(num_rounds)+"_pairs"+str(pairs)+'_distinguisher.h5')

    check = make_checkpoint(
        'present_best_'+str(num_rounds)+"r_pairs"+str(pairs)+'_distinguisher.h5')
    h = net_third.fit(X, Y, epochs=4, batch_size=batch_size,
                  validation_data=(X_eval, Y_eval), callbacks=[check])
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    net_third.save('present_best_'+str(num_rounds)+"r_pairs"+str(pairs)+'_distinguisher.h5')



first_stage(n=10**7, num_rounds=num_rounds, pairs=pairs)
second_stage(n=10**7, num_rounds=num_rounds, pairs=pairs)
stage_train(n=10**7, num_rounds=num_rounds, pairs=pairs)
