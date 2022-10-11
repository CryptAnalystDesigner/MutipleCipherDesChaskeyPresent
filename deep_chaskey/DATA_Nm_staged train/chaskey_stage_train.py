import tensorflow as tf 
import chaskey as chk
from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import multiprocessing as mp
import numpy as np

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)

wdir = "./temp_distinguisher/"
n = 10**7
test_n = int(n/10)
num_rounds = 4
bs = 1000

process_number = 50

def first_stage(pairs):
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(chk.create_train_test_dataset, [(int(10**7*pairs/process_number), int(
            10**6*pairs/process_number), num_rounds, 0, 0, pairs, (0x8400, 0x0400, 0, 0),) for i in range(process_number)])
    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    X_eval = accept_XY[0][2]
    Y_eval = accept_XY[0][3]
    for i in range(process_number-1):
        X = np.concatenate((X, accept_XY[i+1][0]))
        Y = np.concatenate((Y, accept_XY[i+1][1]))
        X_eval = np.concatenate((X, accept_XY[i+1][2]))
        Y_eval = np.concatenate((Y, accept_XY[i+1][3]))
    # print("multiple processing end ......")

    # X, Y, X_eval, Y_eval = chk.create_train_test_dataset(n1=n, n2=test_n, x=num_rounds-1, y=0,
    #                     head=0, group_size=pairs, diff=(0xc0240100, 0x44202100, 0x0c200008, 0x0c200000))
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    # print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():
        net = load_model("chaskey_best_3r_pairs"+str(pairs)+"_diff_(2147483648, 0, 0, 2147483648)_distinguisher.h5")
        net_json = net.to_json()
        net_first = model_from_json(net_json)
        net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        net_first.load_weights("chaskey_best_3r_pairs"+str(pairs)+"_diff_(2147483648, 0, 0, 2147483648)_distinguisher.h5") 

    check = make_checkpoint(
        wdir+'first_best_'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    net_first.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check],verbose=0)
    net_first.save(wdir+"net_first.h5")

def second_stage(pairs):

    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(chk.create_train_test_dataset, [(int(10**7*pairs/process_number), int(
            10**6*pairs/process_number), num_rounds+1, 0, 0, pairs, (0x8400, 0x0400, 0, 0),) for i in range(process_number)])
    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    X_eval = accept_XY[0][2]
    Y_eval = accept_XY[0][3]
    for i in range(process_number-1):
        X = np.concatenate((X, accept_XY[i+1][0]))
        Y = np.concatenate((Y, accept_XY[i+1][1]))
        X_eval = np.concatenate((X, accept_XY[i+1][2]))
        Y_eval = np.concatenate((Y, accept_XY[i+1][3]))
    # print("multiple processing end ......")
    
    # X, Y, X_eval, Y_eval = chk.create_train_test_dataset(n1=n, n2=test_n, x=num_rounds, y=0,
    #                 head=0, group_size=pairs, diff=(0x8400, 0x0400, 0, 0))
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    # print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model(wdir+'first_best_'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
        net_json = net.to_json()

        net_second = model_from_json(net_json)
        net_second.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
        net_second.load_weights(wdir+'first_best_'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5') 
    
    check = make_checkpoint(
        'chaskey_best_'+str(num_rounds+1)+"r_pairs"+str(pairs)+'_distinguisher.h5')
    h = net_second.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check],verbose=0)
    print("Best validation accuracy: ", np.max(h.history['val_acc']))

    net_second.save("chaskey_"+str(num_rounds+1)+"_"+str(pairs)+"_distinguisher.h5")

for pairs in [2,4,8,16]:
    print("pairs = ",pairs)
    first_stage(pairs)
    second_stage(pairs)
