
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model
import multiprocessing as mp
import present as ps

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable = True) 
import sys 
sys.path.append("..") 
# wdir = './DATA_Nm_good_trained_nets/'
wdir = './DATA_N_good_trained_nets/'
pairs = 16

net6 = load_model(wdir+'present_best_'+str(6)+'r'+"_pairs"+str(pairs)+'_distinguisher.h5')
net7 = load_model(wdir+'present_best_'+str(7)+'r'+"_pairs"+str(pairs)+'_distinguisher.h5')

def evaluate(net,X,Y):
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = 1000 * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X,batch_size=batch_size).flatten();
    Zbin = (Z >= 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);


process_number = 50
with mp.Pool(process_number) as pool:
    # accept_XY6 = pool.starmap(ps.make_dataset_with_group_size, [(int(10**6*pairs/process_number),6,0x9,pairs,) for i in range(process_number)])
    accept_XY6 = pool.starmap(ps.make_dataset_with_group_size, [(int(10**6/process_number),6,0x9,pairs,) for i in range(process_number)])

X6 = accept_XY6[0][0]
Y6 = accept_XY6[0][1]

for i in range(process_number-1):
    X6 = np.concatenate((X6,accept_XY6[i+1][0]))
    Y6 = np.concatenate((Y6,accept_XY6[i+1][1]))

with mp.Pool(process_number) as pool:
    # accept_XY7 = pool.starmap(ps.make_dataset_with_group_size, [(int(10**6*pairs/process_number),7,0x9,pairs,) for i in range(process_number)])
    accept_XY7 = pool.starmap(ps.make_dataset_with_group_size, [(int(10**6/process_number),7,0x9,pairs,) for i in range(process_number)])

X7 = accept_XY7[0][0]
Y7 = accept_XY7[0][1]

for i in range(process_number-1):
    X7 = np.concatenate((X7,accept_XY7[i+1][0]))
    Y7 = np.concatenate((Y7,accept_XY7[i+1][1]))


print('Testing neural distinguishers against 6 to 7 blocks in the ordinary real vs random setting');


print('6 rounds:');
evaluate(net6, X6, Y6);
print('7 rounds:');
evaluate(net7, X7, Y7);


