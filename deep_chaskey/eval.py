
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model
import multiprocessing as mp
import chaskey as chk

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable = True) 
import sys 
sys.path.append("..") 
# wdir = './DATA_N_good_trained_nets/'
wdir = './DATA_Nm_good_trained_nets/'
pairs = 16

net3 = load_model(wdir+'chaskey_best_'+str(3)+'r'+"_pairs"+str(pairs)+'_distinguisher.h5')
net4 = load_model(wdir+'chaskey_best_'+str(4)+'r'+"_pairs"+str(pairs)+'_distinguisher.h5')
# net5 = load_model(wdir+'chaskey_best_'+str(5)+'r'+"_pairs"+str(pairs)+'_distinguisher.h5')

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
    accept_XY3 = pool.starmap(chk.create_train_test_dataset, [(int(10**6*pairs/process_number), int(
        10**6*pairs/process_number), 3, 0, 0, pairs, (
                0x8400, 0x0400, 0, 0),) for i in range(process_number)])
    # accept_XY3 = pool.starmap(chk.create_train_test_dataset, [(int(10**6/process_number), int(
    #     10**6/process_number), 3, 0, 0, pairs, (
    #             0x8400, 0x0400, 0, 0),) for i in range(process_number)])
X3 = accept_XY3[0][0]
Y3 = accept_XY3[0][1]

for i in range(process_number-1):
    X3 = np.concatenate((X3, accept_XY3[i+1][0]))
    Y3 = np.concatenate((Y3, accept_XY3[i+1][1]))
    
with mp.Pool(process_number) as pool:
    accept_XY4 = pool.starmap(chk.create_train_test_dataset, [(int(10**6*pairs/process_number), int(
        10**6*pairs/process_number), 4, 0, 0, pairs,(
                0x8400, 0x0400, 0, 0),) for i in range(process_number)])
    # accept_XY4 = pool.starmap(chk.create_train_test_dataset, [(int(10**6/process_number), int(
    #     10**6/process_number), 4, 0, 0, pairs, (
    #             0x8400, 0x0400, 0, 0),) for i in range(process_number)])
X4 = accept_XY4[0][0]
Y4 = accept_XY4[0][1]

for i in range(process_number-1):
    X4 = np.concatenate((X4, accept_XY4[i+1][0]))
    Y4 = np.concatenate((Y4, accept_XY4[i+1][1]))

    
# with mp.Pool(process_number) as pool:
#     # accept_XY5 = pool.starmap(chk.create_train_test_dataset, [(int(10**6*pairs/process_number), int(
#     #     10**6*pairs/process_number), 5, 0, 0, pairs, diff,) for i in range(process_number)])
#     accept_XY5 = pool.starmap(chk.create_train_test_dataset, [(int(10**6/process_number), int(
#         10**6/process_number), 5, 0, 0, pairs, (
#                 0x8400, 0x0400, 0, 0),) for i in range(process_number)])
# X5 = accept_XY5[0][0]
# Y5 = accept_XY5[0][1]

# for i in range(process_number-1):
#     X5 = np.concatenate((X5, accept_XY5[i+1][0]))
#     Y5 = np.concatenate((Y5, accept_XY5[i+1][1]))


print('Testing neural distinguishers against 3 to 5 blocks in the ordinary real vs random setting');

print('3 rounds:');
evaluate(net3, X3, Y3);
print('4 rounds:');
evaluate(net4, X4, Y4);
# print('5 rounds:');
# evaluate(net4, X5, Y5);


