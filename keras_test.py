import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed, Masking
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
import reber_utility as ru

step_size = 50
n_step=2

batch_size = 10
n_batch = 15 # total number of batches
n_test_batch = 2 # testing part

train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))

N_Epoch = 80

m = ru.generate_reber_machine_discrete()

X,Y = m.to_X_and_Y(m.make_words(n_batch*batch_size,min_steps=(step_size*n_step+1)))

print( X.shape)
print( Y.shape)

model = Sequential()
model.add(Masking(mask_value= -1., batch_input_shape=(batch_size,step_size,len(m.transitions))))
model.add(LSTM(20, return_sequences=True,stateful=True, batch_input_shape=(batch_size,step_size,len(m.transitions))))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(m.transitions))))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

metrics_train = np.zeros((N_Epoch,len(model.metrics_names),n_batch-n_test_batch))
metrics_test = np.zeros((N_Epoch,len(model.metrics_names),n_test_batch))

print('Starting to learn:')
for i in range(N_Epoch):
    print('------- {} out of {} Epoch -----'.format(i+1,N_Epoch))

    ## Epochs should take all data; batches presented random, reset at each end of batch_size
    np.random.shuffle(train_idxes)
    batches_idxes = np.reshape(train_idxes, (-1,batch_size))
    for j, batch  in enumerate(batches_idxes):
        #print('batch {} of {}'.format(j+1,n_batch-n_test_batch))
        for k in range(n_step):
            metrics_train[i,:,j] += model.train_on_batch(X[batch,k*step_size:(k+1)*(step_size)], Y[batch,k*step_size:(k+1)*step_size]) #python 0:3 gives 0,1,2 (which is not intuitive at all)
        model.reset_states()

    test_batch_idxes = np.reshape(test_idxes,(-1,batch_size))
    for j, test_batch in enumerate(test_batch_idxes):
        for k in range(n_step):
                metrics_test[i,:,j] += model.test_on_batch(X[test_batch,k*step_size:(k+1)*(step_size)], Y[test_batch,k*step_size:(k+1)*step_size])
        model.reset_states()

    metrics_test[i] = metrics_test[i]/float(n_step) # divide only i indice, else division would be done for all at each epoch
    metrics_train[i] = metrics_train[i]/float(n_step)

    print('Train results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_train[i],axis=1)))
    print('Test results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_test[i],axis=1) ))

model.save('embedCerg_model_4.h5')
np.save('embedXdata_4.npy',X)
np.save('embedydata_4.npy',Y)
np.save('embedTrainMetrics_4', metrics_train)
np.save('embedTestMetrics_4', metrics_test)