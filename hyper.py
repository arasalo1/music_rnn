import numpy as np
import pypianoroll as pp
import tensorflow as tf
import keras
from Generator_compressed import Generator_compressed

from hyperopt import Trials, STATUS_OK, tpe
import hyperas
from hyperas.distributions import choice, uniform

def data():
    sl = 100
    b = 16
    subset=0.1
    print("load training")
    training_generator = Generator_compressed('../lpd',file_name="train_names.txt",sequence_length=sl,batch_size=b, subset=subset)
    print("load validation")
    validation_generator = Generator_compressed('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=b, subset=subset)
    
    return training_generator, validation_generator

def model(datagen):

    model = keras.Sequential()
    model.add(keras.layers.CuDNNLSTM(128,input_shape=(sl,128),return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.CuDNNLSTM(512,return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.CuDNNLSTM(256))
    model.add(keras.layers.Dense(256,activation='elu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128,activation='elu'))
    model.add(keras.layers.Dense(128,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    history = model.fit_generator(generator=datagen[0],
                        validation_data=datagen[1],
                        use_multiprocessing=True,
                        epochs=4,
                        workers=10)
    
    validation_acc = np.amax(result.history['val_acc']) 

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=model,data=data,
                                        algo=tpe.suggest,
                                        max_evals=5,
                                        trials=Trials())

print("Evalutation of best performing model:")
#print(best_model.evaluate(X_test, Y_test))
print(best_run)

keras.backend.clear_session()
