import numpy as np
import pypianoroll as pp
import tensorflow as tf
import keras
from Generator_compressed import Generator_compressed

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

from hyperas.distributions import choice, uniform

def data():
    sl = 100
    b = 16
    subset=1.0
    print("load training")
    training_generator = Generator_compressed('../lpd',file_name="train_names.txt",sequence_length=sl,batch_size=b, subset=subset)
    print("load validation")
    validation_generator = Generator_compressed('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=b, subset=1.0)
    
    return training_generator, validation_generator

def model(training_generator,validation_generator):
    sl = 100
    model = keras.Sequential()
    model.add(keras.layers.CuDNNLSTM({{choice([64,128,256])}},input_shape=(sl,128),return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    if {{choice(['two','three'])}} == 'three':
      model.add(keras.layers.CuDNNLSTM({{choice([128,256,512])}},return_sequences=True))
      model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.CuDNNLSTM({{choice([128,256])}}))
    model.add(keras.layers.Dense({{choice([128,256])}},activation={{choice(['elu','relu'])}}))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense({{choice([128,256])}},activation={{choice(['elu','relu'])}}))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=1,
                        workers=10)
    
    validation_acc = np.amax(history.history['val_acc']) 

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

trials = Trials()
best_run, best_model = optim.minimize(model=model,data=data,
                                        algo=tpe.suggest,
                                        max_evals=50,
                                        trials=trials)

print("Evalutation of best performing model:")
#print(best_model.evaluate(X_test, Y_test))
print(best_run)
best_trials = sorted(trials.results, key=lambda x: x['loss'], reverse=False)
print("best trials")
print(best_trials)
best_model.save("hyper.h5")
keras.backend.clear_session()
