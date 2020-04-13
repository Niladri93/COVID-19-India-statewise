import numpy as np
import os
from os import listdir
from os.path import isfile, join
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

rowCount = 0
nonzero = 0
scaler = 0
train_X = 0

def getLastFileInPath(filePath):
    files = [f for f in listdir(filePath) if isfile(join(filePath, f))]
    return  files[-1]

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	print("-------------------------")
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
    # drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
    
	return agg

def dataPreprocessing(masterDict,states):
    ##Formulation of the dataset
    global rowCount
    global scaler
    global nonzero
    for state,data in masterDict.items():
        if state in states:
            rowCount = len(data['New Indian'])
            nonzero = np.count_nonzero(data['New Indian'], axis=0)
            dataset = pd.DataFrame(data=data)
#            dataset['CumulativeCount'] = data['New Indian']
#            dataset['Recovered'] = data['Statewise recovered increment']
#            dataset['Deceased'] = data['Statewise deceased increment']
#            dataset['Humidity'] = data['Humidity']
#            dataset['Temperature'] = data['Temperature']
    #
    #Cleaning and normalization of dataset
    # ensure all data is float
    dataset = dataset.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    dropList = []
    i=len(data.items())+1
    while  i < 2*len(data.items()):
        dropList.append(i)
        i +=1
    reframed.drop(reframed.columns[dropList], axis=1, inplace=True)
    print(reframed.head())
    return reframed


def trainAndTest(masterDict,states):
    
    global scaler
    global train_X
    training_loss = 0
    validation_loss = 0
    data = dataPreprocessing(masterDict,states)
    # split into train and test sets
    values = data.values
    n_train_hours = int(rowCount*(3/4))

    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
   
    # design network
    model = Sequential()
    model.add(LSTM(40, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    # fit network and save model based on least loss values
    early_stop = EarlyStopping(monitor='loss', min_delta=0.00000001, patience=3, mode='min', verbose=2)
    checkpoint_path = "training_2/cp-{epoch:04d}.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    
    history = model.fit(train_X, train_y, batch_size=n_train_hours, epochs=400, verbose=2, callbacks=[checkpoint], validation_data=(test_X, test_y), shuffle=False)
    
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    
    # make a prediction
    fileNames = (os.popen("ls ./"+ checkpoint_dir).read()).split()
    model.load_weights('./'+checkpoint_dir+'/'+fileNames[-1])
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    print("The predicted variable: ", inv_yhat.shape)
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    print("The predicted variable: ", inv_yhat.shape)
    print("The predicted Y^ variable: ", inv_yhat)
    print("Actual Y variable: ", inv_y)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

def futurePrediction(masterDict,argv):
    global train_X
    data = dataPreprocessing(masterDict,argv)
    values = data.values[nonzero-1:,:]
    
    # design network
    model = Sequential()
    model.add(LSTM(40, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # loading the best saved model
    checkpoint_path = "training_2/cp-{epoch:04d}.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # make a prediction
    fileNames = (os.popen("ls ./"+ checkpoint_dir).read()).split()
    model.load_weights('./'+checkpoint_dir+'/'+fileNames[-1])
    print ("---------------------------------------------------------------")
    print ("---------------------------------------------------------------")
    for i in range (0, (len(values) - 2)):
        
        test_X = values[i, :-1]
        test_X = np.transpose(test_X)
        test_X = test_X.reshape((1, 1, 3))
        #print ()
        yhat = model.predict(test_X)
        
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        
        values [i+1,0] = yhat
        print ("Prediction at day-{} = {}".format(i,inv_yhat))