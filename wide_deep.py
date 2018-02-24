#coding=utf-8
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Input,Dense, Merge,concatenate
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load(file):
    data_array = np.loadtxt(file)
    return data_array

def oneHotEncoder(array_1d):
    label = LabelEncoder().fit_transform(array_1d)
    label = label.reshape(len(label), 1)
    one_hot = OneHotEncoder(sparse=False).fit_transform(label)
    return one_hot

def minMaxScale(array_2d):
    return MinMaxScaler().fit_transform(array_2d)

def preprocess(array_2d):
   y = LabelEncoder().fit_transform(array_2d[:, 0])
   feature = array_2d[:,1:]
   (row_num, col_num) = array_2d.shape
   x = np.zeros((row_num, 1))
   for i in range(1, 13):
     x = np.c_[x, oneHotEncoder(array_2d[:, i])]
   x = np.c_[x, minMaxScale(array_2d[:, 13:392])]
   x = np.c_[x, oneHotEncoder(array_2d[:, 392])]
   x=np.delete(x, 0, axis=1) 
   return x, y

def main():
   print "---loading and preprocessing the data---"
   train_data = load('./data.train')
   test_data = load('./data.test')
   data = np.r_[train_data, test_data]
   train_len = train_data.shape[0]
   x, y = preprocess(data)
   X_train = x[0:train_len, :]
   y_train = y[0:train_len]
   X_test = x[train_len:, :]
   y_test = y[train_len:]

   print "---build the wide&deep model---"
   
   # wide
   #wide = Sequential()
   wide = Input(shape=(X_train.shape[1],))
   
   # deep
   deep_data = Input(shape=(X_train.shape[1],))
   deep = Dense(input_dim=X_train.shape[1], output_dim=256, activation='relu')(deep_data)
   deep = Dense(128, activation='relu')(deep)
   
   # wide & deep 
   wide_deep = concatenate([wide, deep])
   wide_deep = Dense(1, activation='sigmoid')(wide_deep)
   model = Model(inputs=[wide, deep_data], outputs=wide_deep)
    
   print "---starting the training---"
   model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
   )
    
   model.fit([X_train, X_train], y_train, nb_epoch=10, batch_size=32)
    
   loss, accuracy = model.evaluate([X_test, X_test], y_test)
   print('\n', 'test accuracy:', accuracy)
   y_pred = model.predict([X_test,X_test])
   print "auc is ", roc_auc_score(y_test, y_pred)


if __name__ == '__main__':
   main()

