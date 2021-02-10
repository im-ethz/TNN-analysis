import keras
from keras.layers import Input, Dense, Concatenate, Dropout, Lambda, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.metrics import RSquare

class NN: # time-delay neural network
	def __init__(self, n_input, n_output, n_hidden, act_hidden, do_hidden,
		optimizer='Adam', loss='mse'):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.act_hidden = act_hidden
		self.do_hidden = do_hidden
		self.n_output = n_output

		self.optimizer = optimizer
		self.loss = loss
		self.model = self.neural_network()

	def neural_network(self):
		inp = Input(shape = (self.n_input,))

		hid = Dense(self.n_hidden[0], activation = self.act_hidden[0])(inp)
		hid = Dropout(self.do_hidden[0])(hid)
		for h in range(1,len(self.n_hidden)):
			hid = Dense(self.n_hidden[h], activation = self.act_hidden[h])(hid)
			hid = Dropout(self.do_hidden[h])(hid)

		out = Dense(self.n_output, activation = 'linear')(hid)

		self.model = Model(inputs = inp, outputs = out)
		self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = RSquare(y_shape=(self.n_output,)))
		return self.model

class LSTM:
	def __init__(self, n_input, n_output, n_hidden, act_hidden, n_timesteps_p, n_features_p, n_timesteps_g, n_features_g, 
		optimizer='Adam', loss='mse'):
		self.n_timesteps_p = n_timesteps_p # window size
		self.n_features_p = n_features_p # features (so n_cols = n_timesteps * n_features)

		self.n_timesteps_g = n_timesteps_g # window size
		self.n_features_g = n_features_g # features (so n_cols = n_timesteps * n_features)
		
		self.n_output = n_output
		self.n_hidden = n_hidden
		self.act_hidden = act_hidden

		self.optimizer = optimizer
		self.loss = loss
		self.model = lstm()

	def lstm(self):
		inp_s = Input(shape = (self.n_input,))
		inp_p = Input(input_shape=(self.n_timesteps_p, self.n_features_p))
		inp_g = Input(input_shape=(self.n_timesteps_g, self.n_features_g))

		s = Dense(self.n_hidden[0], activation = self.act_hidden[0])(inp_s)
		s = Dropout(self.do_hidden[0])(s)
		for h in range(1,len(self.n_hidden)):
			s = Dense(self.n_hidden[h], activation = self.act_hidden[h])(s)
			s = Dropout(self.do_hidden[h])(s)
		s = Dense(10, activation = 'linear')(s)

		p = LSTM(10)(inp_p)
		#p = Dropout(self.do_hidden[0])(p)
		#p = Dense(100, activation='relu')(p)
		p = Dense(10, activation='linear')(p)

		g = LSTM(3)(inp_g)
		#g = Dropout(self.do_hidden[0])(g)
		#p = Dense(100, activation='relu')(g)
		g = Dense(3, activation='linear')(g)

		comb = Concatenate([s, p, g])
		comb = Dense(20, activation='relu')(comb)
		out = Dense(self.n_output, activation='linear')(comb)

		model = Model(inputs=[inp_s, inp_p, inp_g], outputs=out)
		model.compile(optimizer=self.optimizer, loss = self.loss, metrics = RSquare(y_shape=(self.n_output,)))
		return self.model