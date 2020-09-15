import sys

from argparse import ArgumentParser
import math
import numpy as np
import random
from tqdm import tqdm

beta_ibit = 5
beta_fbit = 27

random.seed(123) # Seed for generating a random model (not used when using a model file)

logfile = './energy.log'

def commandParsing():
	parser = ArgumentParser()
	parser.add_argument('-i', '--i_file', help='Model file (A random model is generated when nothing is specified.)')
	parser.add_argument('-n', '--num', help='#. of spins included in a random model', type=int, default=256)
	parser.add_argument('-O', '--o_loop', type=int, default=600)
	parser.add_argument('-I', '--i_loop', type=int, default=4000)
	parser.add_argument('-S', '--tmp_st', type=float, default=20.0)
	parser.add_argument('-E', '--tmp_en', type=float, default=0.5)
	parser.add_argument('-s', '--seed', type=int, default=123)
	parser.add_argument('-d', '--debug', help='show process', action='store_true')
	args = parser.parse_args()
	return args


## Transform to binary number
## 0 <= data < 2 ** ibit
def decimal_to_bin(data, ibit, fbit):
	if data >= (2 ** ibit):
		print('(Conversion error) bit overflow: {0} ({1}bit)'.format(data, ibit))
		sys.exit(1)
	if data < 0:
		print('(Conversion error) negative value: {0}'.format(data))
		sys.exit(1)

	data_int = int(math.floor(data))
	data_frc = data - math.floor(data)
	# frc part
	str_frc = ''
	for i in range(fbit):
		data_frc *= 2
		str_frc += str(int(math.floor(data_frc)))
		data_frc -= math.floor(data_frc)
	# int part
	str_int = ''
	if data_int == 0: str_int = '0' * ibit
	else: str_int = bin(data_int)[2:].zfill(ibit)

	return str_int, str_frc

## Transform to decimal number 
def binary_to_dec(str_int, str_frc):
	str_data = str_int + str_frc
	value = int(str_data, 2) / float(2 ** len(str_frc))

	return value

## binary adder
def binary_add(int1, frc1, int2, frc2):
	if len(frc1) != len(frc2):
		print('(Addition error) fraction part length unmatch')
		sys.exit(1)
	ibit = max(len(int1), len(int2)) + 1
	fbit = len(frc1)
	str_data1 = int1 + frc1
	str_data2 = int2 + frc2
	data1 = int(str_data1, 2)
	data2 = int(str_data2, 2)
	result = data1 + data2
	str_result = bin(result)[2:].zfill(ibit+fbit)
	i_0 = 0
	i_1 = ibit
	i_2 = ibit + fbit
	str_result_int = str_result[i_0:i_1]
	str_result_frc = str_result[i_1:i_2]

	return str_result_int, str_result_frc

## binary multiplier
def binary_mul(int1, frc1, int2, frc2, ibit, fbit):
	ibit_sum = len(int1) + len(int2)
	fbit_sum = len(frc1) + len(frc2)
	if ibit > ibit_sum:
		print('(Multiplication error) ibit {0} is larger than {1}'.format(ibit, ibit_sum))
		sys.exit(1)
	if fbit > fbit_sum:
		print('(Multiplication error) fbit {0} is larger than {1}'.format(fbit, fbit_sum))
		sys.exit(1)
	str_data1 = int1 + frc1
	str_data2 = int2 + frc2
	data1 = int(str_data1, 2)
	data2 = int(str_data2, 2)
	result = data1 * data2
	if result >> (ibit + fbit_sum) > 0:
		print('(Multiplication error) bit overflow: {0} ({1}bit)'.format(result >> (fbit_sum), ibit))
		sys.exit(1)
	str_result = bin(result)[2:].zfill(ibit_sum+fbit_sum)
	i_0 = ibit_sum - ibit
	i_1 = ibit_sum
	i_2 = ibit_sum + fbit
	str_result_int = str_result[i_0:i_1]
	str_result_frc = str_result[i_1:i_2]

	return str_result_int, str_result_frc


class Model():

	## CONSTRUCTOR ##
	def __init__(self, file_name, n):
		self.f_str = file_name
		self.N = n
		self.J = None
		self.h = None
		self.generateModel()
		##self.printModel()

	def generateModel(self):
		if self.f_str is not None:
			print('Model file:', self.f_str)
			self.readSpecifiedModel()
		else:
			print('Random model is generated')
			self.generateRandomModel()

	def printModel(self):
		print('N =', self.N)
		print('-- J --')
		print(self.J)
		print('-- h --')
		print(self.h)

	def readSpecifiedModel(self):

		## READ DATA FILE ##
		try:
			with open(self.f_str) as f:
				__str = f.read()
				str_list = __str.split()
		except IOError:
			print('ERROR: CANNOT OPEN FILE')
			sys.exit(1)

		## #spins
		self.N = int(str_list[0])

		## spin-spin interactions
		self.J = np.zeros((self.N, self.N), dtype=np.int32)
		for i in range(int(len(str_list)/3)):
			x = int(str_list[3*i+1]) - 1
			y = int(str_list[3*i+2]) - 1
			val = int(str_list[3*(i+1)])
			self.J[x][y] = val
			self.J[y][x] = val

		## magnetic field
		self.h = np.zeros((self.N,), dtype=np.int32)
		for i in range(self.N):
			self.h[i] = self.J[i][i]
			self.J[i][i] = 0

	def generateRandomModel(self):

		## spin-spin interactions
		self.J = np.zeros((self.N, self.N), dtype=np.int32)

		## magnetic field
		self.h = np.zeros((self.N,), dtype=np.int32)

		for j in range(self.N):
			self.h[j] = random.randint(-999, 999) # [-999,999]
			for i in range(self.N):
				if j >= i: continue
				val = random.randint(-99, 99) # [-99, 99]
				self.J[j][i] = val
				self.J[i][j] = val

	def getModel(self):
		return self.N, self.J, self.h


class PiecewiseLinearAppLUT():

	## CONSTRUCTOR ##
	def __init__(self):
		self.f_n_str_arr = []
		self.f_o_str_arr = []
		self.generateTable()
		##self.printTable()

	def sigmoid_org(self, x):
		return 1. / (1 + np.exp(-x))

	def generateTable(self):
		n = np.arange(8, dtype=float) # n = 0 ~ 7
		f_n0 = self.sigmoid_org(n)
		f_n1 = self.sigmoid_org(n+1)
		for i in range(8):
			_, f_n_str_frc = decimal_to_bin(f_n0[i], 0, 16) # lambda
			_, f_o_str_frc = decimal_to_bin(f_n1[i] - f_n0[i], 0, 16) # mu
			self.f_n_str_arr.append(f_n_str_frc)
			self.f_o_str_arr.append(f_o_str_frc)

	def printTable(self):

		## lambda(n) & mu(n)
		print('----------------------------------------')
		print('n    lambda             mu')
		print('----------------------------------------')
		for i in range(8):
			print('{0:d}: 0.{1} 0.{2}'.format(i, self.f_n_str_arr[i], self.f_o_str_arr[i]))
		print('----------------------------------------')

	def getTable(self):
		return self.f_n_str_arr, self.f_o_str_arr


class Annealing():

	## CONSTRUCTOR ##
	def __init__(self, N, J, h, p, d, la, mu):
		self.N = N
		self.J = J
		self.h = h
		self.o_loop = p[0]
		self.i_loop = p[1]
		self.tmp_st = p[2]
		self.tmp_en = p[3]
		self.seed = p[4]
		self.tmp_delta = pow((self.tmp_en/self.tmp_st), (1./(self.o_loop)))
		self.debug = d
		self.la = la
		self.mu = mu
		self.state = np.ones((N,), dtype=np.int32)
		self.generateRandomState()
		self.local_field = np.dot(self.state, self.J) + self.h
		self.init_H = 0
		self.fin_H = 0

	def generateRandomState(self):
		random.seed(self.seed)
		for i in range(self.N):
			self.state[i] = -1 if random.random() < 0.5 else 1

	def printState(self): print(self.state)

	def calcH(self):
		energy = -1 * (np.dot(np.dot(self.state, self.J), self.state) // 2 + np.dot(self.h, self.state))
		return energy

	## Random number generator
	def XORshift32(self, rand_val):
		rand_val = rand_val ^ (rand_val << 13 & 0xFFFFFFFF)
		rand_val = rand_val ^ (rand_val >> 17 & 0xFFFFFFFF)
		rand_val = rand_val ^ (rand_val << 15 & 0xFFFFFFFF)
		return rand_val & 0xFFFFFFFF

	## The sigmoid function
	## input: 4bit + 16bit
	## output: 16bit (0.[...y_frc...])
	def sigmoid_app(self, x_int, x_frc, x_sign):
		y_frc = '1111111111111111'
		if x_int[0] == '0':
			trgt_n = int(x_int[1:], 2)
			_, z_frc = binary_mul('', self.mu[trgt_n], '', x_frc, 0, 16)
			_, y_frc = binary_add('', self.la[trgt_n], '', z_frc)
		if x_sign > 0:
			y_frc, _ = decimal_to_bin(2**16-1-binary_to_dec(y_frc, ''), 16, 0)
		return y_frc

	## Calculate x (input of the sigmoid function)
	## from |delta_E / 2| (32bit + 0bit) and beta (5bit + 27bit)
	## output: (1+3)bit + 16bit
	def calc_x(self, int1, int2, frc2):
		str_data1 = int1
		str_data2 = int2 + frc2
		data1 = int(str_data1, 2)
		data2 = int(str_data2, 2)
		result = data1 * data2
		str_result = bin(result)[2:].zfill(64)
		msb = '0'
		if result >> 30 > 0:
			msb = '1'
		str_result_int = msb + str_result[34:37]
		str_result_frc = str_result[37:53]
		return str_result_int, str_result_frc

	def run(self):

		self.init_H = self.calcH()

		beta_int, beta_frc = decimal_to_bin(2./self.tmp_st, beta_ibit, beta_fbit)
		beta_delta_int, beta_delta_frc = decimal_to_bin(1./self.tmp_delta, beta_ibit, beta_fbit)
		## check
		beta_tmp_int, beta_tmp_frc = beta_int, beta_frc
		for i in range(self.o_loop):
			beta_tmp_int, beta_tmp_frc = binary_mul(beta_tmp_int, beta_tmp_frc, beta_delta_int, beta_delta_frc, beta_ibit, beta_fbit)
		beta_tmp_fin = binary_to_dec(beta_tmp_int, beta_tmp_frc)

		print('----------------------------------------')
		print('-- Model & Parameters ------------------')
		print('----------------------------------------')
		print(' N =', self.N)
		print(' #Loops  = {0:d} x {1:d}'.format(self.o_loop, self.i_loop))
		print(' T(set)  = {0} --> {1}'.format(self.tmp_st, self.tmp_en))
		print(' T(real) = {0} --> {1}'.format(self.tmp_st, 2./beta_tmp_fin))
		print(' seed =', self.seed)
		print('----------------------------------------')

		rand_value = self.seed # seed (0 is invalid)

		if self.debug:
			log = open(logfile, mode='w')
			log.write('-- Energy transition --\n')

		for i in tqdm(range(self.o_loop)):

			# Update beta
			beta_int, beta_frc = binary_mul(beta_int, beta_frc, beta_delta_int, beta_delta_frc, beta_ibit, beta_fbit)

			for j in range(self.i_loop):

				rand_value = self.XORshift32(rand_value) # Obtain 32bit unsigned random number
				val_H = rand_value >> 16 & 0xFFFF
				val_L = rand_value & 0xFFFF

				trgt_idx = val_H % self.N

				delta_E = self.local_field[trgt_idx] * self.state[trgt_idx] # delta-E / 2
				delta_E_sign = -1 if delta_E < 0 else 1
				delta_E_data = abs(delta_E)

				delta_E_data_int, _ = decimal_to_bin(delta_E_data, 32, 0)
				x_int, x_frc = self.calc_x(delta_E_data_int, beta_int, beta_frc)
				y_frc = self.sigmoid_app(x_int, x_frc, delta_E_sign)

				y = binary_to_dec(y_frc, '')
				if y > val_L:
					self.state[trgt_idx] *= -1 # Update spin
					self.local_field += 2 * self.J[trgt_idx] * self.state[trgt_idx] # Update local field

			if self.debug:
				energy = -1 * np.dot(self.local_field + self.h, self.state) // 2
				log.write('Loop {0:4d}: {1}\n'.format(i, energy))

		self.fin_H = self.calcH()

		print('----------------------------------------')
		print('-- Result ------------------------------')
		print('----------------------------------------')
		print(' H(init) =', self.init_H)
		print(' H(fin)  =', self.fin_H)
		print('----------------------------------------')


def main():
	## Parse the command line
	args = commandParsing()

	## Set model & params
	model = Model(args.i_file, args.num)
	N, J, h = model.getModel()
	params = (args.o_loop, args.i_loop, args.tmp_st, args.tmp_en, args.seed)

	lut = PiecewiseLinearAppLUT()
	la, mu = lut.getTable()

	anneal = Annealing(N, J, h, params, args.debug, la, mu)
	anneal.run()

if __name__ == '__main__':
	main()
