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
varfile = './var.log'

def commandParsing():
	parser = ArgumentParser()
	parser.add_argument('-i', '--i_file', help='Model file', required=True)
	parser.add_argument('-C', '--c_file', help='One-hot constraint file', required=True)
	parser.add_argument('-c', '--check' , help='check only', action='store_true')
	parser.add_argument('-O', '--o_loop', type=int, default=100)
	parser.add_argument('-I', '--i_loop', type=int, default=1000)
	parser.add_argument('-S', '--tmp_st', type=float, default=100.0)
	parser.add_argument('-E', '--tmp_en', type=float, default=0.1)
	parser.add_argument('-s', '--seed',   type=int, default=12345)
	parser.add_argument('-d', '--debug',  help='record energy transition', action='store_true')
	parser.add_argument('-v', '--var',    help='record final state', action='store_true')
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
	def __init__(self, file_name):
		self.f_str = file_name
		self.N = 0
		self.J = None
		self.h = None
		self.C = 0
		self.generateModel()
		self.tbl = None # One-hot table
		##self.printModel()

	def generateModel(self):
		print('Model file:', self.f_str)
		self.readSpecifiedModel()

	def printModel(self):
		print('N =', self.N)
		print('-- J --')
		print(self.J)
		print('-- h --')
		print(self.h)
		print('-- C --')
		print(self.C)

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
			x = int(str_list[3*i+1])
			y = int(str_list[3*i+2])
			val = int(str_list[3*(i+1)])
			self.J[x][y] = val
			self.J[y][x] = val

		## magnetic field
		self.h = np.zeros((self.N,), dtype=np.int32)
		for i in range(self.N):
			self.h[i] = self.J[i][i]
			self.J[i][i] = 0

		## constant value
		if len(str_list) % 3 == 2:
			self.C = int(str_list[len(str_list)-1])

	def getModel(self):
		return self.N, self.J, self.h, self.C

	def getOneHotConstraintTable(self, file_name):
		if file_name is None:
			print('One-hot constraint is not specified')
			sys.exit(1)
		
		## READ DATA FILE ##
		try:
			with open(file_name) as f:
				__str = f.read()
				str_list = __str.split()
		except IOError:
			print('ERROR: CANNOT OPEN FILE')
			sys.exit(1)

		self.tbl = []
		tbl_size = int(str_list[0])
		for i in range(tbl_size):
			st = int(str_list[2*i+1])
			en = int(str_list[2*i+2])
			self.tbl.append((st, en))

		self.printOneHotTable()

		return self.tbl

	def printOneHotTable(self):
		print('One-hot table (size: {0:d})'.format(len(self.tbl)))
		print(self.tbl)


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
	def __init__(self, N, J, h, C, p, d, v, c, la, mu, tbl):
		self.N = N
		self.J = J
		self.h = h
		self.C = C
		self.o_loop = p[0]
		self.i_loop = p[1]
		self.tmp_st = p[2]
		self.tmp_en = p[3]
		self.seed = p[4]
		self.tmp_delta = pow((self.tmp_en/self.tmp_st), (1./(self.o_loop)))
		self.debug = d
		self.var = v
		self.check = c
		self.la = la
		self.mu = mu
		self.tbl = tbl
		self.state = np.zeros((N,), dtype=np.int32)
		if self.check: self.generateRandomState()
		else: self.generateCustomState()
		self.local_field = np.dot(self.state, self.J) + self.h
		self.init_H = 0
		self.fin_H = 0

	def generateRandomState(self):
		random.seed(self.seed)
		for i in range(self.N):
			self.state[i] = 0 if random.random() < 0.5 else 1

	def generateCustomState(self):
		tbl_size = len(self.tbl)
		for i in range(tbl_size):
			self.state[self.tbl[i][0]] = 1

	def printState(self): print(self.state)

	def calcH(self):
		energy = np.dot(np.dot(self.state, self.J), self.state) // 2 + np.dot(self.h, self.state) + self.C
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

	def searchOne(self, idx_0):
		group = -1
		for i in range(len(self.tbl)):
			st = self.tbl[i][0]
			en = self.tbl[i][1]
			if idx_0 >= st and idx_0 <= en:
				group = i
				break
		if group == -1:
			print('ONE-HOT SEARCH ERROR:', idx_0)
			sys.exit(1)
		idx_1 = -1
		for i in range(self.tbl[i][0], self.tbl[i][1] + 1):
			if self.state[i] == 1:
				idx_1 = i
				break
		if idx_1 == -1:
			print('1\'s STATE SEARCH ERROR')
			sys.exit(1)
		return idx_1

	def checkOneHotConstraint(self):
		check_num = 0
		for i in range(len(self.tbl)):
			s = 0
			for j in range(self.tbl[i][0], self.tbl[i][1] + 1):
				s += self.state[j]
			if s != 1:
				check_num += 1
		return check_num

	def outputFinState(self):
		log = open(varfile, mode='w')
		for i in range(self.N):
			log.write('{0} {1}\n'.format(i, self.state[i]))
		log.close()

	def run(self):

		self.init_H = self.calcH()

		beta_int, beta_frc = decimal_to_bin(1./self.tmp_st, beta_ibit, beta_fbit)
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
		print(' T(real) = {0} --> {1}'.format(self.tmp_st, 1./beta_tmp_fin))
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

				if not self.check and self.state[trgt_idx] == 1: continue

				comp_idx = trgt_idx if self.check else self.searchOne(trgt_idx)

				delta_E = 0 # delta-E
				if self.check:
					delta_E += self.local_field[trgt_idx] * (1 - self.state[trgt_idx] * 2)
				else:
					delta_E += self.local_field[trgt_idx] * (1 - self.state[trgt_idx] * 2)
					delta_E += self.local_field[comp_idx] * (1 - self.state[comp_idx] * 2)
					delta_E -= self.J[trgt_idx][comp_idx]
				delta_E_sign = -1 if delta_E < 0 else 1
				delta_E_data = abs(delta_E)

				delta_E_data_int, _ = decimal_to_bin(delta_E_data, 32, 0)
				x_int, x_frc = self.calc_x(delta_E_data_int, beta_int, beta_frc)
				y_frc = self.sigmoid_app(x_int, x_frc, delta_E_sign)

				y = binary_to_dec(y_frc, '')
				if y > val_L:
					if self.check:
						self.local_field += self.J[trgt_idx] * (1 - self.state[trgt_idx] * 2) # Update local field
						self.state[trgt_idx] = 1 - self.state[trgt_idx] # Update spin
					else:
						self.local_field += self.J[trgt_idx] * (1 - self.state[trgt_idx] * 2) # Update local field
						self.local_field += self.J[comp_idx] * (1 - self.state[comp_idx] * 2) # Update local field
						self.state[trgt_idx] = 1 - self.state[trgt_idx] # Update spin
						self.state[comp_idx] = 1 - self.state[comp_idx] # Update spin

			if self.debug:
				energy = np.dot(self.local_field + self.h, self.state) // 2 + self.C
				log.write('Loop {0:4d}: {1}\n'.format(i, energy))

		if self.debug: log.close()

		self.fin_H = self.calcH()

		print('----------------------------------------')
		print('-- Result ------------------------------')
		print('----------------------------------------')
		print(' H(init) =', self.init_H)
		print(' H(fin)  =', self.fin_H)
		if self.var:
			print(' Output fin state to', varfile)
			self.outputFinState()
		print(' #. One-hot constraint violations =', self.checkOneHotConstraint())
		print('----------------------------------------')


def main():
	## Parse the command line
	args = commandParsing()

	## Set model & params
	model = Model(args.i_file)
	tbl = model.getOneHotConstraintTable(args.c_file)
	N, J, h, C = model.getModel()
	params = (args.o_loop, args.i_loop, args.tmp_st, args.tmp_en, args.seed)

	lut = PiecewiseLinearAppLUT()
	la, mu = lut.getTable()

	anneal = Annealing(N, J, h, C, params, args.debug, args.var, args.check, la, mu, tbl)
	anneal.run()

if __name__ == '__main__':
	main()
