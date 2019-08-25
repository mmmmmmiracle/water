#++++++++++++++++++++++++++++++++++
# @author : yongsheng hou
# @date   : 2019.07.07
# @Email  : houys@tju.edu.cn
#++++++++++++++++++++++++++++++++++

import util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error

from robo.fmin import bayesian_optimization
from robo.fmin import random_search
from robo.fmin import entropy_search
import json
import time
import  os

class Optimizer:
	def __init__(self,src_file_index,model_index,is_ensemble=False,with_rain=False,method='bayesian_optimization'):
		self.set_bounds(lower_bounds=[36,2],upper_bounds=[60,30])
		self.model,self.model_name = util.get_model(is_ensemble=is_ensemble,model_index=model_index)
		self.src = util.get_src_file(src_file_index=src_file_index)
		self.method = method
		self.num_iterations=200
		self.with_rain=with_rain
		self.results = None

	def set_bounds(self,lower_bounds,upper_bounds):
		self.lower_bounds=np.array(lower_bounds,dtype=np.int16)
		self.upper_bounds=np.array(upper_bounds,dtype = np.int16)


	def objective_function(self,x):
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

	def optimization(self):
		if(self.method == 'bayesian_optimization'):
			results = bayesian_optimization(self.objective_function, lower=self.lower_bounds,
											upper=self.upper_bounds, num_iterations=self.num_iterations,
											acquisition_func='ei')
			print(results['x_opt'], results['f_opt'])
			self.results = {'x_opt': results['x_opt'], 'f_opt': results['f_opt'], 'mse': results['incumbent_values']}
		elif(self.method == 'random_search'):
			results = random_search(self.objective_function, lower=self.lower_bounds,
									upper=self.upper_bounds, num_iterations=self.num_iterations)
			print(results['x_opt'], results['f_opt'])
			self.results = {'x_opt': results['x_opt'], 'f_opt': results['f_opt'], 'mse': results['incumbent_values']}
		else:
			pass

def one_optimize(src_file_index,model_index,is_ensemble,with_rain):
	optimizer1 = None
	optimizat2 = None
	optimizer1 = Optimizer(src_file_index,model_index,is_ensemble,with_rain)
	optimizer2 = Optimizer(src_file_index,model_index,is_ensemble,with_rain, method='random_search')

	des_file_name = optimizer1.src.split('.')[0].split('/')[-1]
	path = 'optimization_result/' +'with_rain_' + str(with_rain) + '/'+ des_file_name
	if not os.path.exists(path):
		os.makedirs(path)

	t_start = time.time()
	optimizer1.optimization()
	t_end = time.time()
	t1 = round(t_end - t_start)
	t_start = time.time()
	optimizer2.optimization()
	t_end = time.time()
	t2 = round(t_end - t_start)
	res1 = optimizer1.results
	res2 = optimizer2.results

	res = {}
	res1['time_consume'] = str(t1) + 's'
	res2['time_consume'] = str(t2) + 's'
	res['bayesian_optimization'] = res1
	res['random_search'] = res2
	json_str = json.dumps(res, indent=4)  # 注意这个indent参数
	with open(path+'/'+ optimizer1.model_name +'.json', 'w') as json_file:
		json_file.write(json_str)

	plt.rcParams['figure.figsize'] = (8.0, 5.0)
	plt.plot(res1['mse'])
	plt.plot(res2['mse'])
	plt.title('performance:' + des_file_name)
	plt.legend(('method:bayesian_optimization\ncost:' + str(t1) + 's\nmin mse:' + str(res1['f_opt']) + '\nOP:' + str(
	np.array(res1['x_opt'], dtype=np.int16)),
			'method:random_search\ncost:' + str(t2) + 's\nmin mse:' + str(res2['f_opt']) + '\nOP:' + str(
				np.array(res2['x_opt'], dtype=np.int16))))
	plt.savefig(path + '/' + optimizer1.model_name + '.png', dpi=60)
	#plt.show()
	plt.close()


if __name__ == '__main__':
	linear_models_len = util.get_model_list_length(is_ensemble=False)
	ensemble_len = util.get_model_list_length(is_ensemble=True)
	src_file_len = util.get_src_file_list_length()

	for src_file_index in range(1,src_file_len):

		for with_rain in [True,False]:
			for linear_model_index in range(linear_models_len):
			# for linear_model_index in range(2,3):
				one_optimize(src_file_index, linear_model_index, is_ensemble=False, with_rain=with_rain)

			for ensemble_index in range(ensemble_len):
				one_optimize(src_file_index, ensemble_index, is_ensemble=True, with_rain=with_rain)
