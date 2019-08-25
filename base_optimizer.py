import util
import numpy as np
from sklearn.metrics import mean_squared_error

from robo.fmin import bayesian_optimization
from robo.fmin import random_search

import json
import time

class BaseOptimizer(object):
	def __init__(self,src_file_index,model_index,bounds):
		self.src = util.get_src_file(src_file_index=src_file_index)
		self.model,self.model_name = util.get_model(is_ensemble=True,model_index=model_index)
		self.lower_bounds = bounds["lower_bounds"]
		self.upper_bounds = bounds["upper_bounds"]
		self.with_rain = False
		self.optimization_methods = ["random_search",'bayesian_optimization']
		self.num_iterations = 200
		self.results = {}
		self.result_save_path = 'optimization_result/with_rain_'+str(self.with_rain)+'/'+self.src.split('.')[0].split('/')[-1]+'/'
		self.optimization()
		self.save_optimization_result()

	def objective_function(self,x):
		self.tune_params = ['offset','period']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

	def optimization(self):
		for optimization_method in self.optimization_methods:
			t_start = time.time()
			if(optimization_method == 'bayesian_optimization'):
				results = bayesian_optimization(self.objective_function, lower=self.lower_bounds,
												upper=self.upper_bounds, num_iterations=self.num_iterations,
												acquisition_func='ei')
				t_end= time.time()
				t = round(t_end - t_start)
				x_opt = {}
				for i in range(len(self.tune_params)):
					x_opt[self.tune_params[i]] = results['x_opt'][i]
				print(x_opt, results['f_opt'])
				self.results['bayesian_optimization'] = {'method':'bayesian_optimization','x_opt': x_opt, 'f_opt': results['f_opt'], 'mse': results['incumbent_values'],'time_consume':t}
			elif(optimization_method == 'random_search'):
				results = random_search(self.objective_function, lower=self.lower_bounds,
										upper=self.upper_bounds, num_iterations=self.num_iterations)
				t_end= time.time()
				t = round(t_end - t_start)
				x_opt = {}
				print(results['x_opt'])
				for i in range(len(self.tune_params)):
					x_opt[self.tune_params[i]] = results['x_opt'][i]
				print(x_opt, results['f_opt'])
				self.results['random_search'] = {'method':'random_search','x_opt': x_opt, 'f_opt': results['f_opt'], 'mse': results['incumbent_values'],'time_consume':t}
			else:
				pass

	def save_optimization_result(self):
		json_str = json.dumps(self.results, indent=4)  # 注意这个indent参数
		#print(json_str)
		with open(self.result_save_path + self.model_name +'.json', 'w') as json_file:
			json_file.write(json_str)

class LinearRegressionOptimizer(BaseOptimizer):
	pass

class RidgeOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("Ridge最优化中...")
		self.tune_params = ['offset','period','alpha','max_iter','tol']
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.alpha = x[2]
		self.model.max_iter = x[3]
		self.model.tol = x[4]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class BayesianRidgeOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("BayesianRidge最优化中...")
		self.tune_params = ['offset','period','n_iter','tol','alpha_1','alpha_2','lambda_1','lambda_2']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.n_iter = int(x[2])
		self.model.tol = x[3]
		self.model.alpha_1 = x[4]
		self.model.alpha_2 = x[5]
		self.model.lambda_1 = x[6]
		self.model.lambda_2 = x[7]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class RidgeCVOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("RidgeCV最优化中...")
		self.tune_params = ['offset','period','alpha0','alpha1','alpha2','alpha3','cv']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.alphas=[x[2],x[3],x[4],x[5]]
		self.cv = int(x[6])
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class LassoOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("Lasso最优化中...")
		self.tune_params = ['offset','period','alpha','max_iter','tol']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.alpha = x[2]
		self.model.max_iter = x[3]
		self.model.tol = x[4]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class LassoLarsOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("LassoLars最优化中...")
		self.tune_params = ['offset','period','alpha','max_iter','eps']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.alpha = x[2]
		self.model.max_iter = x[3]
		self.model.eps = x[4]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class LassoLarsCVOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("LassoLarsCV最优化中...")
		self.tune_params = ['offset','period','max_iter','max_n_alphas','eps']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.max_iter = x[2]
		self.model.max_n_alphas = x[3]
		self.model.eps = x[4]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse
class AdaBoostRegressorOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("AdaBoostRegressor优化中...")
		self.tune_params = ['offset','period','n_estimators','learning_rate']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.n_estimators = int(x[2])
		self.model.learning_rate = x[3]
		self.model.loss = 'square'
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class BaggingRegressorOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("BaggingRegressor优化中...")
		self.tune_params = ['offset','period','n_estimators','max_samples','max_features']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.n_estimators = int(x[2])
		self.model.max_samples = x[3]
		self.model.max_features = x[4]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class GradientBoostingRegressorOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("GradientBoostingRegressor优化中...")
		self.tune_params = ['offset','period','n_estimators','learning_rate','subsample','min_samples_split','min_samples_leaf',
			'min_weight_fraction_leaf','max_depth','alpha']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.n_estimators = int(x[2])
		self.model.learning_rate = x[3]
		self.model.subsample = x[4]
		self.model.min_samples_split = int(x[5])
		self.model.min_samples_leaf = int(x[6])
		self.model.min_weight_fraction_leaf = x[7]
		self.model.max_depth = int(x[8])
		self.model.alpha = x[9]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class RandomForestRegressorOptimizer(BaseOptimizer):
	def objective_function(self,x):
		print("RandomForestRegressor优化中...")
		self.tune_params = ['offset','period','n_estimators','max_depth','min_samples_split','min_samples_leaf','min_weight_fraction_leaf']
		print(self.model_name)
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		self.model.n_estimators = int(x[2])
		self.model.max_depth = int(x[3])
		self.model.min_samples_split = int(x[4])
		self.model.min_samples_leaf = int(x[5])
		self.model.min_weight_fraction_leaf = x[6]
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse


if __name__ == "__main__":
	for src_index in range(1,util.get_src_file_list_length()):
		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30],dtype=np.float32)
		# LinearRegressionOptimizer(src_index,0,bounds)

		
		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,200,5e-4,5e-7,5e-7,5e-7,5e-7],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,500,5e-3,5e-6,5e-6,5e-6,5e-6],dtype=np.float32)
		# BayesianRidgeOptimizer(src_index,1,bounds)

		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,0.8,800,5e-4],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,1.2,1200,5e-2],dtype=np.float32)
		# RidgeOptimizer(src_index,2,bounds)


		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,0.08,0.8,8,80,2],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,0.12,1.2,12,120,10],dtype=np.float32)
		# RidgeCVOptimizer(src_index,3,bounds)

		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,0.8,800,5e-5],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,1.2,1600,5e-4],dtype=np.float32)
		# LassoOptimizer(src_index,4,bounds)

		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,0.8,400,1e-16],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,1.2,800,5e-16],dtype=np.float32)
		# LassoLarsOptimizer(src_index,5,bounds)

		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,400,800,1e-16],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,800,1200,5e-16],dtype=np.float32)
		# LassoLarsCVOptimizer(src_index,6,bounds)

		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,10,0.1],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,100,1.0],dtype=np.float32)
		# AdaBoostRegressorOptimizer(src_index,0,bounds)

		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,5,0.1,0.1],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,100,1.0,1.0],dtype=np.float32)
		# BaggingRegressorOptimizer(src_index,1,bounds)

		# bounds = {}
		# bounds["lower_bounds"] = np.array([36,5,5,0.1,0.1,2,1,0.0,2,0.6],dtype=np.float32)
		# bounds["upper_bounds"] = np.array([60,30,100,1.0,1.0,4,3,0.2,4,0.999],dtype=np.float32)
		# GradientBoostingRegressorOptimizer(src_index,2,bounds)

		bounds = {}
		bounds["lower_bounds"] = np.array([36,5,5,2,2,1,0.0],dtype=np.float32)
		bounds["upper_bounds"] = np.array([60,30,100,4,4,3,0.2],dtype=np.float32)
		RandomForestRegressorOptimizer(src_index,3,bounds)