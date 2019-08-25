import util
from base_optimizer import BaseOptimizer

import xgboost as xgb
from xgboost import XGBRegressor,XGBRFRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys

import numpy as np

optimization_methods = ["random_search",
						# "bayesian_optimization"
						]

class XGBRegressorOptimizer(BaseOptimizer):
	def __init__(self,src_file_index,bounds):
		self.model = XGBRegressor()
		self.model_name = "XGBRegressor"
		self.src = util.get_src_file(src_file_index=src_file_index)
		self.lower_bounds = bounds["lower_bounds"]
		self.upper_bounds = bounds["upper_bounds"]
		self.with_rain = False
		self.optimization_methods = optimization_methods
		self.num_iterations = 200
		self.results = {}
		self.result_save_path = 'optimization_result/with_rain_'+str(self.with_rain)+'/'+self.src.split('.')[0].split('/')[-1]+'/'
		self.optimization()
		self.save_optimization_result()

	def objective_function(self,x):
		print("XGBRegressor优化中...")
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		print(self.model_name)
		self.tune_params = ['offset','period','max_depth',
		# 					'learning_rate',
							'n_estimators','gamma',
							'min_child_weight','max_delta_step','subsample',
							'colsample_bytree','colsample_bylevel','colsample_bynode','reg_alpha',
							'reg_lambda','scale_pos_weight','base_score'
							]
		self.model.max_depth = int(x[2])
		self.model.n_estimators = int(x[3])
		self.model.gamma = x[4]
		self.model.min_child_weight = int(x[5])
		self.model.max_delta_step = int(x[6])
		self.model.subsample = x[7]
		self.model.colsample_bytree = x[8]
		self.model.colsample_bylevel = x[9]
		self.model.colsample_bynode = x[10]
		self.model.reg_alpha = x[11]
		self.model.reg_lambda = x[12]
		self.model.scale_pos_weight = x[13]
		self.model.base_score = x[14]
		self.model.objective = 'reg:squarederror'
		self.model.learning_rate = 0.001
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class XGBRFRegressorOptimizer(BaseOptimizer):
	def __init__(self,src_file_index,bounds):
		self.model = XGBRFRegressor()
		self.model_name = "XGBRFRegressor"
		self.src = util.get_src_file(src_file_index=src_file_index)
		self.lower_bounds = bounds["lower_bounds"]
		self.upper_bounds = bounds["upper_bounds"]
		self.with_rain = False
		self.optimization_methods = optimization_methods
		self.num_iterations = 200
		self.results = {}
		self.result_save_path = 'optimization_result/with_rain_'+str(self.with_rain)+'/'+self.src.split('.')[0].split('/')[-1]+'/'
		self.optimization()
		self.save_optimization_result()

	def objective_function(self,x):
		print("XGBRegressor优化中...")
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		print(self.model_name)
		self.tune_params = ['offset','period','max_depth',
							# 'learning_rate',
		 					'n_estimators',
							'gasmma',
							'min_child_weight','max_delta_step','subsample',
							'colsample_bytree','colsample_bylevel','colsample_bynode','reg_alpha',
							'reg_lambda','scale_pos_weight','base_score'
							]
		self.model.max_depth = int(x[2])
		self.model.n_estimators = int(x[3])
		self.model.gamma = x[4]
		self.model.min_child_weight = int(x[5])
		self.model.max_delta_step = int(x[6])
		self.model.subsample = x[7]
		self.model.colsample_bytree = x[8]
		self.model.colsample_bylevel = x[9]
		self.model.colsample_bynode = x[10]
		self.model.reg_alpha = x[11]
		self.model.reg_lambda = x[12]
		self.model.scale_pos_weight = x[13]
		self.model.base_score = x[14]
		self.model.objective = 'reg:squarederror'
		self.model.learning_rate = 0.001
		self.model.fit(X=train_x,y=train_y)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class LightGBMRegressorOptimizer(BaseOptimizer):
	def __init__(self,src_file_index,bounds):
		self.model = None
		self.model_name = "LightGBMRegressor"
		self.src = util.get_src_file(src_file_index=src_file_index)
		self.lower_bounds = bounds["lower_bounds"]
		self.upper_bounds = bounds["upper_bounds"]
		self.with_rain = False
		self.optimization_methods = optimization_methods
		self.num_iterations = 200
		self.results = {}
		self.result_save_path = 'optimization_result/with_rain_'+str(self.with_rain)+'/'+self.src.split('.')[0].split('/')[-1]+'/'
		self.optimization()
		self.save_optimization_result()

	def objective_function(self,x):
		print("LightGBMRegressor优化中...")
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		print(self.model_name)
		self.tune_params = ['offset','period','num_leaves','learning_rate','feature_fraction','bagging_fraction','bagging_freq']
		params = {
			'task': 'train',
			'boosting_type': 'gbdt',       # 设置提升类型
			'objective': 'regression',     # 目标函数
			'metric': {'mse'},       # 评估函数
			'num_leaves': int(x[2]),            # 叶子节点数
			'learning_rate': x[3],         # 学习速率
			'feature_fraction': x[4],      # 建树的特征选择比例
			'bagging_fraction': x[5],      # 建树的样本采样比例
			'bagging_freq': int(x[6]),          # k 意味着每 k 次迭代执行bagging
			'verbose': 0,                   # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
			'device':'gpu'
		}
		train_x,val_x,train_y,val_y = train_test_split(train_x,train_y,test_size=0.1)
		self.model = lgb.train(params,lgb.Dataset(train_x,train_y),num_boost_round=100,valid_sets=lgb.Dataset(val_x,val_y),early_stopping_rounds=5)
		y_hat = self.model.predict(test_x,num_iteration=self.model.best_iteration)
		mse = mean_squared_error(y_hat,test_y)
		return mse

class CatBoostRegressorOptimizer(BaseOptimizer):
	def __init__(self,src_file_index,bounds):
		self.model = CatBoostRegressor()
		self.model_name = "CatBoostRegressor"
		self.src = util.get_src_file(src_file_index=src_file_index)
		self.lower_bounds = bounds["lower_bounds"]
		self.upper_bounds = bounds["upper_bounds"]
		self.with_rain = False
		self.optimization_methods = optimization_methods
		self.num_iterations = 200
		self.results = {}
		self.result_save_path = 'optimization_result/with_rain_'+str(self.with_rain)+'/'+self.src.split('.')[0].split('/')[-1]+'/'
		self.optimization()
		self.save_optimization_result()

	def objective_function(self,x):
		print("CatBoostRegressor优化中...")
		train_x, test_x, train_y, test_y = util.get_train_test_split(self.src,int(np.round(x[0])),int(np.round(x[1])),with_rain=self.with_rain)
		print(self.model_name)
		self.tune_params = ['offset','period','max_depth','subsample','learning_rate','n_estimators',
							'min_child_samples','max_leaves']
		self.model.max_depth = int(x[2])
		self.model.subsample = x[3]
		self.model.learning_rate = x[4]
		self.model.n_estimators = int(x[5])
		self.model.min_child_samples = x[6]
		self.model.max_leaves = x[7]
		self.model.task_type = 'gpu'
		self.model.iterations = 100
		self.model.loss_function = 'MSE'
		train_x,val_x,train_y,val_y = train_test_split(train_x,train_y,test_size=0.1)
		self.model.fit(train_x,train_y,eval_set=(val_x,val_y),plot=True)
		y_hat = self.model.predict(test_x)
		mse = mean_squared_error(y_hat,test_y)
		return mse

def CATOptimization(src_index):
	params_bound = {}
	params_bound["offset"] = [36,60]
	params_bound['period'] = [5,30]
	params_bound['max_depth'] = [2,4]
	params_bound['subsample'] = [0.1,0.3]
	params_bound['learning_rate'] = [0.0001,0.1]
	params_bound['n_estimators'] = [10,100]
	params_bound['min_child_samples'] = [0.1,0.9]
	params_bound['max_leaves'] = [10,100]
	lower_bounds = []
	upper_bounds = []
	for bounds_value in params_bound.values():
		lower_bounds.append(bounds_value[0])
		upper_bounds.append(bounds_value[1])
	bounds = {}
	bounds["lower_bounds"] = np.array(lower_bounds,dtype=np.float32)
	bounds["upper_bounds"] = np.array(upper_bounds,dtype=np.float32)
	CatBoostRegressorOptimizer(src_index,bounds)


def GBMOptimization(src_index):
	params_bound = {}
	params_bound["offset"] = [36,60]
	params_bound['period'] = [5,30]
	params_bound['num_leaves'] = [10,100]
	params_bound['learning_rate'] = [0.0001,0.1]
	params_bound['feature_fraction'] = [0.1,1.0]
	params_bound['bagging_fraction'] = [0.1,1.0]
	params_bound['bagging_freq'] = [1,10]
	lower_bounds = []
	upper_bounds = []
	for bounds_value in params_bound.values():
		lower_bounds.append(bounds_value[0])
		upper_bounds.append(bounds_value[1])
	bounds = {}
	bounds["lower_bounds"] = np.array(lower_bounds,dtype=np.float32)
	bounds["upper_bounds"] = np.array(upper_bounds,dtype=np.float32)
	LightGBMRegressorOptimizer(src_index,bounds)

def XGBOptimization(src_index,type=None):
	params_bound = {}
	params_bound["offset"] = [36,60]
	params_bound['period'] = [5,30]
	params_bound['max_depth'] = [2,10]
	# params_bound['learning_rate'] = [0.00001,0.1]
	params_bound['n_estimators'] = [80,500]
	params_bound['gamma'] = [0.0,1.0]
	params_bound['min_child_weight'] = [1,10]
	params_bound['max_delta_step'] = [0,2]
	params_bound['subsample'] = [0.2,1.0]
	params_bound['colsample_bytree'] = [0.2,1.0]
	params_bound['colsample_bylevel'] = [0.2,1.0]
	params_bound['colsample_bynode'] = [0.2,1.0]
	params_bound['reg_alpha'] = [0.0,1.0]
	params_bound['reg_lambda'] = [0.0,1.0]
	params_bound['scale_pos_weight'] = [0.1,1.0]
	params_bound['base_score'] = [0.1,1.0]
	lower_bounds = []
	upper_bounds = []
	for bounds_value in params_bound.values():
		lower_bounds.append(bounds_value[0])
		upper_bounds.append(bounds_value[1])
	bounds = {}
	bounds["lower_bounds"] = np.array(lower_bounds,dtype=np.float32)
	bounds["upper_bounds"] = np.array(upper_bounds,dtype=np.float32)
	if(type == None):
		XGBRegressorOptimizer(src_index,bounds)
	elif(type == 'rf'):
		XGBRFRegressorOptimizer(src_index,bounds)

if __name__ == "__main__":
	src_index = sys.argv[1]
	src_index = int(src_index)
	if(src_index < 13):
		if(sys.argv[2] == 'cat'):
			CATOptimization(src_index)
		elif(sys.argv[2] == 'gbm'):
			GBMOptimization(src_index)
		elif(sys.argv[2] == 'xgb'):
			XGBOptimization(int(src_index))
		elif(sys.argv[2] == 'xgbrf'):
			XGBOptimization(int(src_index),type='rf')



