#++++++++++++++++++++++++++++++++++
# @author : yongsheng hou
# @date   : 2019.07.07
# @Email  : houys@tju.edu.cn
#++++++++++++++++++++++++++++++++++

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from sklearn.linear_model import *
import sklearn.linear_model
import sklearn.ensemble

from sklearn.preprocessing import MinMaxScaler

linear_model_list=[
	"LinearRegression",   # 0
	"BayesianRidge",              # 1
	"Ridge",      # 2
	"RidgeCV",            # 3
	"Lasso",              # 4
	"LassoCV",            # 5
	"LassoLars",          # 6
	"LassoLarsCV"         # 7
]

ensemble_model_list=[
	"AdaBoostRegressor",            # 0 
	"BaggingRegressor",             # 1
	"GradientBoostingRegressor",    # 2
	"RandomForestRegressor"        # 3
	#"VotingRegressor",              # 4
	#"HistGradientBoostingRegressor" # 5
]

svm_model_list=[
	"Linear"
]

src_file_list=[
	"data/seasons/2015_spring.csv",
	"data/seasons/2015_summer.csv",
	"data/seasons/2015_autumn.csv",
	"data/seasons/2015_winter.csv",
	"data/seasons/2016_spring.csv",
	"data/seasons/2016_summer.csv",
	"data/seasons/2016_autumn.csv",
	"data/seasons/2016_winter.csv",
	"data/seasons/2017_spring.csv",
	"data/seasons/2017_summer.csv",
	"data/seasons/2017_autumn.csv",
	"data/seasons/2017_winter.csv",
	"data/datafill_modified.csv",
	"data/alltest.csv"
]

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5),scoring='mean_squared_error',n_jobs=16):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    
    from sklearn.model_selection  import learning_curve

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

def get_model_list_length(is_ensemble=False):
	if(is_ensemble):
		return len(ensemble_model_list)
	else:
		return len(linear_model_list)

def get_src_file_list_length():
	return len(src_file_list)

def get_scaled_data(data):
	scaler = MinMaxScaler()
	return scaler.fit_transform(data),scaler

def get_model(is_ensemble=False,model_index=0):
	if(is_ensemble):
		if(model_index<get_model_list_length(is_ensemble=True) and model_index >= 0):
			name = ensemble_model_list[model_index]
			m = getattr(sklearn.ensemble,name)
			model = m()
			return model,name
		else:
			print("model_index invalid")
			return None
	else:
		if(model_index<get_model_list_length(is_ensemble=False) and model_index >= 0):
			name = linear_model_list[model_index]
			m = getattr(sklearn.linear_model,name)
			model = m()
			return model,name
		else:
			print("model_index invalid")
			return None

def get_src_file(src_file_index):
	return src_file_list[src_file_index]

def read_local_data(src):
	'''
	##function:
		get source data from local file in numpy.array form
	'''
	return np.array(pd.read_csv(src,header= None))

def get_random_state_num(seed=100):
	'''
	##function:
		get random state num
	'''
	np.random.seed(seed)
	return np.random.randint(0,10000);


def get_reshaped_data_by_OP(src=None,offset=48,period=10,with_rain=False,scaler=None,data=None,is_scaled=False):

	if(data is None and src is not None):
		data = read_local_data(src)  #source data in numpy.array form
	if(is_scaled == False):
		if(scaler is None):
			data,scaler = get_scaled_data(data)
		else:
			data = scaler.transform(data)
	zhexi_in = data[:,0]     	 #water in
	xiaoxi_out = data[:,1]		 #water out		
	lsj_add = data[:,2]	 #rain observation 1
	xinhua_add = data[:,3]  #rain observation 2
	zhexi_add = data[:,4]   #rain observation 3
	
	size = len(zhexi_in)

	if(with_rain):
		source_xiaoxi_out=[[] for i in range(period)]
		source_zhexi_in = [[] for i in range(period)]
		source_lsj_add=[[] for i in range(period)]
		source_xinhua_add=[[] for i in range(period)]
		source_zhexi_add=[[] for i in range(period)]
		for i in range(period):
			source_xiaoxi_out[i]=xiaoxi_out[i :size-offset-period+i]
			source_zhexi_in[i] = zhexi_in[i+offset:size-period+i]
			source_lsj_add[i]=lsj_add[offset+i :size-period+i]
			source_xinhua_add[i]=xinhua_add[offset+i :size-period+i]
			source_zhexi_add[i]=zhexi_add[offset+i :size-period+i]
		X = np.hstack((np.array(source_xiaoxi_out).transpose(1,0),
							np.array(source_zhexi_in).transpose(1,0),
							np.array(source_lsj_add).transpose(1,0),
							np.array(source_xinhua_add).transpose(1,0),
							np.array(source_zhexi_add).transpose(1,0),))

		y = zhexi_in[offset+period:]
	else:
		source_xiaoxi_out=[[] for i in range(period)]
		source_zhexi_in = [[] for i in range(period)]
		for i in range(period):
			source_xiaoxi_out[i]=xiaoxi_out[i :size-offset-period+i]
			source_zhexi_in[i] = zhexi_in[i+offset:size-period+i]
		X = np.hstack((np.array(source_xiaoxi_out).transpose(1,0),
		                 np.array(source_zhexi_in).transpose(1,0)))
		y = zhexi_in[offset+period:]
	return X,y

def get_train_test_split(src=None,data=None,offset=48,period=20,test_size=0.1,with_rain=False,seed=100):
	X,y = get_reshaped_data_by_OP(src,offset,period,with_rain=with_rain,data=data)
	train_x, test_x, train_y, test_y = train_test_split(X, y, 
										   test_size=test_size,
										   random_state=get_random_state_num(seed=seed))
	return train_x, test_x, train_y, test_y

if __name__ == '__main__':
	X,y = get_reshaped_data_by_OP('data/seasons/2015_autumn.csv',5,5,with_rain=False,is_scaled = True)
	print(X.shape,y.shape)
	# train_x, test_x, train_y, test_y = get_train_test_split('data/seasons/2015_autumn.csv',5,5,with_rain=True)
	# print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
	print(X,y)