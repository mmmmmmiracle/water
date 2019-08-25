# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @Author: Yongsheng Hou
# @Email: houys@tju.edu.cn
# @Date: 2019-3-2
# @Copyright: Copyright (c) 2019. All rights reserved
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
import os

class DataLoader:
	"""
	##function:
		get seasons data and store them in a path
	##parameters:
		@1 src_data_path: the path of source file
		@2 seasons_data_path: the path of seasons data savd in
	##methods:
		@1 read_data(): get source file whole content
		@2 get_seasons_data: divide the whole content into seasons data
	"""
	def __init__(self, src_data_path, seasons_data_path):
		self.src_data_path = src_data_path
		self.seasons_data_path = seasons_data_path

	def read_data(self):
		""" 
		##function: 
			get the whole content of given source file
		##parameters:
			@1 src: the path of source file
		##return:
			@1 data: the content of source file, it's a DataFrame
		"""
		data = pd.read_csv(self.src_data_path,parse_dates=[0],index_col='TIME')
		return data
	
	def get_seasons_data(self,data):
		'''
		##function:
			divide the whole content into seasons data
		##parameters:
			@1 data: type is DataFrame, total source data
		'''
		years = ['2015','2016','2017','2018']
		seasons = {"spring": ['1', '3'], "summer": ['4', '6'], "autumn": ['7', '9'], "winter": ['10', '12']}
		days = {'3':'31','6':'30','9':'30','12':'31'}
		for year in years:
			for season in seasons:
				print(year+'_'+season)
				if(year == '2018' and season == 'summer'):
					break
				else:
					trunc_before = year+'/'+seasons[season][0]+'/1 0:00'
					trunc_after = year+'/'+seasons[season][1]+'/'+days[seasons[season][1]]+'/ 23:00'
					seasonData = data.truncate(before=trunc_before,after=trunc_after)
					file_name = os.path.join(self.seasons_data_path, '{0}_{1}.csv'.format(year , season))
					if not os.path.exists(self.seasons_data_path):
						os.makedirs(self.seasons_data_path)
					seasonData.to_csv(file_name,header = None, index = None)

if __name__ == '__main__':
	dataloader = DataLoader(src_data_path='datafill.csv', seasons_data_path='seasons/')
	data = dataloader.read_data()
	dataloader.get_seasons_data(data)
