for req in $(cat srcs.txt); do python XGBOptimizer.py $req xgbrf; done
for req in $(cat srcs.txt); do python XGBOptimizer.py $req xgb; done
for req in $(cat srcs.txt); do python XGBOptimizer.py $req gbm; done
for req in $(cat srcs.txt); do python XGBOptimizer.py $req cat; done
