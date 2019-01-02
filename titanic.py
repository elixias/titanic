#titanic.py

import pandas as pd
from sklearn.linear_model import ElasticNet as en
from sklearn.linear_model import Ridge as ridge
from sklearn.linear_model import LogisticRegression as logistic

train_data = pd.read_csv("train.csv",index_col="PassengerId")
df = train_data[["Pclass","Sex","Age","Fare","Embarked","Survived"]].dropna()#,"SibSp","Parch"


test_data = pd.read_csv("test.csv",index_col="PassengerId")
test_data = test_data[["Name","Pclass","Sex","Age","Fare","Embarked"]].dropna()#,"SibSp","Parch"

df_x = df.iloc[:,:len(df.columns)-1]#.values
df_y = df.iloc[:,-1].values
namelist = pd.DataFrame(test_data.values)
test_data = test_data.iloc[:,1:len(test_data.columns)]

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

ct = make_column_transformer(
		(StandardScaler(),['Age', 'Fare']),
		(OneHotEncoder(), ['Pclass','Sex','Embarked'])
)

print(df_x.head(20))
preprocessed = ct.fit_transform(df_x)
preprocessed= pd.DataFrame(preprocessed).drop([5],axis=1)
print(pd.DataFrame(preprocessed).head(20))
test_processed = ct.transform(test_data)
test_processed = pd.DataFrame(test_processed).drop([5],axis=1)
#print(preprocessed)
#preprocess data ie hotencoder, standardise

"""model = en(
	alpha=0.05,
	l1_ratio=0.05,
	#fit_intercept=False,
	#normalize=False,
	#precompute=False,
	#max_iter=1000,
	#copy_X=False,
	#tol=0.0001,
	#warm_start=True,
	#positive=False,
	#random_state=0,
	#selection='cyclic'
)"""

#model = ridge()

model = logistic(
penalty="l2",
#dual=False,
#tol=, 
#C=,
fit_intercept=True,
#intercept_scaling=,
#class_weight=,
#random_state=,
solver="lbfgs",
#max_iter ,
#multi_class ,
#verbose 
#warm_start 
#n_jobs 
)

model.fit(preprocessed,df_y)

print("Coef:",str(model.coef_))

print("Intercept",model.intercept_)

#print(model.get_params)

print("Score:",model.score(preprocessed,df_y))

#print(namelist)
result = pd.DataFrame(model.predict(test_processed))
#print([result[result>0.5].dropna().index.values])
#print(result[result>0.5].dropna().index)

arr = result[result>0.5].dropna().index.values.tolist()
print("Predicting test list survivors")
print(namelist.loc[arr][0])#result[result>0.5].dropna().index.values
