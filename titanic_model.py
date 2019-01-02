from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

target = titanic.survived.values
features = titanic[['pclass', 'sex', 'age', 'fare', 'embarked']].copy()
# I still fill the missing values for the embarked column, because we cannot (yet) easily handle categorical missing values
features['embarked'].fillna(features['embarked'].value_counts().index[0], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)

numerical_features = features.dtypes == 'float'
categorical_features = ~numerical_features

preprocess = make_column_transformer(
    (numerical_features, make_pipeline(SimpleImputer(), StandardScaler())),
    (categorical_features, OneHotEncoder()))

model = make_pipeline(
    preprocess,
    LogisticRegression())
	
model.fit(X_train, y_train)
print("logistic regression score: %f" % model.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
param_grid = {
    'columntransformer__pipeline__simpleimputer__strategy': ['mean', 'median'],
    'logisticregression__C': [0.1, 1.0, 1.0],
    }
	
grid_clf = GridSearchCV(model, param_grid, cv=10, iid=False)
grid_clf.fit(X_train, y_train);
grid_clf.best_params_

print("best logistic regression from grid search: %f" % grid_clf.best_estimator_.score(X_test, y_test))