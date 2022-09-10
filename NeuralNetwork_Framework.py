import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("penguins_size.csv")

df=df.dropna()
df=df[df['sex'] != "."]

X=df.drop("species",axis=1)
y=pd.DataFrame(df["species"])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(X.island.unique().tolist())
X['island']=le.transform(X['island'])
le.fit(X.sex.unique().tolist())
X['sex']=le.transform(X['sex'])
le.fit(y.species.unique().tolist())
y=le.transform(y)
y=pd.DataFrame(y,columns=['species'])

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=1)

grid={"hidden_layer_sizes":[(6,2,2),(6,4,2),(6,2),(6,4),(6,1)],
    "activation":["relu","logistic","tanh","identity"],
    "solver":["sgd","adam"],
    "learning_rate":["adaptive"],
    "learning_rate_init":[0.01,.05,0.1,0.5,1],
    "max_iter":[300],
    "random_state":[5]
}
model_gs=GridSearchCV(estimator=MLPClassifier(),param_grid=grid,refit=True,verbose=0,return_train_score=True)
model_gs.fit(X_train,y_train)

print("Best parameters: ",model_gs.best_params_)
print("Training score: ",model_gs.score(X_train,y_train))
print("Testing score: ",model_gs.score(X_test,y_test))

y_pred=model_gs.predict(X_test)
for i in range(len(y_pred)):
    print("Real:",y_test["species"].tolist()[i], "| Predicted:",y_pred[i])