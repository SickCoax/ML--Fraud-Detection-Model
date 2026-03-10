import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score , recall_score , precision_score

df1 = pd.read_csv("dataset/creditcard.csv")

x = df1.drop(["Class"] , axis=1)
y = df1["Class"]

xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size=0.2 , random_state=42 , stratify=y)


param_grid = {"n_estimators" : [200,250,300]}

grid = GridSearchCV(RandomForestClassifier(max_depth=None , class_weight="balanced") , param_grid , cv=5 , scoring="recall" , n_jobs=-1)

grid.fit(xtrain , ytrain)

model = grid.best_estimator_

ypred = model.predict(xtest)
print()

# ---------------------
# Evaluation Metric
# ---------------------

print(f"F1 Score : {f1_score(ytest , ypred)}")
print(f"Recall : {recall_score(ytest , ypred)}")
print(f"Precision : {precision_score(ytest , ypred)}")