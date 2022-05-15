import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from warnings import filterwarnings

filterwarnings("ignore")

data =pd.read_csv("btc.csv") #verileri okuduk
data = data.drop("time",axis=1) #time verisini sildik


#veriler hakkınd abilgi aldık
print(data.head())
print(data.shape)
print(data.info())
print(data.describe().T)


#değişkenlerin birbirlerini etkileme oranlarına bakıldı -1 en düşük +1 en yüksek
corr_matrix =data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.show()

threshold = 0.75
filtre = np.abs(corr_matrix["close"])>threshold
corr_feature =corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_feature].corr(),annot=True,fmt=".2f")
plt.show()


#veri setlerini ayırdık
x=data.drop(["close"],axis=1)# tahmin işlemini yapacak değişkenler
y=data.close # tahmin edilecek değişken


X_train,X_test ,Y_train,Y_test =train_test_split(x,y,test_size=0.1,random_state=42)
#veri setlerini ayırdık train ile veri eğitilecek test ilede veri test edilecek

#standarlaştırma
scaler=StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)



#multi linear regression
from sklearn import metrics

lr = LinearRegression().fit(X_train,Y_train)#model eğitildi ağırlıklar bulundu
y_pred_lr =lr.predict(X_test) # tahmin işlemi gerçekleşti
mse =mean_squared_error(Y_test,y_pred_lr)
print("Lineer Regresyon MSE :" ,mse)
print('Lineer MAE:', metrics.mean_absolute_error(Y_test, y_pred_lr))
cv_lr =cross_val_score(lr,X_train,Y_train,cv=10,scoring="neg_mean_squared_error")
ortalama =np.sqrt(np.mean(-(cv_lr)))
print("multi linear regression cv",ortalama)

#Grafik
pred=pd.DataFrame(y_pred_lr)
plt.figure(figsize = (18, 6))
index=Y_test.reset_index()["close"]
ax=index.plot(label="original values")
ax=pred[0].plot(label = "predicted values")
plt.legend(loc='upper right')
plt.title("Test vs Pred")
plt.show()

print("********************************")
print("********************************")



#ridge regression
from sklearn.linear_model import Ridge
ridge= Ridge().fit(X_train,Y_train)
y_pred_r =ridge.predict(X_test)
mse =mean_squared_error(Y_test,y_pred_r)
print("ridge Regresyon MSE :" ,mse)
print('ridge MAE:', metrics.mean_absolute_error(Y_test, y_pred_r))
cv_ridge =cross_val_score(ridge,X_train,Y_train,cv=10,scoring="neg_mean_squared_error")
ortalama =np.sqrt(np.mean(-(cv_ridge)))
print("Ridge cv",ortalama)

#Grafik
pred=pd.DataFrame(y_pred_r)
plt.figure(figsize = (18, 6))
index=Y_test.reset_index()["close"]
ax=index.plot(label="original values")
ax=pred[0].plot(label = "predicted values")
plt.legend(loc='upper right')
plt.title("Test vs Pred")
plt.show()

print("********************************")
print("********************************")

#lasso regression
from sklearn.linear_model import Lasso
lasso=Lasso().fit(X_train,Y_train)
y_pred_la =lasso.predict(X_test)
mse =mean_squared_error(Y_test,y_pred_la )
print("lasso Regresyon MSE :" ,mse)
print('lasso MAE:', metrics.mean_absolute_error(Y_test, y_pred_la ))
cv_Lasso =cross_val_score(lasso,X_train,Y_train,cv=10,scoring="neg_mean_squared_error")
ortalama =np.sqrt(np.mean(-(cv_Lasso)))
print("Lasso cv",ortalama)

#Grafik
pred=pd.DataFrame(y_pred_la )
plt.figure(figsize = (18, 6))
index=Y_test.reset_index()["close"]
ax=index.plot(label="original values")
ax=pred[0].plot(label = "predicted values")
plt.legend(loc='upper right')
plt.title("Test vs Pred")
plt.show()

print("********************************")
print("********************************")



#desicion tree regression
from sklearn.tree import DecisionTreeRegressor

dtreeRegressor= DecisionTreeRegressor().fit(X_train,Y_train)
y_pred_dtr =dtreeRegressor.predict(X_test)
mse =mean_squared_error(Y_test,y_pred_dtr)
print("desicion Regresyon MSE :" ,mse)
print('desicion MAE:', metrics.mean_absolute_error(Y_test, y_pred_dtr))
cv_dtreeRegressor =cross_val_score(dtreeRegressor,X_train,Y_train,cv=10,scoring="neg_mean_squared_error")
ortalama =np.sqrt(np.mean(-(cv_dtreeRegressor)))
print("DecisionTreeRegressor cv",ortalama)

#Grafik
pred=pd.DataFrame(y_pred_dtr)
plt.figure(figsize = (18, 6))
index=Y_test.reset_index()["close"]
ax=index.plot(label="original values")
ax=pred[0].plot(label = "predicted values")
plt.legend(loc='upper right')
plt.title("Test vs Pred")
plt.show()

print("********************************")
print("********************************")

#support vector regrresion
from sklearn.svm import SVR


svectorRegrresion=SVR().fit(X_train,Y_train)
y_pred_svr =svectorRegrresion.predict(X_test)
mse =mean_squared_error(Y_test,y_pred_svr)
print("support vector regrresion MSE :" ,mse)
print('support vector regrresion MAE:', metrics.mean_absolute_error(Y_test, y_pred_svr))
cv_supportvektor =cross_val_score(svectorRegrresion,X_train,Y_train,cv=10,scoring="neg_mean_squared_error")
ortalama =np.sqrt(np.mean(-(cv_supportvektor)))
print("cross validation support cv",ortalama)

#Grafik
pred=pd.DataFrame(y_pred_svr)
plt.figure(figsize = (18, 6))
index=Y_test.reset_index()["close"]
ax=index.plot(label="original values")
ax=pred[0].plot(label = "predicted values")
plt.legend(loc='upper right')
plt.title("Test vs Pred")
plt.show()

print("********************************")
