import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , StandardScaler ,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import  mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


dataset = pd.read_csv("Bangalore  house data.csv")
print(dataset.head())
print(dataset.describe())
print(dataset.info())
print((dataset.isna().sum().sum()))


dataset.drop(columns=['area_type','availability','society','balcony'],inplace=True)


selected = dataset.iloc[:,[0,1]] 
si1=SimpleImputer(strategy="most_frequent")
selected = si1.fit_transform(selected)
dataset.iloc[:,[0,1]] = selected
dataset = pd.DataFrame(dataset) 
# print(dataset.isnull().sum())
print((dataset.isna().sum()))


le = LabelEncoder()
st = StandardScaler()



BHK=[]
sqrft=[]
for i in range(dataset.shape[0]):
  BHK.append((int)(dataset["size"][i][0]))
  T=dataset["total_sqft"][i]
  if(dataset["total_sqft"][i].isdigit() == True):
      sqrft.append(int(dataset["total_sqft"][i]))
  elif("-" in T ):
      t1 ,t2 = map(float,T.split("-"))
      sqrft.append(int((t1+t2)/ 2))
  else:
      sqrft.append(None)

dataset["size" ] = BHK
dataset["total_sqft"] = sqrft
dataset=dataset.dropna()
dataset["total_sqft"]=(dataset["total_sqft"]).astype(int)


def strip_whitespace(x):
    return x.strip()

dataset["location"] = dataset["location"].apply(strip_whitespace)



location_c=dataset["location"].value_counts()
location_c_less = location_c[location_c <= 10]

def replace_location(x):
    if x in location_c_less:
        return 'other'
    else:
        return x

dataset["location"] = dataset["location"].apply(replace_location)



dataset = dataset[((dataset["total_sqft"]/dataset["size"]) >= 400)]
dataset["price_per_sqrft"] = dataset["price"]*100000 / dataset["total_sqft"]
print(dataset["price_per_sqrft"].describe())



#removing ouliers

def out_Sqrft(df):
    out = pd.DataFrame()
    for k , sub in df.groupby("location"):
        m = np.mean(sub.price_per_sqrft)
        std = np.std(sub.price_per_sqrft)
        n_df = sub[(sub.price_per_sqrft >= (m-std)) & (sub.price_per_sqrft <= (m+std))]
        out = pd.concat([out , n_df],ignore_index=True)
    return out

dataset = out_Sqrft(dataset)



def out_bhk(df):
    E_in = np.array([])
    for loc , loc_df in df.groupby("location"):
        bhk_I = {}
        for bhk , bhk_df in loc_df.groupby("size"):
            bhk_I[bhk] = {"Mean : " :np.mean(bhk_df.price_per_sqrft),"Std : " :np.std(bhk_df.price_per_sqrft),"Count : " :bhk_df.shape[0]}
        for bhk,bhk_df in loc_df.groupby("size"):
            I = bhk_I.get(bhk-1)
            if I and I["Count : "]>5:
                E_in = np.append(E_in,bhk_df[bhk_df.price_per_sqrft<(I['Mean : '])].index.values)
    return df.drop(E_in,axis="index")

dataset=out_bhk(dataset)



dataset.drop(columns=["price_per_sqrft"],inplace=True)


dg = dataset
dg["location"] = le.fit_transform(dg["location"])
print(dg.head())
corr = np.corrcoef(dg.values.T)
hm = sb.heatmap(corr , annot=True)
plt.show()
sb.pairplot(dg)
plt.show()


tr = dataset["price"]
dataset.drop(columns=["price"] ,inplace=True ) 

cala_col=dataset["location"].values.reshape(-1, 1)
dataset.drop(columns=["location"],inplace=True,axis=1)
dataset.columns = [241,242,243]
oe=OneHotEncoder(sparse_output=False)
df = oe.fit_transform(cala_col)
df= pd.DataFrame(df)
dataset.reset_index(drop=True, inplace=True)
dataset = pd.concat([df, dataset], axis=1)
# print(dataset.head())


xtrain , xtest , ytrain, ytest = train_test_split(dataset , tr , test_size=0.18)

print("\n","----"*20,"Applying Classification Algorithms","----"*20,"\n")

print("\n","**"*20,"Linear Regression","**"*20)
le=LinearRegression()
P1 = make_pipeline(st,le)
P1.fit(xtrain,ytrain)
P1_train_res = P1.predict(xtrain)
P1_test_res = P1.predict(xtest)
print("#"*15,"Training","#"*15,"\t"*2,"   ","#"*15,"Testing","#"*15)
# print("    MSE      : ",mean_squared_error(ytrain,P1_train_res),"\t"*4,"MSE      : ",mean_squared_error(ytest,P1_test_res))
print("    R2_Score : ",r2_score(ytrain,P1_train_res),"\t"*4,"R2_Score : ",r2_score(ytest,P1_test_res))





print("\n\n","**"*20,"Random Forest Regressor","**"*20)
re = RandomForestRegressor()
P3 = make_pipeline(st,re)
P3.fit(xtrain,ytrain)
rf_train_res = P3.predict(xtrain)
rf_test_res = P3.predict(xtest)
print("#"*15,"Training","#"*15,"\t"*3," ","#"*15,"Testing","#"*15)
# # print("    MSE      : ",mean_squared_error(ytrain,rf_train_res),"\t"*4,"      MSE      : ",mean_squared_error(ytest,rf_test_res))
print("    R2_Score : ",r2_score(ytrain,rf_train_res),"\t"*4,"      R2_Score : ",r2_score(ytest,rf_test_res))




print("\n\n","**"*20,"KNeighbors Regressor","**"*20)
knn=KNeighborsRegressor()
P2 = make_pipeline(st,knn)
P2.fit(xtrain,ytrain)
knntrainr=P2.predict(xtrain)
P2_test_res=P2.predict(xtest)
print("#"*15,"Training","#"*15,"\t"*3,"#"*15,"Testing","#"*15)
# # print("    MSE      : ",mean_squared_error(ytrain,knntrainr),"\t"*4,"    MSE      : ",mean_squared_error(ytest,P2_test_res))
print("    R2_Score : ",r2_score(ytrain,knntrainr),"\t"*4,"    R2_Score : ",r2_score(ytest,P2_test_res))




print("\n\n","**"*20,"Decision Tree Regressor","**"*20)
DTR=DecisionTreeRegressor()
dt=make_pipeline(st,DTR)
dt.fit(xtrain,ytrain)
dt_train_res=dt.predict(xtrain)
dt_test_res=dt.predict(xtest)
print("#"*15,"Training","#"*15,"\t"*3," ","#"*15,"Testing","#"*15)
# # print("    MSE      : ",mean_squared_error(ytrain,dt_train_res),"\t"*4,"      MSE      : ",mean_squared_error(ytest,dt_test_res))
print("    R2_Score : ",r2_score(ytrain,dt_train_res),"\t"*4,"      R2_Score : ",r2_score(ytest,dt_test_res))





print("\n\n","**"*20,"Gradient Boosting Regressor","**"*20)
gr = GradientBoostingRegressor(max_depth=5,max_leaf_nodes=32,learning_rate=0.1)
P4 = make_pipeline( st , gr)
P4.fit(xtrain , ytrain)

gr_train=P4.predict(xtrain)
gr_test=P4.predict(xtest)

print("#"*15,"Training","#"*15,"\t"*3,"     ","#"*15,"Testing","#"*15)
# # print("    MSE      : ",mean_squared_error(ytrain,gr_train),"\t"*5,"  MSE      : ",mean_squared_error(ytest,gr_test))
print("    R2_score : ",r2_score(ytrain,gr_train),"\t"*5,"  R2_score : ",r2_score(ytest,gr_test))



#plots
#LR
# Plotting for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(ytrain, P1_train_res, color='blue', label='Predicted (Training)')
plt.plot([min(ytrain), max(ytrain)], [min(ytrain), max(ytrain)], linestyle='--', color='red', label='Ideal Prediction')
plt.title('Linear Regression: Actual vs Predicted (Training)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.show()

# Residual plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(P1_train_res, ytrain - P1_train_res, color='blue', label='Residuals (Training)')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Residuals')
plt.title('Linear Regression: Residual Plot (Training)')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Distribution plot of residuals for Linear Regression
plt.figure(figsize=(10, 6))
sb.histplot(ytrain - P1_train_res, kde=True, color='blue', label='Residuals (Training)')
plt.title('Linear Regression: Distribution of Residuals (Training)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()



#GBR
# Plotting for training data
plt.figure(figsize=(10, 6))
plt.scatter(ytrain, gr_train, color='blue', label='Predicted (Training)')
plt.plot([min(ytrain), max(ytrain)], [min(ytrain), max(ytrain)], linestyle='--', color='red', label='Ideal Prediction')
plt.title('Gradient Boosting: Actual vs Predicted (Training)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.show()

# Plotting for test data
plt.figure(figsize=(10, 6))
plt.scatter(ytest, gr_test, color='green', label='Predicted (Test)')
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], linestyle='--', color='orange', label='Ideal Prediction')
plt.title('Gradient Boosting: Actual vs Predicted (Test)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.show()

# Residual plot for training data
plt.figure(figsize=(10, 6))
plt.scatter(gr_train, ytrain - gr_train, color='blue', label='Residuals (Training)')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Residuals')
plt.title('Gradient Boosting: Residual Plot (Training)')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Residual plot for test data
plt.figure(figsize=(10, 6))
plt.scatter(gr_test, ytest - gr_test, color='green', label='Residuals (Test)')
plt.axhline(y=0, color='orange', linestyle='--', label='Zero Residuals')
plt.title('Gradient Boosting: Residual Plot (Test)')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Distribution plot of residuals for training data
plt.figure(figsize=(10, 6))
sb.histplot(ytrain - gr_train, kde=True, color='blue', label='Residuals (Training)')
plt.title('Gradient Boosting: Distribution of Residuals (Training)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Distribution plot of residuals for test data
plt.figure(figsize=(10, 6))
sb.histplot(ytest - gr_test, kde=True, color='green', label='Residuals (Test)')
plt.title('Gradient Boosting: Distribution of Residuals (Test)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

