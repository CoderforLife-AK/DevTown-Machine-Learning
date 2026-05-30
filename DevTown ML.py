#Import all the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Plot for Quantity vs Unit price and encode Country 
df=pd.read_csv("customer_retail.csv")
df=df.dropna()
df=df[['Quantity','UnitPrice','Country']]
encoder=LabelEncoder()
df['Country_encoded']=encoder.fit_transform(df['Country'])
plt.figure(figsize=(8,5))
plt.scatter(df['Quantity'],df['UnitPrice'])
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title('Scattter Plot of Quantity vs unitPrice')
plt.show()

#Encoding for country and assigning fields for x and y
x=df[['Quantity','UnitPrice']]
y=df['Country_encoded']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Logisctic Regression Model
print("Logistic Regression")
log_model=LogisticRegression()
log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

#Decision Tree Model
print("Decision Tree")
dt_model=DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
y_pred=dt_model.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print("Accuracy:",acc)
print(confusion_matrix(y_test,y_pred))

#KNN Model
print("KNN")
knn_model=KNeighborsClassifier()
knn_model.fit(x_train,y_train)
y_pred=knn_model.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))

#Bundling up All Models and their accuracies
model=['Logistic Regression','Decision Tree','KNN']
accuracies=[accuracy,acc,acc]

#Actual comparison of the three models based on their accuracies
plt.figure(figsize=(8,5))
plt.bar(model,accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()
