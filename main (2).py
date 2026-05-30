#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np


# In[90]:


df=pd.read_csv(r"C:\Users\medag\Downloads\jyothi.csv")
df.head()


# In[92]:


# Pivot to convert pollutants into columns
df_wide = df.pivot_table(index=['country', 'state', 'city', 'station', 'last_update'],
                         columns='pollutant_id',
                         values='pollutant_avg').reset_index()
df_wide = df_wide.fillna(df_wide.mean(numeric_only=True))


# In[94]:


# Approximate AQI using PM10 (or PM2.5 if available)
df_wide['AQI'] = df_wide['PM10']

def classify_aqi(aqi):
    if pd.isna(aqi):
        return Unknown
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

df_wide['AQI_Category'] = df_wide['AQI'].apply(classify_aqi)


# In[96]:


# Fill missing values in original dataset
cols = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
df[cols] = df[cols].fillna(df[cols].mean())

# After pivot
df_wide.fillna(df_wide.mean(numeric_only=True), inplace=True)


# In[98]:


# Convert Strings to Datetime
df['last_update'] = pd.to_datetime(df['last_update'], dayfirst=True, errors='coerce')


# In[100]:


df.head()


# In[102]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[104]:


# Drop rows with missing target values 
data_cleaned = df.dropna(subset=["pollutant_avg"]).copy()


# In[106]:


# Categorize 'pollutant_avg' into classes (Low, Moderate, High)
# Using arbitrary thresholds for demonstration 
bins = [0, 50, 100, float('inf')]
labels = ["Low", "Moderate", "High"]
data_cleaned["pollutant_class"] = pd.cut(data_cleaned["pollutant_avg"], bins=bins, labels=labels)


# In[108]:


# Encode categorical features
label_encoders = {}
categorical_columns = ["country", "state", "city", "station", "pollutant_id"]

for col in categorical_columns:
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col])
    label_encoders[col] = le


# In[110]:


# Drop unnecessary columns
data_preprocessed = data_cleaned.drop(columns=["last_update", "pollutant_min", "pollutant_max", "pollutant_avg"])


# In[65]:


#split data into features(x) and target(y)


# In[112]:


X = data_preprocessed.drop(columns=["pollutant_class"])
y = data_preprocessed["pollutant_class"]


# In[114]:


from sklearn.preprocessing import LabelEncoder

X = df_wide[['PM10', 'NH3', 'OZONE']].fillna(df_wide[['PM10', 'NH3', 'OZONE']].mean())
le = LabelEncoder()
y = le.fit_transform(df_wide['AQI_Category'])


# In[68]:


#split into training and testing set


# In[116]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, pd.Series(y_train).value_counts(), pd.Series(y_test).value_counts()


# In[70]:


#KNN


# In[118]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# In[120]:


knn_class=KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2 )
knn_class.fit(X_train,y_train)


# In[122]:


y_pred_knn=knn_class.predict(X_test)
Accuracy_Knn=round((metrics.accuracy_score(y_test, y_pred_knn)*100),2)
print('Accuracy (KNN): ',Accuracy_Knn,"%")


# In[74]:


#classification report


# In[124]:


unique_classes = np.unique(np.concatenate((y_test, y_pred_knn)))

print(classification_report(
    y_test,
    y_pred_knn,
    target_names=[f'class {i}' for i in unique_classes],
    zero_division=0
))




# In[76]:


#Decision Tree


# In[126]:


from sklearn.tree import DecisionTreeClassifier


# In[128]:


DT_class=DecisionTreeClassifier(criterion='entropy', random_state=0)
DT_class.fit(X_train, y_train)


# In[130]:


y_pred_DT= DT_class.predict(X_test)
Accuracy_DT=round((metrics.accuracy_score(y_test, y_pred_DT)*100),2)
print('Accuracy (Decision Tree): ',Accuracy_DT,"%")


# In[80]:


#Classification Report


# In[134]:


unique_classes = np.unique(np.concatenate((y_test, y_pred_DT)))

print(classification_report(y_test, y_pred_DT, target_names=[f'class {i}' for i in unique_classes]))


# In[ ]:


#Random Forest


# In[136]:


from sklearn.ensemble import RandomForestClassifier


# In[138]:


random_class= RandomForestClassifier(n_estimators= 20, criterion="entropy")
random_class.fit(X_train, y_train)


# In[140]:


y_pred_RF= random_class.predict(X_test)
Accuracy_RF=round((metrics.accuracy_score(y_test, y_pred_RF)*100),2)
print('Accuracy (Random Forest): ',Accuracy_RF,"%")


# In[142]:


#Classification Report


# In[144]:


unique_classes = np.unique(np.concatenate((y_test, y_pred_RF)))
print(classification_report(
    y_test,
    y_pred_RF,
    target_names=[f'class {i}' for i in unique_classes],
    zero_division=0
))


# In[146]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Save model
joblib.dump(best_model, "aqi_rf_model.pkl")
joblib.dump(le, "label_encoder.pkl")


# In[147]:


#Data Visualization


# In[154]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[149]:


#Histogram


# In[156]:


plt.figure(figsize=(8, 6))
sns.histplot(df, x="pollutant_avg", kde=True, bins=30, color="blue", label="Pollutant Avg")
plt.title("Distribution of Pollutant Average Levels")
plt.xlabel("Pollutant Average")
plt.ylabel("Frequency")
plt.legend()  # now it has something to show
plt.show()


# In[ ]:


#Heat Map


# In[158]:


# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include='number')

# Now plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:


#Count Plot


# In[162]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="pollutant_id", hue="pollutant_id", 
              order=df["pollutant_id"].value_counts().index, 
              palette="viridis", legend=False)
plt.title("Count of Each Pollutant ID")
plt.xlabel("Pollutant ID")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[164]:


import joblib

# Save the best trained model to a file
joblib.dump(best_model, "aqi_rf_model.pkl")
print("Model loaded successfully!")


# In[166]:


import os
os.path.getsize("main.ipynb")


# In[ ]:




