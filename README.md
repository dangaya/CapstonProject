```python
#importing the necessary libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
#import the dataset to my IDE
data = pd.read_csv(r"C:\Users\ABBA\Downloads\covid 19\covid_19_clean_complete.csv")
```

Data Preprocessing


```python
#checking the first 5 rows of the dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Date</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Active</th>
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.93911</td>
      <td>67.709953</td>
      <td>2020-01-22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.15330</td>
      <td>20.168300</td>
      <td>2020-01-22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.03390</td>
      <td>1.659600</td>
      <td>2020-01-22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.50630</td>
      <td>1.521800</td>
      <td>2020-01-22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.20270</td>
      <td>17.873900</td>
      <td>2020-01-22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking the data
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 49068 entries, 0 to 49067
    Data columns (total 10 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Province/State  14664 non-null  object 
     1   Country/Region  49068 non-null  object 
     2   Lat             49068 non-null  float64
     3   Long            49068 non-null  float64
     4   Date            49068 non-null  object 
     5   Confirmed       49068 non-null  int64  
     6   Deaths          49068 non-null  int64  
     7   Recovered       49068 non-null  int64  
     8   Active          49068 non-null  int64  
     9   WHO Region      49068 non-null  object 
    dtypes: float64(2), int64(4), object(4)
    memory usage: 3.7+ MB
    


```python
#checking for missing values
data.isnull().sum()
```




    Province/State    34404
    Country/Region        0
    Lat                   0
    Long                  0
    Date                  0
    Confirmed             0
    Deaths                0
    Recovered             0
    Active                0
    WHO Region            0
    dtype: int64




```python
#checking for duplicates 
data.duplicated().sum()
```




    0



Exploratory Data Analysis (EDA)


```python
# checking for garbage Value
for i in data.select_dtypes(include="object").columns:
    print(data[i].value_counts())
    print("***"*10)
```

    Province/State
    Australian Capital Territory    188
    Yunnan                          188
    Mayotte                         188
    Guadeloupe                      188
    French Polynesia                188
                                   ... 
    Guizhou                         188
    Guangxi                         188
    Guangdong                       188
    Gansu                           188
    Saint Pierre and Miquelon       188
    Name: count, Length: 78, dtype: int64
    ******************************
    Country/Region
    China             6204
    Canada            2256
    France            2068
    United Kingdom    2068
    Australia         1504
                      ... 
    Holy See           188
    Honduras           188
    Hungary            188
    Iceland            188
    Lesotho            188
    Name: count, Length: 187, dtype: int64
    ******************************
    Date
    2020-01-22    261
    2020-05-30    261
    2020-05-21    261
    2020-05-22    261
    2020-05-23    261
                 ... 
    2020-03-26    261
    2020-03-27    261
    2020-03-28    261
    2020-03-29    261
    2020-07-27    261
    Name: count, Length: 188, dtype: int64
    ******************************
    WHO Region
    Europe                   15040
    Western Pacific          10340
    Africa                    9024
    Americas                  8648
    Eastern Mediterranean     4136
    South-East Asia           1880
    Name: count, dtype: int64
    ******************************
    


```python
#checking the statistics of the numercal data
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lat</th>
      <th>Long</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>49068.000000</td>
      <td>49068.000000</td>
      <td>4.906800e+04</td>
      <td>49068.000000</td>
      <td>4.906800e+04</td>
      <td>4.906800e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.433730</td>
      <td>23.528236</td>
      <td>1.688490e+04</td>
      <td>884.179160</td>
      <td>7.915713e+03</td>
      <td>8.085012e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24.950320</td>
      <td>70.442740</td>
      <td>1.273002e+05</td>
      <td>6313.584411</td>
      <td>5.480092e+04</td>
      <td>7.625890e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-51.796300</td>
      <td>-135.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>-1.400000e+01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.873054</td>
      <td>-15.310100</td>
      <td>4.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.634500</td>
      <td>21.745300</td>
      <td>1.680000e+02</td>
      <td>2.000000</td>
      <td>2.900000e+01</td>
      <td>2.600000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>41.204380</td>
      <td>80.771797</td>
      <td>1.518250e+03</td>
      <td>30.000000</td>
      <td>6.660000e+02</td>
      <td>6.060000e+02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>71.706900</td>
      <td>178.065000</td>
      <td>4.290259e+06</td>
      <td>148011.000000</td>
      <td>1.846641e+06</td>
      <td>2.816444e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking the statistics of non-numerical data
data.describe(include="object")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Date</th>
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14664</td>
      <td>49068</td>
      <td>49068</td>
      <td>49068</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>78</td>
      <td>187</td>
      <td>188</td>
      <td>6</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Australian Capital Territory</td>
      <td>China</td>
      <td>2020-01-22</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>188</td>
      <td>6204</td>
      <td>261</td>
      <td>15040</td>
    </tr>
  </tbody>
</table>
</div>




```python
#using scatter plot to check the relationship between data
for i in['Confirmed', 'Recovered', 'Active']:
    sns.scatterplot(data=data,x=i,y='Deaths')
    plt.show()
```


    
![png](output_11_0.png)
    



    
![png](output_11_1.png)
    



    
![png](output_11_2.png)
    



```python
#checking the correlation of the features using heatmap
heatmap=data.select_dtypes(include="number").corr()
sns.heatmap(heatmap,annot=True)
```




    <Axes: >




    
![png](output_12_1.png)
    


Model Development


```python
#machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Data Preprocessing
# Create a binary target variable: "High Risk" if deaths > 100, else "Low Risk"
data['Risk_Level'] = ['High Risk' if x > 100 else 'Low Risk' for x in data['Deaths']]

# Encode categorical variables
label_encoder = LabelEncoder()
data['WHO Region Encoded'] = label_encoder.fit_transform(data['WHO Region'])

# Select features and target variable
features = ['Confirmed', 'Recovered', 'Active', 'WHO Region Encoded']
target = 'Risk_Level'

# Split data into training and testing sets
X = data[features]
y = data[target]
y = label_encoder.fit_transform(y)  # Encode target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```

    Accuracy: 0.9781527675878373
                  precision    recall  f1-score   support
    
       High Risk       0.93      0.95      0.94      2111
        Low Risk       0.99      0.99      0.99     10156
    
        accuracy                           0.98     12267
       macro avg       0.96      0.97      0.96     12267
    weighted avg       0.98      0.98      0.98     12267
    
    


```python

```
# CapstonProject
