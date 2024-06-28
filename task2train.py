# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:56:45 2024

@author: Radhika
"""

import pandas as pd
import seaborn as sns


titanic = pd.read_csv(r'D:/Nikhil Analytics/Prodigy Infotech/train.csv')
titanic.head
titanic.shape 

# =============================================================================
# ANALYZING DATA
# =============================================================================

sns.countplot(x='Survived', data=titanic) #those who did not survived (more than 500) are greater than who servived (more than 300)

sns.countplot(x='Survived', hue='Sex', data=titanic , palette= 'winter')

# Analysis : 0 represent not servived and 1 is for servived 
# Female are thrice more likely to servive than males

sns.countplot(x='Survived', hue='Pclass', data=titanic , palette= 'PuBu')

# Analysis : the passengers who did not survived belong to 3rd class
# 1st class passenger are more likely to survived 

titanic['Age'].plot.hist()

# We noticed that highest age group travelling are among the young  age between 20-40
# very few passengers in age group 70-80

titanic['Fare'].plot.hist(bins=20, figsize=(10,5))

# We observed that most of the tickets bought are under fare 500
#  and very few on the higher side of fare i.e. 220-500 range

sns.countplot(x='SibSp', data=titanic, palette='rocket')

#  We noticed that most of the people do not have their siblings aboard

titanic['Parch'].plot.hist()

sns.countplot(x='Parch' , data=titanic, palette='summer')

#  the number of parents and siblings who aboard the ship are less

# =============================================================================
# DATA WRANGLING
# =============================================================================

# Data wwrangling means cleaning the data,removing the null values ,
# dropping unwanted columns, adding new ones if needed

titanic.isnull().sum()

# age and cabin has most null values. and embarked too has null values 

sns.heatmap(titanic.isnull(), cmap='spring')

# here yellow colour is showing the null values, highest in cabin followed by age

sns.boxplot(x='Pclass', y='Age', data=titanic)

# we can observe that older age group are travelling more in class 1 and 2
 # as compared to class 3

titanic.columns
titanic.drop('Cabin', axis=1, inplace=True)

titanic.head() # dropped the cabin column 

titanic.dropna(inplace=True)

sns.heatmap(titanic.isnull() , cbar=False)

# this shows that we dont have any null values. we can also check it

titanic.isnull().sum()

# drop column Age

titanic = titanic.drop(['Age'], axis = 1)


# =============================================================================
# Train Data
# =============================================================================

titanic.head()

# =============================================================================
# Logistic Regression
# =============================================================================

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

titanic['Sex']=labelencoder.fit_transform(titanic['Sex'])
titanic.head(2)
titanic.columns


sns.countplot( x = titanic['Sex'] , hue= titanic['Survived'])


X = titanic[['Pclass','Sex']]
Y = titanic['Survived']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2,random_state= 0)

from sklearn.linear_model import LogisticRegression

log =LogisticRegression(random_state=0)
log.fit(X_train,Y_train)

prediction = print(log.predict(X_test))

print(Y_test)

import warnings
warnings.filterwarnings("ignore")

res = log.predict([[2,0]])
if(res==0):
    print("so sorry! not survived")
else:
    print("survived")
 
#  output = SURVIVED
 
# =============================================================================
#     ------------------- END-------------------------
# =============================================================================
        
        
        
        
        
        
    










