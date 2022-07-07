import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump, load

train = pd.read_csv('gs://gcp-ml-udemy/titanic_input_dataset/titanic_train.csv')
# train = pd.read_csv(r"gs://vertex-ai-custom/CrabAgePrediction.csv")

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# predictions = logmodel.predict(X_test)
# classification_metrics = classification_report(y_test,predictions,output_dict=True)

# df = pd.DataFrame(classification_metrics).transpose()
dump(logmodel, 'gs://gcp-ml-udemy/model_output/logistic_regression/logistics_model.joblib')
# clf = load('logistics_model.joblib')
# predicted_values = clf.predict(X_test)


 