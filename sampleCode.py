import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split

def dataload():
    df = pd.read_csv('/Users/Zhou/Desktop/cargo/test_v2.csv', delimiter=',')
    df['target']=df.A*10.0**7+df.B*10**6+df.C*10**5+df.D*10.0**4+df.E*10**3+df.F*10**2+df.G*10.0**1
    df2=df.loc[:,['shopping_pt', 'record_type', 'day','location', 'group_size', 'homeowner', 'car_age','age_oldest', 'age_youngest', 'married_couple','target']]
    df2.dtypes
    df2=df2.dropna(axis=0)#drop rows with missing values 
    return df2

def main():
    print('Split into 80% Train and 20% Test')
    df2=dataload()
    x = df2.values[:,:-1]
    y = df2.values[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =.2)
    return X_train,X_test,y_train,y_test

def treeCf(X_train,X_test,y_train,y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    acc=0
    for i in range(len(pred)):
        if pred[i]==y_test[i]:acc+=1
    error=(len(pred)-acc)*1.0/len(pred)
    print error 
    
if __name__=='__main__':
    X_train,X_test,y_train,y_test=main()
    treeCf(X_train,X_test,y_train,y_test)

