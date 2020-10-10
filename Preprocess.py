import pandas as pd
import sys
import recordlinkage
from recordlinkage import preprocessing
from recordlinkage.standardise import clean
from sklearn import svm, model_selection, preprocessing
from recordlinkage.index import Full
import sklearn
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def cleaning():
    #this is to read the two data files that we have and to see if we need to perform any type of cleaning on this data
     df1=pd.read_csv('bikedekho.csv')
     print(df1.shape)
     print(df1.head())
     print(df1.dtypes)
     print(df1['bike_name'].nunique())
     df1['bike_name']=clean(df1['bike_name'])

     df2=pd.read_csv('bikewale.csv')
     print(df2.shape)
     print(df2.head())
     print(df2.dtypes)
     print(df2['bike_name'].nunique())
     df2['bike_name']=clean(df2['bike_name'])




def labeldata():
    #in here we are dividing the labelled data into their constitent files and then to apply blocking to see how manny truw
    #matches do we actually recover and which attributes can be used for blocking so that we dont lose the true matches
    #in the process
    df1=pd.read_csv('labeled_data.csv',skiprows=5,usecols=[1,2,13])
    print(df1['ltable'].nunique())
    print(df1['rtable'].nunique())
    df1=df1[df1['gold']==1]
    print(df1.shape)
    print(df1.head())
    print(df1.columns)
    df2=pd.read_csv('bikedekho.csv')
    df3=pd.read_csv('bikewale.csv')
    df4=pd.merge(df1,df2,left_on=['ltable.id'],right_on=['id'])
    print(df4.shape)
    print(df4.head())
    print(df4.columns)
    #taking only the true matches and putting into separate file
    #df4.to_csv('bikedekho_lable.csv',index=False)
    df5=pd.merge(df1,df3,left_on=['rtable.id'],right_on=['id'])
    print(df5.shape)
    print(df5.head())
    print(df5.columns)
    #taking only true match id's from the second file and putting it into a separate file so that we can then apply full indexing
    #and apply blocking to get the matches as well as non matches from these filea to trrain our models
    #even though the original label data file has non-matches as well but we are not using it as we can generate our own non matches d
    #for this project and we also wanted to analyse how the blocking is working
    #df5.to_csv('bikewale_lable.csv',index=False)

def blocking():
    df1=pd.read_csv('bikedekho_lable.csv',usecols=[2,3,4,5,6,7,8,9,10,11,12])
    df1.set_index('id', inplace=True)
    df1 = df1[~df1.index.duplicated(keep='first')]
    df1['bike_name']=clean(df1['bike_name'])

    print(df1.shape)
    print(df1.columns)
    print(df1.dtypes)
    print(df1.head())
    df2=pd.read_csv('bikewale_lable.csv',usecols=[2,3,4,5,6,7,8,9,10,11,12])
    df2.set_index('id', inplace=True)
    df2 = df2[~df2.index.duplicated(keep='first')]
    df2['bike_name']=clean(df2['bike_name'])
    print(df2.columns)
    print(df2.shape)
    indexer=recordlinkage.index.Full()
    pcl=indexer.index(df1,df2)

    print(pcl.shape)
    #We can also divide the pairs into chunks if we have some sort of memory issue, so that the processing will be done without any error

    x = recordlinkage.index_split(pcl, 1000)
    for chunk in x:
        #for each chunk of 1000 we compare the attributes to compute the similarity between them.
        compare_cl = recordlinkage.Compare()
        #to compare the string type we are using the jarowinkler similarity, others can also be used though

        compare_cl.string('bike_name', 'bike_name', method='jarowinkler', label='bikenameJW')
        compare_cl.string('city_posted', 'city_posted', method='jarowinkler', label='citypostedJW')
        compare_cl.string('color', 'color', method='jarowinkler', label='colorJW')
        compare_cl.string('fuel_type', 'fuel_type', method='jarowinkler', label='fueltypeJW')
        #to compare  the integer type we are using the exact comparison where the function returns the value 1 if the values
        #are totally similar otherwise it returns the value 0.
        compare_cl.exact('km_driven', 'km_driven', label='km_driven')
        compare_cl.exact('price', 'price', label='price')
        compare_cl.exact('model_year', 'model_year', label='model_year')
        features = compare_cl.compute(chunk, df1, df2)
        # the feature vectors computed for these are written into this file.
        with open('labelFeature.csv', 'a') as f:
            features.to_csv(f, header=False)
        #until here we have the feature vectors of the candidate pairs that we derived by applying the full index over
        #the true pairs that we got, 15621 is the total number of pairs that we got.
        #next step is to look into this file that we have and then label the true matches as 1 and non matches as 0 to

def preparetrain():
    df1=pd.read_csv('labelFeature.csv')
    df1.columns=['BDID','BWID','bikenameJW','citypostedJW','colorJW','fueltypeJW','km_driven','price','model_year']
    df2=pd.read_csv('labeled_data.csv',skiprows=5,usecols=[1,2,13])
    df2=df2[df2['gold']==1]
    print(df2.head())
    print(df2.shape)
    df3=pd.merge(df1,df2,left_on=['BDID','BWID'],right_on=['ltable.id' , 'rtable.id' ])
    print(df3.shape)
    print(df3.head())
    print(df3.columns)
    #in order to write the selected columns into the csv file first just modify the dataframe to include only those columns which
    #you want to write into the file like below
    df3 = df3[['BDID', 'BWID','bikenameJW','citypostedJW', 'colorJW', 'fueltypeJW',
       'km_driven', 'price', 'model_year','gold']]
    print(df3.shape)
    #df3.to_csv('matchpairs.csv',index=False)
    #in here we merged the feature vectors with the true matches ID from the original file to extractv the feature vectors of those
    #of true matches into a separate file with their label as well.
    #next thing to doo is to assign the labels to the feature vectors of non matches and then cncatenate the matches and
    #nonmatches into the same file and our training data is ready !!!!

def preparetrain2():
    #always assign  the column names to labelFeature.csv as it does not contain the headers
    df1=pd.read_csv('labelFeature.csv')
    df1.columns = ['BDID', 'BWID', 'bikenameJW', 'citypostedJW', 'colorJW', 'fueltypeJW', 'km_driven', 'price',
                   'model_year']
    print(df1.head())
    df2=pd.read_csv('matchpairs.csv')
    print(df2.head())
    #to get the rows which are not in the second file, merge using the left only with the indicator column, then get the
    #rows which contains the (left_only )value in the indicator column which will give you the rows which are not in another dataframe
    df1=pd.merge(df1,df2,how='left',indicator=True)
    df1=df1[df1['_merge']=='left_only']
    print(df1.shape)
    df1=df1[['BDID', 'BWID', 'bikenameJW', 'citypostedJW', 'colorJW', 'fueltypeJW',
       'km_driven', 'price', 'model_year','gold']]
    #Assign the value o to the label column for all the non matches
    df1['gold']=0
    print(df1.head(5))
    #write non match pairs into a separate file
    #df1.to_csv('nonmatchpairs.csv',index=False)
    #this is to concatenate the match and non match pair files to write it into a single one
    def concat():
         df1=pd.read_csv('matchpairs.csv')
         df2=pd.read_csv('nonmatchpairs.csv')
         df3=pd.concat([df1,df2])
         print(df3.shape)
         df3.to_csv('labelFeature.csv',index=False)

#now that our training data is ready, lets train our models. We can do one thing

def randomforest():
    df1=pd.read_csv('labelFeature.csv')
    X=df1.drop('gold',axis=1)

    y=df1['gold']
    #dividing our train data into train, validation and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    #X_train = preprocessing.scale(X_train)
    #y_train = preprocessing.scale(y_train)
    #lab_enc = preprocessing.LabelEncoder()
    #y_train = lab_enc.fit_transform(y_train)
    clf=RandomForestClassifier(n_estimators=100,max_features='sqrt',max_depth=200,min_samples_leaf=2,min_samples_split=6,bootstrap=False)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    svclassifier = SVC(C=1, gamma=0.001, kernel='rbf')
    svclassifier.fit(X_train,y_train)
    y_pred = svclassifier.predict(X_test)
    print(len(y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
#since I only use the true matches to generate the non matchces from them by applying the full index over it, we cannot find
#the same ones in the labeled data file that was already provided by them. So if I am looking to reduce the number of non
#matches so that my system works, I cannot use the one given in the labeled data instead I can reduce the number of
#non-matches that I have.


def svm():
    df1 = pd.read_csv('labelFeature.csv')
    X = df1.drop('gold', axis=1)

    y = df1['gold']
    # dividing our train data into train, validation and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    # X_train = preprocessing.scale(X_train)
    # y_train = preprocessing.scale(y_train)
    # lab_enc = preprocessing.LabelEncoder()
    # y_train = lab_enc.fit_transform(y_train)

