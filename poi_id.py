#!/usr/bin/python

import sys
import pickle
import random
import numpy 
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

#from poi_flag_email import poiFlagEmail, getToFromStrings

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

#features_list =['poi','total_payments','total_stock_value','salary', 'bonus',  'expenses'] 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


###  Explore Dataset -- Learning about the dataset/data file
#### 1) Length of data set
length = len(data_dict)
print "1) Length of data set: ", length

#### 2) Information of Key and Features

keys = data_dict.keys()
values = data_dict.values()
print "2) Features List: ", features_list
key_length = len(keys)
value_length = len(values)
print "   2.a) Keys Length: ", key_length
print "   2.b) Values Length: ", value_length
print "   2.c) Features Length: ", sum(len(v) for v in data_dict.itervalues())/key_length

#### 3) Number of POIs
poi_count = 0
for key, value in data_dict.items():
    if value["poi"] == 1:
        poi_count += 1

print "3) Number of POI: ", poi_count

#### 4) How many POIs were there total?

name_f = open("../final_project/poi_names.txt", "r")
count_name = 0
for line in name_f:
    if line.startswith('('):
        count_name += 1

print "4) Number of POIs in poi_names.txt: ", count_name

#### 5) Aany missing data value in the main features?

mi_salary = 0          #1
mi_deferral_payment = 0   #2
mi_total_payment = 0   #3
mi_loan_advances = 0   #4
mi_bonus = 0           #5
mi_restricted_stock_deferred = 0   #6
mi_deferred_income = 0       #7
mi_total_stock_value = 0     #8
mi_expense = 0               #9
mi_exercised_stock = 0       #10
mi_other = 0                 #11
mi_long_term_incent = 0      #12
mi_restricted_stock = 0      #13
mi_director_fees = 0         #14
mi_email_addr = 0 

for key, value in data_dict.items():
    if value['salary'] == 'NaN':
        mi_salary += 1
    if value['deferral_payments'] == 'NaN':
        mi_deferral_payment += 1
    if value['total_payments'] == 'NaN':
        mi_total_payment += 1
    if value['loan_advances'] == 'NaN':
        mi_loan_advances += 1
    if value['bonus'] == 'NaN':
        mi_bonus += 1
    if value['restricted_stock_deferred'] == 'NaN':
        mi_restricted_stock_deferred += 1
    if value['deferred_income'] == 'NaN':
        mi_deferred_income += 1
    if value['total_stock_value'] == 'NaN':
        mi_total_stock_value += 1
    if value['expenses'] == 'NaN':
        mi_expense += 1
    if value['exercised_stock_options'] == 'NaN':
        mi_exercised_stock += 1
    if value['other'] == 'NaN':
        mi_other += 1
    if value['long_term_incentive'] == 'NaN':
        mi_long_term_incent += 1
    if value['restricted_stock'] == 'NaN':
        mi_restricted_stock += 1
    if value['director_fees'] == 'NaN':
        mi_director_fees += 1
    if value['email_address'] == 'NaN':
        mi_email_addr += 1

print "5) Some major features that had missing value counts: "     
print "   5.1) Missing Salary Count: ", mi_salary
print "   5.2) Missing Deferral_Payments Count: ", mi_deferral_payment
print "   5.3) Missing Total_Payments Count: ", mi_total_payment
print "   5.4) Missing Loan_Advance Count: ", mi_loan_advances
print "   5.5) Missing Bonus Count: ", mi_bonus
print "   5.6) Missing Restricted_Stock_Deferred Count: ", mi_restricted_stock_deferred
print "   5.7) Missing Deferred_Income Count: ", mi_deferred_income
print "   5.8) Missing Total_Stock_Value Count: ", mi_total_stock_value
print "   5.9) Missing Expenses Count: ", mi_expense
print "   5.10) Missing Exercised_Stock_Options Count: ", mi_exercised_stock
print "   5.11) Missing Other Count: ", mi_other
print "   5.12) Missing Long_Term_Incentive Count: ", mi_long_term_incent
print "   5.13) Missing Restricted_Stock Count: ", mi_restricted_stock 
print "   5.14) Missing Director_Fees Count: ", mi_director_fees
print "   5.15) Missing Email_Address Count: ", mi_email_addr

#### 6. What's the value of total stock options exercised by some of top executives?
print "6) Jeff Skilling's Total Stock Options: ", data_dict['SKILLING JEFFREY K']['total_stock_value']
print "   Kenneth Lay's Total Stock Options: ", data_dict['LAY KENNETH L']['total_stock_value']
print "   Lou Pai's Total Stock Options: ", data_dict['PAI LOU L']['total_stock_value']
print "   Kenneth Rice's Total Stock Options: ", data_dict['RICE KENNETH D']['total_stock_value']
print "   Andrew Fastow's Total Stock Options: ", data_dict['FASTOW ANDREW S']['total_stock_value']

#### 7. Lay, Skilling and Fastow, how much money did they took home?  How much money was it?
print "7) Kenneth Lay's total payment: ", data_dict['LAY KENNETH L']['total_payments']
print "   Jeff Skilling's total payment: ", data_dict['SKILLING JEFFREY K']['total_payments']
print "   Andrew Fastow's total payment: ", data_dict['FASTOW ANDREW S']['total_payments']
print "   Lou Pai's total payment: ", data_dict['PAI LOU L']['total_payments']
print "   Kenneth Rice's total payment: ", data_dict['RICE KENNETH D']['total_payments']

#### 8. Lay, Skilling, Fastow, Pai, and Rice, how much did they got paid? 
print "8) Kenneth Lay's Salary: ", data_dict['LAY KENNETH L']['salary']
print "   Jeff Skilling's Salary: ", data_dict['SKILLING JEFFREY K']['salary']
print "   Andrew Fastow's Salary: ", data_dict['FASTOW ANDREW S']['salary']
print "   Lou Pai's Salary: ", data_dict['PAI LOU L']['salary']
print "   Kenneth Rice's Salary: ", data_dict['RICE KENNETH D']['salary']

#### 9.  How many folks in this dataset have a quantified salary? Known email address?
salary_count = 0
email_count = 0
for key, value in data_dict.items():
    if value["salary"] != 'NaN':
         salary_count = salary_count+1
    if value["email_address"] != 'NaN':
         email_count = email_count+1

print "9) Valid Salary count: ", salary_count
print "   Valid Email count: ", email_count

#### 10.  What percentage of people in the dataset have "NaN" for their total payments?

na_tp = 0
yes_tp = 0

for key, value in data_dict.items():
    if value["total_payments"] == 'NaN':
         na_tp = na_tp +1
    else:
         yes_tp = yes_tp + 1

np_percentage = (na_tp/(float)(na_tp+yes_tp)) *100
print "10) NaN Total number people: ", na_tp 
print "    NaN Percentage: ", np_percentage

#### 11. What percentage of POIs in the dataset have "NaN" for their total payments?

npoi_tp = 0

for key, value in data_dict.items():
    if (value["poi"]== 1 ) & (value["total_payments"] == 'NaN'):
         npoi_tp = npoi_tp +1

npoi_percentage = (npoi_tp/(float)( len(keys))) *100

print "11) POI - NaN Total number people: ", npoi_tp 
print "    POI NaN Percentage: ", npoi_percentage

#### 12. POIs vs Non-POIs ratio
npoi = 0
poi_npoi_ratio = 0.0
for key, value in data_dict.items():
    if value['poi'] != 1:
        npoi = npoi+1
poi_npoi_ratio = float(poi_count)/npoi

print "12) Number of POIs vs Non-POIs: ", poi_count, npoi
print "    POIs vs Non-POIs Ratio is: ", poi_npoi_ratio

#### 13.  How many emails were from/to Kenneth Lay and Jeffrey Skillings to POIs?
print "13) Email count for Kenneth Lay and Jeffrey Skilling: "
print "    From Kenneth Lay to POIs: ", data_dict['LAY KENNETH L']['from_this_person_to_poi']
print "    From POIs to Kenneth Lay: ", data_dict['LAY KENNETH L']['from_poi_to_this_person']
print "    From Jeffrey Skilling to POIs: ", data_dict['SKILLING JEFFREY K']['from_this_person_to_poi']
print "    From POIs to Jeffrey Skilling: ", data_dict['SKILLING JEFFREY K']['from_poi_to_this_person']
print 

#### 14. The Top 2 executive Information:
print "14) The top 2 executive Information: "
print "    *** Kenneth Lay ***", data_dict['LAY KENNETH L']
print "    *** Jeffrey Skilling ***", data_dict['SKILLING JEFFREY K']
print
print "********************************************"

features_list =['poi','total_payments','total_stock_value','salary']

### Task 2: Remove outliers

#### Regression Code 
#data = featureFormat(data_dict, features_list)
o_data = featureFormat( data_dict, features_list, remove_any_zeroes=True)

target, feature = targetFeatureSplit(o_data )

### training-testing split needed in regression, just like classification

from sklearn.cross_validation import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature, target, test_size=0.5, random_state=42)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, label_train)

slope = reg.coef_
print "slope: ", slope
intercept = reg.intercept_
print "intercept: ",  intercept

print "features: Total_Payments vs Total_Stock_Value"
test_score = reg.score(feature_test, label_test)
print "test score: ", test_score
train_score = reg.score(feature_train, label_train)
print "train score: ", train_score


#### Outlier Code
length_data = len (o_data)
print "*** length : ", length

key = data_dict.keys()
value = data_dict.values()
#print "(key) -- people's name", key
print "Length_keys: ", len(key)

d = {}
residual = []
error = 0.0 

for i in range(0, length_data):
     for key, value in data_dict.items():
         if (feature[i][0] == value["total_payments"]) and (feature[i][1] == value["total_stock_value"]):
             error = abs(feature[i][0]- feature[i][1])
             #print ">>> key, residual -- ", key, error, target[i]
             residual.append(error)
             d[key] = error

residual = sorted(residual, reverse=True)
#print "residual: ", residual
residual_0 = residual[0]
residual_1 = residual[1]
residual_2 = residual[2]
residual_3 = residual[3]
residual_4 = residual[4]
residual_5 = residual[5]

print "largest residual: " , residual_0

for key, value in d.iteritems():
    #print "<<< key, value >>> ", key, value
    if value == residual_0:
        print "[0] Name: ", key, value
        removed_key = key
    elif value == residual_1:
        print "[1] Name: ", key, value
    elif value == residual_2:
        print "[2] Name: ", key, value
    elif value == residual_3:
        print "[3] Name: ", key, value
    elif value == residual_4:
        print "[4] Name: ", key, value
    elif value == residual_5:
        print "[5] Name: ", key, value
    else: 
        continue

### Any more outliers?
for key, value in data_dict.items():
    if key == removed_key:
        data_dict.pop(key, 0)

data = featureFormat(data_dict, features_list)

### graphing -Total Payment vs Total Stock Value

for point in data:
    #print "+++ point: ", point
    t_payment = point[1]
    t_stock_value = point[2]
    matplotlib.pyplot.scatter( t_payment, t_stock_value )

matplotlib.pyplot.xlabel("Total Payments")
matplotlib.pyplot.ylabel("Total Stock Value")
matplotlib.pyplot.show()

### graphing - Email from POI vs Email to POI

features_list =['poi','total_payments','total_stock_value','salary', 'bonus', 'expenses', 
	'restricted_stock', 'exercised_stock_options', 'other', 'long_term_incentive', 'deferred_income', 'deferral_payments', 
	'restricted_stock_deferred', 'director_fees', 'loan_advances', 'from_poi_to_this_person', 'from_this_person_to_poi', 
	'from_messages', 'to_messages', 'shared_receipt_with_poi']

### Extract features and labels from dataset for local testing
print "Features_List: ", features_list
data = featureFormat(data_dict, features_list)

for point in data:
    #print "+++ point: ", point
    from_poi = point[15]
    to_poi = point[16]
    matplotlib.pyplot.scatter(from_poi, to_poi )

matplotlib.pyplot.xlabel("# of Emails from POIs to this person")
matplotlib.pyplot.ylabel("# of Emails from the person send to POI")
matplotlib.pyplot.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#print "My_dataset -- ", my_dataset

def calculatefraction (poi_mesg, all_mesg):
    if poi_mesg == "NaN" or all_mesg == "NaN":
        fraction = 0.0
    else:
        fraction = float(poi_mesg)/(float(all_mesg))
    return fraction

from_poi_to_receiver = 0
from_sender_to_poi = 0
sub_dict = {}
for name in my_dataset:
    data_point = my_dataset[name]
    #print 
    #print "Name --  ", name  

    from_poi_to_receiver = data_point["from_poi_to_this_person"]
    to_mesg = data_point["to_messages"]
    fraction_from_poi = calculatefraction(from_poi_to_receiver, to_mesg)
    #print "Fraction from POI: ", fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    from_sender_to_poi = data_point["from_this_person_to_poi"]
    from_mesg = data_point["from_messages"]
    fraction_to_poi = calculatefraction(from_sender_to_poi, from_mesg)
    #print "Fraction to POI: ", fraction_to_poi
    sub_dict[name] = {"from_poi_to_this_person": fraction_from_poi, "from_this_person_to_poi": fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
    my_dataset[name] = data_point

#print "***My_dataset -- ", my_dataset


features_list =['poi','total_payments','total_stock_value','salary', 'bonus', 'expenses', 'fraction_from_poi', 'fraction_to_poi', 
	'restricted_stock', 'exercised_stock_options', 'other', 'long_term_incentive', 'deferred_income', 'deferral_payments', 
	'restricted_stock_deferred', 'director_fees', 'loan_advances', 'from_poi_to_this_person', 'from_this_person_to_poi', 
	'from_messages', 'to_messages', 'shared_receipt_with_poi']

### Extract features and labels from dataset for local testing
print "Features_List: ", features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### graphing # 2 

for point in data:
    #print "+++ point: ", point
    fraction_from_poi = point[6]
    fraction_to_poi = point[7]
    matplotlib.pyplot.scatter(fraction_from_poi, fraction_to_poi )

matplotlib.pyplot.xlabel("Fraction of From_Poi_To_Receiver")
matplotlib.pyplot.ylabel("Fraction of From_Sender_To_Poi")
matplotlib.pyplot.show()

##cross_validation.train_test_split(features, labels)

#### Univariate feature selection
#data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation

X, y = features, labels

selector = SelectKBest(f_classif, k=12).fit(X, y)
#print "Selector --- ", selector.scores_ , selector.pvalues_ 

print
best_features = numpy.where(selector.get_support())[0]
print "Best Feature : ", best_features
print "len(best_features) -- ", len(best_features)
for i in range(0, len(best_features)):
    print "+++ ", features_list[best_features[i]]

selector = selector.fit_transform(X, y)
#print "2-- Selector -- ", selectror

##features_list =['poi','total_payments','salary','restricted_stock']   # K=3 (3 high)
##features_list =['poi','total_payments','salary','restricted_stock','total_stock_value'] # K=4 (top 4 high)
##features_list =['poi','total_payments','salary','restricted_stock','total_stock_value','fraction_from_poi']   # K=5 (top 5 high)
##features_list =['poi','total_payments','salary','restricted_stock','total_stock_value','fraction_from_poi','long_term_incentive']   # K=6 (top 6 high)
##features_list =['poi','total_payments','salary','restricted_stock','total_stock_value','fraction_from_poi','long_term_incentive','other']   # K=7 (top 7 high)
##features_list =['poi','total_payments','salary','restricted_stock','total_stock_value','fraction_from_poi','long_term_incentive','other','fraction_to_poi']   # K=8 (top 8 high)
##features_list =['poi','total_payments','total_stock_value','salary','fraction_from_poi','fraction_to_poi','restricted_stock','other','long_term_incentive']   # K=9 (top 9 high)
##features_list =['poi','total_payments','total_stock_value','salary','fraction_from_poi','fraction_to_poi',
##	'restricted_stock','other','long_term_incentive', 'to_messages']   # K=10 (top 10 high)
##features_list =['poi','total_payments','total_stock_value','salary','fraction_from_poi','fraction_to_poi',
##	'restricted_stock','other','long_term_incentive', 'director_fees','to_messages']   # K=11 (top 11 high)
##features_list =['poi','total_payments','total_stock_value','salary','bonus','fraction_from_poi',
##	'fraction_to_poi','restricted_stock','other','long_term_incentive', 'director_fees','loan_advances','to_messages']   # K=13 (top 13 high)
##features_list =['poi','total_payments','total_stock_value','salary','bonus','fraction_from_poi','fraction_to_poi',
##	'restricted_stock','exercised_stock_options','other','long_term_incentive', 'director_fees','loan_advances','to_messages']   # K=14 (top 14 high)

#features_list = ['poi', 'expenses', 'exercised_stock_options', 'total_payments', 'restricted_stock', 'total_stock_value']
#features_list =['poi','total_payments','total_stock_value','salary','bonus','expenses'] 

#features_list =['poi','total_payments','total_stock_value','salary','bonus','restricted_stock','other','long_term_incentive', 'director_fees','to_messages']   # K=12 (top 12 high)

features_list =['poi','total_payments','total_stock_value','salary','bonus','fraction_from_poi','fraction_to_poi',
	'restricted_stock','other','long_term_incentive', 'director_fees','to_messages']   # K=12 (top 12 high)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)

### Task 4.1: Try a varity of classifiers  (Random Forests)
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#### Try Random Forests
print "*** Random Forests ***" 
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf = clf.fit(features, labels)
print "Random Forest - importance :", clf.feature_importances_ 
print


### Task 4.2: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import grid_search

t_prec = 0.0
t_reca = 0.0
count = 0.0

#clf = RandomForestClassifier(min_samples_split=5, min_samples_leaf=1,max_depth=10,max_features=5, n_estimators=10)
#clf = clf.best_estimator_

sss = StratifiedShuffleSplit(y, 10, test_size=0.5, random_state=42)
"""
parameters = {'min_samples_split': [2,3,4,5,10,15,20],
              'min_samples_leaf': [1,2,3,4,5],
              'max_depth': [None, 5, 10, 15,25],
              'max_features':[None, 2,3,4,5,6,7],
              'n_estimators': [10, 15, 20, 25,40]}

clf = grid_search.GridSearchCV(clf, parameters, cv=sss)

print "--- Random Forest  after grid_search (feature_train, label_train)---"
#print "--- Random Forest  - grid_scores_ ", clf.grid_scores_
print "--- Random Forest  - best_estimator_ ", clf.best_estimator_
print "--- Random Forest  - best_scores_ ", clf.best_score_
print "--- Random Forest  - best_params_ ", clf.best_params_
print "--- Random Forest  - scorer_ ", clf.scorer_
print "--- Random Forest  - train_score ", clf.score(features, labels)
print
"""

clf = RandomForestClassifier(min_samples_split=5, min_samples_leaf=1,max_depth=10,max_features=5, n_estimators=10)
clf.fit(features, labels)

for train_index, test_index in sss:
#    print("TRAIN:", train_index, "TEST:", test_index)
    feature_train = [X[j] for j in train_index]   
    feature_test = [X[j] for j in train_index]
    label_train = [y[j] for j in test_index]
    label_test = [y[j] for j in test_index]

    ## Same highest mean grid_scores_ 
    #clf = clf.best_estimator_
    #clf = clf.transform(clf)
    #clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1,max_depth=None,max_features=2, n_estimators=25)
    #clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=4,max_depth=5,max_features=None, n_estimators=40)
    #clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1,max_depth=10,max_features=3, n_estimators=10)
    #clf = RandomForestClassifier(min_samples_split=4, min_samples_leaf=1,max_depth=10,max_features=6,n_estimators=20)
    #clf = RandomForestClassifier(min_samples_split=15, min_samples_leaf=4,max_depth=25,max_features=6,n_estimators=15)

    clf = clf.fit(feature_train, label_train)

    pred = clf.predict(feature_test)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, label_test)
    print "accuracy -- (Random Forest)", acc
    print

    from sklearn import metrics
    y_true = numpy.array(feature_test)
    y_score = numpy.array(label_test)
    prec_sc = metrics.precision_score(pred, label_test, average="binary")
    recall_sc = metrics.recall_score(pred, label_test, average="binary")
    print "==>Precision_Score, Recall_Score: ", prec_sc, recall_sc
    print
    t_prec = t_prec + prec_sc
    t_reca = t_reca + recall_sc
    count = count + 1.0

avg_p = t_prec/count
avg_r = t_reca/count
print "Average Precision Score: ", avg_p
print "Average Recall Score: ", avg_r
print

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


### Task 5.1: Try a varity of classifiers  (Decision Tree)
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#### Try Decision Tree
print "*** Decision Tree ***" 

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print "Decision Tree - importance :", clf.feature_importances_ 


##Select features from feature_importances_
#features_list = ['poi','total_payments','bonus','fraction_from_poi','fraction_to_poi', 'restricted_stock', 'exercised_stock_options','shared_receipt_with_poi'] #(origin)
#features_list = ['poi','bonus','fraction_from_poi','fraction_to_poi','restricted_stock','exercised_stock_options','shared_receipt_with_poi']
#features_list = ['poi','total_payments','bonus','restricted_stock','exercised_stock_options','shared_receipt_with_poi']

#features_list = ['poi','salary','bonus','fraction_from_poi','fraction_to_poi', 'restricted_stock', 'exercised_stock_options','shared_receipt_with_poi'] #(origin)
#features_list = ['poi','total_stock_value','bonus','fraction_from_poi','fraction_to_poi', 'restricted_stock', 'exercised_stock_options','shared_receipt_with_poi'] #(origin)
#features_list = ['poi','total_payments','salary','bonus','fraction_from_poi','fraction_to_poi', 'restricted_stock', 'exercised_stock_options','other','shared_receipt_with_poi'] #(origin)
features_list = ['poi','total_payments','bonus','fraction_from_poi','fraction_to_poi', 'restricted_stock', 'exercised_stock_options','shared_receipt_with_poi'] #(origin)
#features_list = ['poi','fraction_from_poi','fraction_to_poi', 'restricted_stock', 'exercised_stock_options','shared_receipt_with_poi'] #(origin)

my_feature_list = features_list
print "Feature_list - ", features_list
print

### Task 5.2: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Testing train_test_split (for GridSearchCV)
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit

##feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.15, random_state=42)
parameters = {'min_samples_split': [2,3,4,5,10],
              'min_samples_leaf': [1,2,3,4,5],
              'max_features': [None,2,3,4,5,6,7],
              'max_depth': [None, 5, 10, 15,20,25],}
#clf = grid_search.GridSearchCV(clf, parameters, cv=20)

#clf = clf.fit(features, labels)

sss = StratifiedShuffleSplit(y, 10, test_size=0.5, random_state=42)
clf = grid_search.GridSearchCV(clf, param_grid=parameters, cv=sss)
clf.fit(features, labels)
#print "clf -----", clf

print
print "--- Decision Tree after grid_search (features, labels)---"
#print "--- Decision Tree - grid_scores_ ", clf.grid_scores_
print "--- Decision Tree - best_estimator_ ", clf.best_estimator_
print "--- Decision Tree - best_scores_ ", clf.best_score_
print "--- Decision Tree - best_params_ ", clf.best_params_
print "--- Decision Tree - scorer_ ", clf.scorer_
print "--- Decision Tree - train_score ", clf.score(features, labels)
print

count = 0.0
mean_sum = 0.0
for params, mean_score, scores in clf.grid_scores_:
     count = count + 1.0
     mean_sum = mean_sum + mean_score

avg_mean = mean_sum/count
print "Average mean_score: ", avg_mean

clf = clf.best_estimator_
t_prec = 0.0
t_reca = 0.0
count = 0.0

for train_index, test_index in sss:
#    print("TRAIN:", train_index, "TEST:", test_index)
    feature_train = [X[j] for j in train_index]   
    feature_test = [X[j] for j in train_index]
    label_train = [y[j] for j in test_index]
    label_test = [y[j] for j in test_index]

    print "Length - train_set, feature, label: ", len(X), len(feature_train), len(label_train)
    print "Length - test_set, feature, label: ", len(y), len(feature_test), len(label_test)


    clf = tree.DecisionTreeClassifier(min_samples_split=3, min_samples_leaf=3, max_depth=25, max_features=5)

    #clf = tree.DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=2, max_depth=10, max_features=3)
    #clf = tree.DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=2, max_depth=10, max_features=6)

    clf = clf.fit(feature_train, label_train)
    print
    print "Score: ", clf.score(feature_test, label_test)
    print

    pred = clf.predict(feature_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, label_test)
    print "accuracy -- (Decision Tree)", acc

    from sklearn import metrics
    y_true = numpy.array(feature_test)
    y_score = numpy.array(label_test)
    prec_sc = metrics.precision_score(pred, label_test, average='binary')
    recall_sc = metrics.recall_score(pred, label_test, average='binary')

    t_prec = t_prec + prec_sc
    t_reca = t_reca + recall_sc
    count = count + 1.0

    print "==>Precision_Score, Recall_Score: ", prec_sc, recall_sc
    print


avg_p = t_prec/count
avg_r = t_reca/count
print "Average Precision Score: ", avg_p
print "Average Recall Score: ", avg_r
print

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
