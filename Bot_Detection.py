
from afinn import Afinn
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np

tweet_df = pd.read_csv('data_train.csv')
tweet_df
tweet_df.columns
tweet_df.drop(['Id','following','followers','actions','is_retweet','location'], axis=1, inplace=True)
tweet_df['Type']=tweet_df['Type'].map({'Quality':0, 'Bot':1})
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = tweet_df['Tweet']
y = tweet_df['Type']

x = cv.fit_transform(x)

print(tweet_df['Tweet'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.shape

models = []
print("KNeighborsClassifier")

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3, weights='distance')
model.fit(x_train, y_train)
predict_knn = model.predict(x_test)
knn_acc = accuracy_score(y_test, predict_knn) * 100
print(knn_acc)
print("CLASSIFICATION REPORT")
print(classification_report(y_test, predict_knn))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, predict_knn))
models.append(('KNeighborsClassifier', model))

#print(model.score(x_test, y_test))

# SVM Model
print("SVM")
from sklearn import svm
lin_clf = svm.LinearSVC()
lin_clf.fit(x_train, y_train)
predict_svm = lin_clf.predict(x_test)
svm_acc = accuracy_score(y_test, predict_svm) * 100
print(svm_acc)
print("CLASSIFICATION REPORT")
print(classification_report(y_test, predict_svm))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, predict_svm))
models.append(('svm', lin_clf))


print("Logistic Regression")

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=0, solver='lbfgs').fit(x_train, y_train)
y_pred = reg.predict(x_test)
print("ACCURACY")
print(accuracy_score(y_test, y_pred) * 100)
print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
models.append(('logistic', reg))


print("Naive Bayes")

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(x_train, y_train)
predict_nb = NB.predict(x_test)
naivebayes = accuracy_score(y_test, predict_nb) * 100
print(naivebayes)
print(confusion_matrix(y_test,predict_nb))
print(classification_report(y_test, predict_nb))

models.append(('naive_bayes', NB))

classifier= VotingClassifier(models)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


tweet ="EBMUD ending penalties for excessive water users https://t.co/D5a1FMVMHd"
tweet_data = [tweet]
vector1 = cv.transform(tweet_data).toarray()
predict_text = classifier.predict(vector1)
if predict_text == 1:
 val = 'Bot'
else:
 val = 'Quality'

