
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np

# Create your views here.
from Remote_User.models import ClientRegister_Model,Tweet_Message,Tweet_Type_Prediction,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_Tweet_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Bot'
    print(kword)
    obj = Tweet_Type_Prediction.objects.all().filter(Q(tweet_type=kword))
    obj1 = Tweet_Type_Prediction.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Quality'
    print(kword1)
    obj1 = Tweet_Type_Prediction.objects.all().filter(Q(tweet_type=kword1))
    obj11 = Tweet_Type_Prediction.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/Find_Tweet_Type_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Tweet_Type_Prediction.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Predict_Tweet_Type(request):
    obj =Tweet_Type_Prediction.objects.all()
    return render(request, 'SProvider/Predict_Tweet_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Tweet_Type_Prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.idno, font_style)
        ws.write(row_num, 1, my_row.Tweet, font_style)
        ws.write(row_num, 2, my_row.following, font_style)
        ws.write(row_num, 3, my_row.followers, font_style)
        ws.write(row_num, 4, my_row.actions, font_style)
        ws.write(row_num, 5, my_row.is_retweet, font_style)
        ws.write(row_num, 6, my_row.location, font_style)
        ws.write(row_num, 7, my_row.tweet_type, font_style)

    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()
    tweet_df = pd.read_csv('data_train.csv')
    tweet_df
    tweet_df.columns
    tweet_df.drop(['Id', 'following', 'followers', 'actions', 'is_retweet', 'location'], axis=1, inplace=True)
    tweet_df['Type'] = tweet_df['Type'].map({'Quality': 0, 'Bot': 1})
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

    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=knn_acc)

    # print(model.score(x_test, y_test))

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

    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

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

    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB
    NB = MultinomialNB()
    NB.fit(x_train, y_train)
    predict_nb = NB.predict(x_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print(naivebayes)
    print(confusion_matrix(y_test, predict_nb))
    print(classification_report(y_test, predict_nb))
    models.append(('naive_bayes', NB))
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

    classifier = VotingClassifier(models)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)




    obj1 = Tweet_Message.objects.values('idno',
    'Tweet',
    'following',
    'followers',
    'actions',
    'is_retweet',
    'location')
    Tweet_Type_Prediction.objects.all().delete()
    for t in obj1:

        idno= t['idno']
        Tweet= t['Tweet']
        following= t['following']
        followers= t['followers']
        actions= t['actions']
        is_retweet= t['is_retweet']
        location= t['location']

        tweet_data = [Tweet]
        vector1 = cv.transform(tweet_data).toarray()
        predict_text = classifier.predict(vector1)
        if predict_text == 1:
            val = 'Bot'
        else:
            val = 'Quality'


        Tweet_Type_Prediction.objects.create(
        idno=idno,
        Tweet=Tweet,
        following=following,
        followers=followers,
        actions=actions,
        is_retweet=is_retweet,
        location=location,
        tweet_type=val)

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})