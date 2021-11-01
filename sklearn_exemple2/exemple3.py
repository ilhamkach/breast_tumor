from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


dataset = load_breast_cancer()
X= dataset.data
y=dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)
rf= RandomForestClassifier()
rf.fit(X_train,y_train)
lr= LogisticRegression()
lr.fit(X_train,y_train)
svc_w_linear_kernel = SVC(kernel="linear")
svc_w_linear_kernel.fit(X_train,y_train)
svc_wo_linear_kernel=SVC()
svc_wo_linear_kernel.fit(X_train,y_train)
dummy = DummyClassifier()
dummy.fit(X_train,y_train)

print('Random Forest Classifier accuracy:',rf.score(X_test,y_test))
print('Logistic Regession accuracy:',lr.score(X_test,y_test))
print('SVC w linear kernel accuracy :',svc_w_linear_kernel.score(X_test,y_test))
print('Dummy accuracy:',dummy.score(X_test,y_test))
print('SVC wo_linear_kernel:',svc_wo_linear_kernel.score(X_test,y_test))