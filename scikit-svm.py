from sklearn import svm
from clsdatareader import get_data

train, train_label, test, test_label = get_data()
clf = svm.SVC()
clf.fit(train, train_label)

hit = 0
for idx, test_data in enumerate(test):
    if clf.predict([test_data]) == test_label[idx]:
        hit += 1
print(hit/len(test), hit, len(test))
