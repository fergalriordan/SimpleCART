import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load folds
folds = []
for i in range(1, 6):
    with open(f'fake_fold_{i}.json') as f:
        fold_data = []
        for line in f:
            obj = json.loads(line)
            fold_data.append(obj)
        folds.append(pd.DataFrame(fold_data))

# Initialize vectorizer and classifier
vectorizer = CountVectorizer()
clf = DecisionTreeClassifier()

# Initialize lists to store results
accuracy_list = []
report_list = []

# Perform 5-fold cross-validation
for i in range(5):
    # Use ith fold as test set, remaining folds as training set
    test_data = folds[i]
    train_data = pd.concat([folds[j] for j in range(5) if j != i], ignore_index=True)
    
    X_train_vect = vectorizer.fit_transform(train_data['text'])
    y_train = train_data['is_deceptive']
    X_test_vect = vectorizer.transform(test_data['text'])
    y_test = test_data['is_deceptive']
    
    clf.fit(X_train_vect, y_train)
    y_pred = clf.predict(X_test_vect)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    report_list.append(report)

# Classification report
avg_report = {}
for key in report_list[0].keys():
    if isinstance(report_list[0][key], float):
        avg_report[key] = np.mean([report[key] for report in report_list])
    else:
        avg_report[key] = {}
        for metric in report_list[0][key].keys():
            avg_report[key][metric] = np.mean([report[key][metric] for report in report_list])

print("Classification Report: ")
print(avg_report)