import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# 1) Load the dataset
loans = pd.read_csv('loan_data.csv')

print('Shape of data:', loans.shape)
print('\nColumns:', loans.columns.tolist())
print('\nInfo:')
print(loans.info())
print('\nHead:')
print(loans.head())
print('\nDescribe:')
print(loans.describe())

# 2) Exploratory data analysis
plt.figure(figsize=(10, 6))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha=0.5, bins=30, label='credit.policy = 1')
loans[loans['credit.policy'] == 0]['fico'].hist(alpha=0.5, bins=30, label='credit.policy = 0')
plt.title('FICO Score Distribution by Credit Policy')
plt.xlabel('FICO')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, bins=30, label='not.fully.paid = 1')
loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, bins=30, label='not.fully.paid = 0')
plt.title('FICO Score Distribution by Loan Repayment Status')
plt.xlabel('FICO')
plt.ylabel('Frequency')
plt.legend()
plt.show()

sns.countplot(data=loans, x='purpose', hue='not.fully.paid')
plt.title('Loan Counts by Purpose')
plt.xticks(rotation=45)
plt.show()

sns.jointplot(x='fico', y='int.rate', data=loans, kind='scatter', color='navy')
plt.show()

sns.lmplot(data=loans, x='fico', y='int.rate', col='not.fully.paid', row='credit.policy', aspect=1.2)
plt.show()

# 3) Prepare the data for machine learning
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=101,
)

print('\nTrain set size:', X_train.shape)
print('Test set size:', X_test.shape)

# 4) Train a single decision tree
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

dt_predictions = DT.predict(X_test)
print('\nDecision Tree Classification Report:')
print(classification_report(y_test, dt_predictions))
print('Decision Tree Confusion Matrix:')
print(confusion_matrix(y_test, dt_predictions))

# 5) Train a random forest
RF = RandomForestClassifier(n_estimators=600)
RF.fit(X_train, y_train)

rf_predictions = RF.predict(X_test)
print('\nRandom Forest Classification Report:')
print(classification_report(y_test, rf_predictions))
print('Random Forest Confusion Matrix:')
print(confusion_matrix(y_test, rf_predictions))

print('\nRandom Forest often performs better than a single decision tree because it combines many trees.\n')

# 6) Compare the models
print('Decision Tree Accuracy is based on the report above.')
print('Random Forest Accuracy is based on the report above.')
print('The better model is the one with higher precision/recall balance and stronger classification report results.')
