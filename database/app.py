import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# # Load data
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

df=pd.read_csv(r"C:\Users\om\Desktop\DAP\iris.csv")

x=df.iloc[:,:-1]
y=df['species']


os.makedirs("model_versions", exist_ok=True)

y1=df['species'].map({'setosa':0,'virginica':1,'versicolor':2})

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Try different regularization strengths
results = []
for i, k in enumerate([1, 3, 5], start=1):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results.append({'version': f'v{i}', 'k': k, 'accuracy': acc})
    joblib.dump(model, f'model_versions/model_v{i}.pkl')
    print(f"Saved model_v{i}.pkl with k={k}, accuracy: {acc:.2f}")


df=pd.DataFrame(results)
print(df)
plt.plot(df['k'],df['accuracy'])


loaded_model = joblib.load("model_versions/model_v1.pkl")
accuracy = loaded_model.score(X_test, y_test)
print(f"Loaded model accuracy: {accuracy:.2f}")




































# IRIS Dataset - EDA + Binary Classification Pipeline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
df = pd.concat([X, y], axis=1)

print("\nChecking for null values:\n", df.isnull().sum())

plt.figure(figsize=(10, 6))
sn.boxplot(data=df.iloc[:, :-1])
plt.title("Boxplots of Features")
plt.savefig("boxplots.png")
plt.show()

sn.pairplot(df, hue='target')
plt.savefig("pairplot.png")
plt.show()

plt.figure(figsize=(8, 6))
sn.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("correlation_heatmap.png")
plt.show()

X.hist(figsize=(10, 6), bins=15)
plt.suptitle("Feature Distributions")
plt.savefig("feature_distributions.png")
plt.show()

try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df, title="Iris Dataset EDA", explorative=True)
    profile.to_file("iris_eda_report.pdf")
except ImportError:
    print("\nInstall ydata-profiling to generate a full PDF report.")

y_bin = (y == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

ConfusionMatrixDisplay.from_estimator(log_model, X_test, y_test)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("log_confusion_matrix.png")
plt.show()

ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test)
plt.title("Confusion Matrix - KNN")
plt.savefig("knn_confusion_matrix.png")
plt.show()

y_score_log = log_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_score_log)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title("Precision-Recall Curve - Logistic Regression")
plt.savefig("log_precision_recall.png")
plt.show()

y_score_knn = knn_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_score_knn)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title("Precision-Recall Curve - KNN")
plt.savefig("knn_precision_recall.png")
plt.show()

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, log_model.predict(X_test)))

print("\n--- KNN Report ---")
print(classification_report(y_test, knn_model.predict(X_test)))
