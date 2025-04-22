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
