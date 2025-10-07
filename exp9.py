# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)

# Split features and labels
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Parameters you can change:
test_size = 0.3          # Change test size (default 0.2)
random_state = 42        # Change random seed for different splits
n_components = 2         # Number of LDA components (max 2 for Iris)
max_depth = 3            # Max depth for Random Forest

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Perform LDA
lda = LDA(n_components=n_components)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Train Random Forest classifier
classifier = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
classifier.fit(X_train, y_train)

# Predict on test set
y_pred = classifier.predict(X_test)

# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
