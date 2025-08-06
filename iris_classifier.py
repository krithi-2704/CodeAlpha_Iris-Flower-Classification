import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Ensure consistent path to dataset ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Iris.csv")

# Load data
data = pd.read_csv(file_path)

# Drop 'Id' column if it exists
if 'Id' in data.columns:
    data.drop('Id', axis=1, inplace=True)

print("Sample Data:\n", data.head())
print("\nDataset Size:", data.shape)

# Pairplot
sns.pairplot(data, hue='Species')
plt.suptitle("Iris Feature Pairplot", y=1.02)
plt.savefig(os.path.join(current_dir, "pairplot.png"))
plt.close()

# Heatmap
feature_corr = data.drop('Species', axis=1).corr()
sns.heatmap(feature_corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(current_dir, "correlation_heatmap.png"))
plt.close()

# Prepare data
features = data.drop('Species', axis=1)
labels = data['Species']

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Models
model_list = {
    "KNN Classifier": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier()
}

# Train and evaluate each model
for model_name, clf in model_list.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    file_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(os.path.join(current_dir, file_name))
    plt.close()
