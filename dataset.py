# Import necessary libraries
import pandas as pd
import zipfile
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Download and extract the ZIP file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
response = requests.get(url)  # Download the ZIP file

# Extract and read the specific CSV file ('bank.csv')
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open('bank.csv') as csv_file:
        df = pd.read_csv(csv_file, sep=';')  # Read CSV with separator

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Preprocess the data (encoding categorical variables)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Step 3: Split the data into features (X) and target (y)
X = df.drop('y', axis=1)  # Features
y = df['y']               # Target variable (whether customer subscribed or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Build and train the Decision Tree Classifier
classifier = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = classifier.predict(X_test)

# Step 6: Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Optional: Plot the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(classifier, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()
