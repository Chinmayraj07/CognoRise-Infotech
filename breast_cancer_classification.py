import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Local path of the dataset
file_path = './data1.csv'  # Update with your local file path

# Load the dataset
df = pd.read_csv(file_path)

# Display the first few rows to understand the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop the 'id' column and the last column which is 'Unnamed: 32'
df = df.drop(columns=['id', 'Unnamed: 32'])

# Map the diagnosis column to numeric values
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Define feature columns and target column
X = df.drop(columns='diagnosis')
y = df['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42)
}

# Train and evaluate the models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f'{name} Performance:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')
    print('\n')
