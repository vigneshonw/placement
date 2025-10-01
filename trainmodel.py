import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('CollegePlacement.csv')

# Drop missing values
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()
df_encoded = df.copy()
for col in df.select_dtypes(include='object'):
    df_encoded[col] = le.fit_transform(df[col])

# Drop columns not used as features
X = df_encoded.drop(['Placement', 'College_ID'], axis=1, errors='ignore')
y = df_encoded['Placement']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train and test (if you want to keep this for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Initialize model
model = RandomForestClassifier(random_state=42)

# Cross-validation (5-fold)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracies: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation CV accuracy: {cv_scores.std():.4f}")

# Train on full training set (or full data here)
model.fit(X_train, y_train)

# Evaluate on test set
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_acc:.4f}")

# Save model, scaler, and encoder
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
