# Analisis Prediktif Kecelakaan Lalu Lintas di India
# Script ini mendokumentasikan setiap tahapan proyek machine learning

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# 2. Data Understanding
# Load dataset
df = pd.read_csv('./accident_prediction_india.csv')

# Tampilkan 5 baris pertama
print("\n5 baris pertama dataset:")
print(df.head())

# Tampilkan informasi dataset
print("\nInformasi dataset:")
print(df.info())

# 3. Exploratory Data Analysis
# Distribusi target 'Accident Severity'
plt.figure(figsize=(8, 6))
df['Accident Severity'].value_counts(normalize=True).plot.pie(
    autopct='%1.1f%%', startangle=140
)
plt.title('Distribusi Kategori Kecelakaan')
plt.ylabel('')
plt.show()

# Analisis distribusi kolom kategoris
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col)
    plt.title(f'Distribusi {col}')
    plt.xticks(rotation=90)
    plt.show()

# Analisis distribusi kolom numerik
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=col, bins=30, kde=True)
    plt.title(f'Distribusi {col}')
    plt.show()

# Analisis korelasi numerik
numeric_cols = ['Number of Vehicles Involved', 'Number of Casualties',
                'Number of Fatalities', 'Speed Limit (km/h)', 'Driver Age']
corr = df[numeric_cols].corr()
fig, ax = plt.subplots()
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=90)
ax.set_yticklabels(numeric_cols)
for (i, j), val in np.ndenumerate(corr.values):
    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white')
plt.title('Korelasi Variabel Numerik', pad=20)
plt.show()

# 4. Data Preparation
# 4.1 Pembersihan dan imputasi
df_clean = df.copy()

print("\nJumlah missing values sebelum imputasi:")
print(df_clean.isnull().sum())

# Imputasi missing values
df_clean['Traffic Control Presence'] = df_clean['Traffic Control Presence'].fillna(
    df_clean['Traffic Control Presence'].mode()[0]
)
df_clean['Driver License Status'] = df_clean['Driver License Status'].fillna('Unknown')

print("\nJumlah missing values setelah imputasi:")
print(df_clean.isnull().sum())

# 4.2 Normalisasi dan Encoding
numeric_features = ['Number of Vehicles Involved', 'Number of Casualties',
                    'Number of Fatalities', 'Speed Limit (km/h)', 'Driver Age']

categorical_features = ['Weather Conditions', 'Vehicle Type Involved', 'Road Type', 
                        'Road Condition', 'Lighting Conditions', 'Driver License Status',
                        'Alcohol Involvement']

# Pipeline preprocessing
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

# 4.3 Split data
X = df_clean[numeric_features + categorical_features]
y = df_clean['Accident Severity'].map({'Minor':0, 'Serious':1, 'Fatal':2})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Modeling
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(class_weight='balanced', random_state=42)
}

# 6. Evaluasi Model
results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Minor','Serious','Fatal']))
    
    results[name] = acc
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, 
        display_labels=['Minor','Serious','Fatal'],
        cmap=plt.cm.Blues, 
        normalize='true'
    )
    plt.title(f'{name} Confusion Matrix')
    plt.show()

# Ringkasan Akurasi
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values('Accuracy', ascending=False)
print("\nRingkasan Akurasi Model:")
print(results_df)

# Feature Importance
for name, clf in models.items():
    if hasattr(clf, 'feature_importances_'):
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        
        try:
            importances = pipe.named_steps['classifier'].feature_importances_
        except AttributeError:
            continue
        
        feature_names = numeric_features + list(
            pipe.named_steps['preprocessor'].transformers_[1][1]
            .named_steps['onehot'].get_feature_names_out(categorical_features)
        )
        
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Feature Importances for {name}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
        plt.xlim([-1, len(importances)])
        plt.show()

# 7. Kesimpulan
print("""
7.1 Ringkasan Hasil
Akurasi terbaik: ~36% (Logistic Regression & Gradient Boosting)
Performa model di bawah target 75%
Kesulitan membedakan kategori Minor, Serious, dan Fatal

7.2 Insight Utama
- Speed Limit dan Driver Age paling berpengaruh
- Jumlah korban luka berpengaruh pada tingkat keparahan
- Kondisi jalan dan cuaca memberikan kontribusi signifikan

7.3 Rekomendasi Bisnis
1. Penegakan Batas Kecepatan
2. Program Pelatihan Pengemudi
3. Peningkatan Infrastruktur Jalan
""")