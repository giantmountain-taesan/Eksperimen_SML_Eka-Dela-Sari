import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from joblib import dump

def automate_preprocessing_skilled(data, save_path, header_path): 
    df_clean = data.copy()

    # 1. IDENTIFIKASI FITUR
    numeric_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()

    # 2. SIMPAN HEADER  
    pd.DataFrame(columns=df_clean.columns).to_csv(header_path, index=False)

    # 3. HANDLING MISSING VALUES (Tahap Awal untuk IQR) 
    for col in numeric_features:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    for col in categorical_features:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 4. HANDLING OUTLIERS (IQR) 
    if numeric_features:
        Q1 = df_clean[numeric_features].quantile(0.25)
        Q3 = df_clean[numeric_features].quantile(0.75)
        IQR = Q3 - Q1
        condition = ~((df_clean[numeric_features] < (Q1 - 1.5 * IQR)) | 
                      (df_clean[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
        df_clean = df_clean.loc[condition].reset_index(drop=True)

    # 5. MEMBANGUN PIPELINE (Struktur Berbeda untuk Otomatisasi)
    # Pipeline Numerik: Imputer + Scaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline Kategorikal: Imputer + Encoder + Scaler
    # Menggunakan OrdinalEncoder agar hasil sama dengan LabelEncoder (0,1,2..) 
    # tetapi bisa masuk ke dalam ColumnTransformer secara otomatis.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder()),
        ('scaler', StandardScaler()) # Scaling penting agar jarak cluster akurat
    ])

    # 6. COLUMN TRANSFORMER
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 7. EKSEKUSI & SIMPAN
    data_ready = preprocessor.fit_transform(df_clean)
    dump(preprocessor, save_path)
    
    print(f" Tahapan Eksperimen Berhasil Diotomatisasi!")
    print(f" Artifact disimpan: {save_path} & {header_path}")

    return data_ready, df_clean
