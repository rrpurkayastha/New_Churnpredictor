import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.title("Customer Churn Predictor")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of data:")
    st.dataframe(df.head())

    # Detect target column
    target_col = None
    for col in df.columns:
        if col.lower() in ["churn", "is_churn", "customer_churn", "exited"]:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    st.write(f"Target column: **{target_col}**")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    # Pipelines
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # ✅ FIX HERE
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y_encoded)
    st.success("Model trained successfully!")

    st.subheader("Enter new customer data")

    user_input = {}
    for col in X.columns:
        if col in numeric_features:
            user_input[col] = st.number_input(col, value=float(X[col].dropna().iloc[0]))
        else:
            user_input[col] = st.text_input(col, value=str(X[col].dropna().iloc[0]))

    if st.button("Predict Churn"):
        input_df = pd.DataFrame([user_input])

        prediction_encoded = model.predict(input_df)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        st.success(f"Predicted Churn: {prediction}")