import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_excel(path)

def preprocess_data(df, fit=True, encoder=None, scaler=None):
    data = df.copy()

    # Drop 'Churn Label' if present
    if 'Churn Label' in data.columns:
        data.drop('Churn Label', axis=1, inplace=True)

    # Drop unnecessary columns if they exist
    columns_to_drop = [
        "CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long",
        "Latitude", "Longitude", "Multiple Lines", "Paperless Billing",
        "Payment Method", "Total Charges", "Churn Reason", "Online Security",
        "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies"
    ]
    existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=existing_cols_to_drop, axis=1, inplace=True)

    # Fill missing values
    for column in data.columns:
        if data[column].isnull().any():
            if data[column].dtype == 'object':
                data[column].fillna('Not Specified', inplace=True)
            else:
                data[column].fillna(data[column].mean(), inplace=True)

    # Encode categorical variables
    categorical_features = data.select_dtypes(include=['object']).columns
    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop="first")
        encoded_array = encoder.fit_transform(data[categorical_features].astype(str))
    else:
        encoded_array = encoder.transform(data[categorical_features].astype(str))
    encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(), index=data.index)
    data = pd.concat([data.drop(columns=categorical_features), encoded_data], axis=1)

    # Scale numerical
    numerical_features = data.select_dtypes(include=['int64', 'float64']).drop('Churn Value', axis=1).columns
    if fit:
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    else:
        data[numerical_features] = scaler.transform(data[numerical_features])

    X = data.drop('Churn Value', axis=1)
    y = data['Churn Value'].astype(int)

    return X, y, encoder, scaler