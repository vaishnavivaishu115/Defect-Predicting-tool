import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Scale the features
    features = ['lines_of_code', 'code_churn', 'num_developers', 'past_defects']
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    return data
