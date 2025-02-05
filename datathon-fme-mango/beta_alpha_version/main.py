import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load data
product_data = pd.read_csv('/Users/nicolasrosales/Documents/GitHub/datathon_2024_fme_mango/datathon-fme-mango/archive/product_data.csv')
attribute_data = pd.read_csv('/Users/nicolasrosales/Documents/GitHub/datathon_2024_fme_mango/datathon-fme-mango/archive/attribute_data.csv')
test_data = pd.read_csv('/Users/nicolasrosales/Documents/GitHub/datathon_2024_fme_mango/datathon-fme-mango/archive/test_data.csv')

# Merge product_data and attribute_data on cod_modelo_color
merged_data = pd.merge(product_data, attribute_data, on='cod_modelo_color', how='inner')

# Preprocess categorical features (label encoding)
le = LabelEncoder()
categorical_columns = ['des_sex', 'des_age', 'des_line', 'des_fabric', 'des_product_category', 
                       'des_product_aggregated_family', 'des_product_family', 'des_product_type', 
                       'attribute_name']
for col in categorical_columns:
    merged_data[col] = le.fit_transform(merged_data[col])

# Split into features (X) and target (y)
X = merged_data.drop(columns=['des_value'])
y = merged_data['des_value']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess images using a pre-trained CNN (e.g., ResNet50)
def extract_image_features(image_paths):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    image_features = []
    for path in image_paths:
        img = load_img(path, target_size=(160, 224))
        img = img_to_array(img)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        features = model.predict(img)
        image_features.append(features.flatten())
    return np.array(image_features)

# Get image paths for training and validation
train_image_paths = [os.path.join('/Users/nicolasrosales/Documents/GitHub/datathon_2024_fme_mango/datathon-fme-mango/archive/images/images', row.des_filename) for row in X_train.itertuples()]
val_image_paths = [os.path.join('/Users/nicolasrosales/Documents/GitHub/datathon_2024_fme_mango/datathon-fme-mango/archive/images/images', row.des_filename) for row in X_val.itertuples()]

# Extract image features
train_image_features = extract_image_features(train_image_paths)
val_image_features = extract_image_features(val_image_paths)

# Combine image features with tabular features
X_train_combined = np.hstack((X_train.drop(columns=['des_filename']), train_image_features))
X_val_combined = np.hstack((X_val.drop(columns=['des_filename']), val_image_features))

# Train a model (e.g., Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_combined, y_train)

# Validate the model
y_pred = model.predict(X_val_combined)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Prepare test data for predictions
test_image_paths = [os.path.join('images', f'{row.cod_modelo_color}', row.des_filename) for row in test_data.itertuples()]
test_image_features = extract_image_features(test_image_paths)
test_data_combined = np.hstack((test_data.drop(columns=['des_filename']), test_image_features))

# Generate predictions for test data
test_predictions = model.predict(test_data_combined)

# Save predictions to submission file
submission = pd.DataFrame({
    'test_id': test_data['test_id'],
    'des_value': test_predictions
})
submission.to_csv('submission.csv', index=False)
