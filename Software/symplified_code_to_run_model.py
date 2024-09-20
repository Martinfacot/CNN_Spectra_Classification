import os
import numpy as np
import tensorflow as tf
import cv2

# Model path for alanine_modelv7
ALANINE_MODEL_PATH = r'C:\Users\PC\Documents\BIOSFER\Sofware\Saved model\alanine_modelv7(bestone).keras'

# Load alanine_modelv7
alanine_model = tf.keras.models.load_model(ALANINE_MODEL_PATH)

# Preprocess the image for prediction
def preprocess_image(image_path, coords=[70, 108], crop_size=[465, 605], resize_shape=[256, 256]):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ensure the image has the expected dimensions
    if img.shape != (600, 800, 3):
        return None

    # Crop and resize the image
    cropped_img = img[coords[0]:(coords[0] + crop_size[0]), coords[1]:(coords[1] + crop_size[1])]
    resized_img = cv2.resize(cropped_img, resize_shape, interpolation=cv2.INTER_AREA)
    normalized_img = resized_img.astype(np.float32) / 255.0
    return normalized_img

# Predict using the alanine_modelv7
def predict_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        raise ValueError('Invalid image format or size')

    # Prepare the image for prediction
    image_tensor = tf.convert_to_tensor(np.expand_dims(processed_image, axis=0), dtype=tf.float32)
    
    # Make prediction
    prediction = alanine_model.predict(image_tensor)
    probability = float(prediction[0][0])

    # Determine the result based on the probability
    if probability >= 0.995:
        result = "High confidence"
        color = "green"
    elif 0.20 < probability < 0.995:
        result = "Medium confidence"
        color = "orange"
    else:
        result = "Low confidence"
        color = "red"

    return {
        'probability': probability,
        'result': result,
        'color': color
    }

def process_folder(folder_path):
    results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            prediction_result = predict_image(image_path)
            results.append({
                'file_name': image_file,
                'prediction': prediction_result
            })
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    return results


####----------------------------------------------------------------------------------------------------------####
# Test the programm
if __name__ == "__main__":
    test_folder_path = r'C:\Users\PC\Documents\BIOSFER\Sofware\Alaline_test_images'
    try:
        folder_results = process_folder(test_folder_path)
        for result in folder_results:
            print(f"File: {result['file_name']}")
            print(f"Prediction: {result['prediction']}")
            print("---")
    except Exception as e:
        print(f"Error: {str(e)}")