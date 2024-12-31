from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("model.h5")

# List of class names (example: CIFAR-10 classes, adjust as needed)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Home route to render the initial HTML form
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("test.html")

# Image preprocessing function to resize and normalize the image
def preprocess_image(image):
    # Resize the image to 32x32 to match the CIFAR-10 image size (if needed)
    image = image.resize((32, 32))
    
    # Convert the image to a numpy array and normalize to [0, 1] range
    image = np.array(image) / 255.0
    
    # Ensure the image has 3 channels (RGB) if not already
    if image.ndim == 2:  # If the image is grayscale (shape: (32, 32))
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB (shape: (32, 32, 3))

    # Ensure the image has the correct shape (32, 32, 3)
    image = np.expand_dims(image, axis=0)  # Add batch dimension: shape becomes (1, 32, 32, 3)
    
    return image

# Route to handle the prediction of the uploaded image
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    # Check if file is provided in the request
    if 'file' not in request.files:
        return "<h2>No file provided. Please upload an image</h2>"
    
    # Get the uploaded file
    file = request.files['file']
    try:
        # Open the image and ensure it's in RGB format
        image = Image.open(file).convert('RGB')
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction using the trained model
        prediction = model.predict(processed_image)

        # Convert the prediction to a list of probabilities (for all classes)
        prediction_list = prediction.tolist()[0]  # Get the first element since the batch size is 1

        # Get the class with the highest probability
        predicted_class_index = np.argmax(prediction, axis=1)

        # Get the name of the predicted class using the index
        predicted_class_name = class_names[predicted_class_index[0]]

        # Prepare the result to display both the predicted class and the full list of probabilities
        prediction_result = f"Predicted Output: {predicted_class_name} (Class {predicted_class_index[0]}) with prediction-list: {prediction_list}"

        # Return the result to the 'predict.html' template
        return render_template("predict.html", prediction_result=prediction_result)
    
    except Exception as e:
        # Return error message if something goes wrong during the image processing or prediction
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
