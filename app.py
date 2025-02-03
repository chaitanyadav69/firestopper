from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as tflite

app = Flask(__name__)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="wildfire_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return jsonify({"message": "Server is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in request body"}), 400

        # Convert input to numpy array and reshape
        input_data = np.array(data["features"], dtype=np.float32).reshape(1, -1)

        print("Received input data:", input_data)  # ✅ Debugging step

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print("Raw Model Output:", output_data)  # ✅ Debugging step

        # Convert output to a readable format
        prediction = output_data.tolist()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
