from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as tflite

app = Flask(__name__)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="wildfire_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # Expecting a list of sensor values
        input_data = np.array(data, dtype=np.float32).reshape(1, -1)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0].tolist()

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
