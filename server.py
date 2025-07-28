from flask import Flask, render_template, request, jsonify
import os
from predict import predict  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    # path to object
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        class_name, ttl = predict(filepath)
        return jsonify({
            "class": class_name,
            "ttl": ttl,
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # delete the uploaded file after processing
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)