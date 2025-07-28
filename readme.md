# Banana Ripeness and TTL Prediction with CNN

> **Cross-platform:** Works on Linux, macOS, and Windows

This project is a complete pipeline for detecting bananas in images, classifying their ripeness stage, and predicting their estimated time to live (TTL) in days. Built with PyTorch and enhanced with a web UI using Flask + HTML/CSS/JS, the project is suitable for both research and real‑time applications.

---

## 🚀 Features

- **Multi‑class Classification:** Predicts ripeness stage (4 classes: A–D)
- **Regression:** Predicts TTL (number of days before expiration)
- **Banana Detection:** Filters out non‑banana images using pretrained ResNet‑18 + ImageNet classes
- **Training Pipeline:** Model training, validation, and metrics logging
- **Model Evaluation:** Accuracy, F1 score, MAE, RMSE computed on validation set
- **Interactive Web UI:** Upload image and get predictions via a Flask server

---

## 📁 Project Structure

```
├── dataset.py               # Dataset class with TTL normalization
├── train.py                 # Training pipeline with 80/20 split, metrics & plots
├── predict.py               # CLI prediction with banana check & TTL de‑normalization
├── server.py                # Flask app serving the HTML/CSS/JS UI
├── static/
│   ├── style.css            # CSS styling for the web UI
│   └── (other assets…)      # e.g. JavaScript if needed
├── templates/
│   └── index.html           # HTML template for uploading & displaying results
├── generate_dataset_csv.py  # Script to build CSV from folder structure
├── banana_model.pth         # Saved PyTorch model weights
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation (this file)
```

---

## 🧠 Model Architecture (BananaNet)

A custom convolutional neural network with a shared feature extractor and dual heads for multi‑task learning:

1. **Feature extractor:**

   - 3 × [Conv2D → ReLU → MaxPool]
   - Input: 3×224×224 → Output: 64×28×28

2. **Fully‑connected trunk:**

   - Flatten → Linear(64×28×28 → 256) → ReLU

3. **Dual heads:**

   - **Classifier head:** Linear(256 → 4) + Softmax → Ripeness (A–D)
   - **Regressor head:** Linear(256 → 1) → TTL (normalized)

---

## 🧪 Dataset

- **Source:** Custom-curated banana images (open‑source repository)
- **Folder layout:**
  ```
  dataset/
    ├── train/
    │   ├── Class A/
    │   ├── Class B/
    │   ├── Class C/
    │   └── Class D/
    ├── validation/
    └── test/
  ```
- **CSV fields:**
  - `image_path` (relative path)
  - `ripeness_class` (Class A/B/C/D)
  - `estimated_day` (day of ripeness)
  - `ttl_days_left` (float days until expiration)
- **Preprocessing:**
  - Resize images to 224×224
  - Normalize pixel values to [0,1]
  - Normalize TTL: `(ttl_days_left – mean) / std` (mean=5.0, std=2.0)

---

## 📊 Training Summary

| Metric                  | Value        |
| ----------------------- | ------------ |
| **Classification loss** | 107.2 → 16.0 |
| **TTL regression loss** | 31.77 → 1.90 |
| **Accuracy**            | 81.89 %      |
| **Precision**           | 82.79 %      |
| **Recall**              | 81.69 %      |
| **F1 Score**            | 81.61 %      |
| **MAE (TTL)**           | 0.1186       |
| **RMSE (TTL)**          | 0.1515       |

You can also view the learning‑rate schedule plot saved as `learning_rate_schedule.png` in the project root.

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** We pin `numpy<2.0` until PyTorch fully supports NumPy 2.x.

### 2. Train the model

```bash
python train.py
```

This will split 80/20, train for 10 epochs, evaluate metrics, and save the model as `banana_model.pth`.

### 3. Start the Flask server

```bash
python server.py
```

After launch, the console will show a URL (e.g. `http://127.0.0.1:5000/` or `http://0.0.0.0:5000/`). Open that in your browser to access the web UI.

### 4. Command-line Prediction (optional)

```bash
python predict.py --image path_to_image.jpg
```

### 5. Optional
If you use python3 and pip3 instead of 
```bash
pip install -r requirements.txt
```
you can use 
```bash
pip3 install -r requirements.txt
```
or

```bash
python3 server.py
```

---

## 🔍 Requirements

```text
Flask>=3.0,<4
torch==2.3.1
torchvision==0.18.1
pillow>=10.0,<11
numpy>=1.26,<2
pandas>=2.2,<3
scikit-learn>=1.4,<2
matplotlib>=3.8,<4
tqdm>=4.66,<5
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📷 Example Output

```text
🔍 Detected object: banana (99.2%)
🍌 Ripeness: Class B (Ripe)
⏳ Estimated days until expiration: 4.87 days
```

---

## ✨ Contributors

- **Rufat Abdulzada** — Project Author and Developer\
  Master’s Candidate, Computer Science\
  Algoma University

---

## 📄 License

Released under the MIT License.

