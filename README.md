# 🩺 Chest Disease Classifier

Welcome to **Chest Disease Classifier**!  
An advanced AI-powered diagnostic tool for classifying chest diseases using deep learning and computer vision.

---

## 📸 Demo

<img src="assets/login_demo.png" alt="Login Demo" width="400"/>
<img src="assets/admin_menu.png" alt="Admin Menu" width="600"/>

---

## 📦 Project Structure

```
.
├── check/                # Utility scripts
├── dummy/                # Placeholder/test data
├── static/               # Static files for web (images, CSS, JS)
├── templates/            # HTML templates for frontend
├── test/                 # Test scripts and datasets
├── train/                # Training image dataset
├── val/                  # Validation image dataset
├── check_all_images      # Batch image checking script
├── check1                # Additional utility script
├── chest                 # Main application file
├── Fn_Prediction         # Prediction helper functions
├── import torch          # Model definition and imports
├── newtrain              # Model training script
├── phase3.pth            # Trained model weights
├── phase3predict         # Inference script
├── Server                # Backend server script
├── Tuberculosis-685.png  # Sample image
├── README.md             # Project documentation
├── LICENSE               # License details
├── .gitignore            # Git ignore file
```

---

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/chest-disease-classifier.git
    cd chest-disease-classifier
    ```

2. **Install dependencies:**
    ```bash
    pip install torch torchvision pillow
    ```

3. **(Optional) Install additional requirements if you have a `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare your data:**
    - Place your training images in the `train/` folder and test images in the `test/` folder, structured for `torchvision.datasets.ImageFolder`.

---

## 🧑‍💻 Usage

```python
# Train the model
python newtrain

# Predict on a single image
python phase3predict --image /path/to/image.jpg
```

---

## 🤝 Contributing

We welcome contributions from the community!  
Steps to contribute:
1. Fork the repo  
2. Create a branch  
3. Make your changes  
4. Commit and push  
5. Open a Pull Request  

Please open an issue first if you want to suggest a major change.

---

## 📄 License

This project is licensed under the **MIT License**.  
You may use, copy, modify, and distribute it freely, provided proper credit is given.  

See the [LICENSE](LICENSE) file for more details.
