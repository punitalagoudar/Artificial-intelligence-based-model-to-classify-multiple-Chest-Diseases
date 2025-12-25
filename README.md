# Chest Disease Classifier

A deep-learning-based application for automated classification of multiple chest diseases from radiographic images. This repository contains scripts for training, inference, and a minimal web backend for demonstration.

---

## Table of contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset organization](#dataset-organization)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Serving the model](#serving-the-model)
- [Model artifacts](#model-artifacts)
- [Development & testing](#development--testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project implements an automated chest-disease classification pipeline using computer vision and deep learning (PyTorch). It includes code for data preparation, model training, evaluation, prediction, and a lightweight server interface for demonstration or integration.

Key goals:
- Provide a reproducible training and inference pipeline
- Offer easy-to-use scripts for experimentation and deployment
- Keep repository structure simple to facilitate extension

---

## Repository layout

```
.
├── check/                # Utility scripts
├── dummy/                # Placeholder / test data
├── static/               # Static web assets (images, CSS, JS)
├── templates/            # HTML templates for the web UI
├── test/                 # Test scripts and sample test dataset
├── train/                # Training images (ImageFolder structure)
├── val/                  # Validation images (ImageFolder structure)
├── check_all_images      # Batch image checking script
├── check1                # Additional utility script
├── chest                 # Main application file (entry point)
├── Fn_Prediction         # Prediction helper functions
├── import torch          # Model definition and imports (script)
├── newtrain              # Model training script / entry point
├── phase3.pth            # Trained model weights (example)
├── phase3predict         # Inference / prediction script
├── Server                # Backend server script / entry point
├── Tuberculosis-685.png  # Sample image
├── README.md             # Project documentation
├── LICENSE               # License file
├── .gitignore            # Git ignore
```

Note: Some scripts may be executable files without a `.py` extension. If required, run them with `python <script>` or mark them executable (`chmod +x <script>`).

---

## Screenshots

<img src="assets/login_demo.png" alt="Login Demo" width="400"/>
<img src="assets/admin_menu.png" alt="Admin Menu" width="600"/>

---

## Requirements

- Python 3.8+
- PyTorch (compatible with your CUDA/CPU)
- torchvision
- Pillow
- (Optional) Additional dependencies listed in `requirements.txt` if present

Recommended installation using a virtual environment.

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/punitalagoudar/Artificial-intelligence-based-model-to-classify-multiple-Chest-Diseases.git
cd Artificial-intelligence-based-model-to-classify-multiple-Chest-Diseases
```

2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows (PowerShell)
```

3. Install required packages

```bash
pip install --upgrade pip
pip install torch torchvision pillow
# If a requirements file exists:
pip install -r requirements.txt
```

---

## Dataset organization

The code expects images to be organized for use with `torchvision.datasets.ImageFolder`. A typical layout:

```
train/
  class_1/
    img001.jpg
    img002.jpg
  class_2/
val/
  class_1/
  class_2/
test/
  <images or same ImageFolder structure>
```

Prepare your dataset accordingly and ensure file permissions allow read access.

---

## Usage

General notes:
- Some repository scripts may be executable without a `.py` extension. If execution fails, prepend `python` to the command.
- Adjust any hard-coded paths in scripts or provide paths via command-line options where available.

### Training

To train a model:

```bash
# If the script is executable:
./newtrain
# Or explicitly with python:
python newtrain
```

The training script should save model checkpoints (for example, `phase3.pth`) in the repository root or a configured output directory.

### Inference

Predict on a single image:

```bash
# If script accepts --image argument as shown:
python phase3predict --image /path/to/image.jpg
```

The prediction utility uses the trained weights (e.g., `phase3.pth`) and the model definition in the repository. Confirm the expected model path inside the script or set via an argument if supported.

### Serving the model

A simple backend server is included for demonstration:

```bash
python Server
```

Visit the server URL (typically http://127.0.0.1:5000 or as printed by the server script) to use the web UI. The `templates/` and `static/` folders contain the frontend assets.

---

## Model artifacts

- phase3.pth — example trained model weights. Replace with your trained checkpoint for production or evaluation.

When sharing models, avoid committing large binaries to git; prefer model hosting (S3, GDrive) and reference download instructions in the README.

---

## Development & testing

- Use the scripts in `check/` and `test/` to validate dataset integrity and pipeline steps.
- Add unit tests or CI checks to verify training and inference behavior before merging changes.
- Keep experiment code (ad-hoc notebooks or scripts) in a separate directory to maintain a clean production path.

---

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a new branch with a descriptive name
3. Implement and test your changes
4. Commit with clear messages
5. Open a pull request describing the change and motivation

Please open an issue to discuss major changes or if you need guidance before implementing large features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full terms.

---

## Contact

Repository: [punitalagoudar/Artificial-intelligence-based-model-to-classify-multiple-Chest-Diseases](https://github.com/punitalagoudar/Artificial-intelligence-based-model-to-classify-multiple-Chest-Diseases)

For questions or support, open an issue or contact the repository owner via GitHub.

---

## Acknowledgements

- Built using PyTorch and torchvision
- Inspired by standard medical-imaging classification workflows
- Thanks to contributors and the open-source community for datasets, tools, and best practices
