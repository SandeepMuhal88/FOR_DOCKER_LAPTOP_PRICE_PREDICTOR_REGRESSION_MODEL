# Potato Leaf Disease Classification using PyTorch and Streamlit

## Project Overview
This project implements a deep learning–based image classification system to detect diseases in potato plant leaves. The model is developed using PyTorch and deployed using a Streamlit web application for real-time prediction.

The system classifies potato leaf images into three categories:
- Potato Early Blight
- Potato Late Blight
- Healthy Potato Leaf

The objective of this project is to assist in early disease detection to improve crop management and agricultural productivity.

---

## Features
- Custom Convolutional Neural Network (CNN) using PyTorch
- High accuracy on validation data
- Streamlit-based interactive web interface
- Simple and lightweight deployment
- Modular and extensible code structure

---

## Step 2: Create Virtual Environment (Optional but Recommended)
```
python -m venv .env
source .env/bin/activate        # Linux / macOS
.env\Scripts\activate           # Windows
```

## Step 3: Install Dependencies
```
pip install -r requirements.txt
```

## Run the Application

To start the Streamlit web application, run:

```
streanlit run application.app
```

If any case the torch in not run it through error then install that 
```
## Prerequisites
Python (version 3.10 or later is recommended for the latest PyTorch).
pip (or Anaconda/Miniconda) installed and up-to-date. 
Installation Steps
Choose your preferred package manager and run the corresponding command in your terminal or Anaconda prompt.
Method 1: Using Pip
This is the most common method. The standard 

```
pip install torch
```
command usually installs a version that includes a default CUDA runtime (if available on the PyPI index), which works even on CPU-only machines but results in a larger file size. 
