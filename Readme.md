
#  Robust Face and Gender Recognition in Challenging Environments

This repository contains our complete solution for the **Comsys Hackathon**, where we address two real-world computer vision tasks using PyTorch, metric learning, and a web-based evaluation frontend.

---

##  Tasks Overview

### 🔹 Task A – Gender Classification
- **Objective**: Classify face images as **male** or **female**
- **Model**: ResNet18 backbone + binary classification head
- **Evaluation**: Accuracy, Precision, Recall, F1-score

### 🔹 Task B – Face Recognition (Matching)
- **Objective**: Match distorted face images to correct identities
- **Model**: Triplet-based embedding model using **FaceNet** (InceptionResnetV1)
- **Evaluation**: Top-1 Accuracy, Macro-averaged F1-score

---

## 📁 Project Structure

<pre> ```
Comsys_Hackathon/
│
├── data/
│   ├── Task_A/...
│   └── Task_B/...
│
│
├── backend.py
├── config.yaml
├── evaluate_task_a.py
├── evaluate_matcher_results.py
├── matcher.py
├── train.py
├── train_matcher.py
├── evaluate.py
├── main.py
│
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── helpers.py
│   └── triplet_dataset.py
│
├── models/
│   ├── multitask_model.py
│   └── embedding_model.py
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.pt     # For Task A
│   │   └── matcher_model.pt  # For Task B
│   │
│   └── results/
│       ├── face_recognition_results.csv
│       └── matcher_progress.txt
│
├── index.html
│
├── style.css
│
│
├── requirements.txt
├── test_path.txt
├── README.md
├── training_result.md
└── summary/
    └── Comsys_Hackathon.pdf
``` </pre>

---

##  Architecture

### Task A – Gender Classification
•	Backbone: ResNet18 (pretrained on ImageNet)
•	Final FC layer replaced with a 1-node Sigmoid output head
•	Only gender head used (no multitask mode)

        Input (224x224x3)
                ↓
        ResNet18 Backbone
                ↓
        Fully Connected Layer (1)
                ↓
            Sigmoid → Binary Gender Output

### Task B – Face Recognition (Matching)
•	Backbone: FaceNet (InceptionResNetV1) from facenet-pytorch
•	Embedding size: 512-D vector
•	Matching Logic:
        o	Reference embedding per identity from 1 clean image
        o	Compare distorted test images using cosine similarity
        o	Apply identity threshold (0.65) to validate match

        Test Image   → EmbeddingModel (FaceNet) → 512D Embedding
        Reference    → EmbeddingModel (FaceNet) → 512D Embedding
                        ↓
                Cosine Similarity
                        ↓
                Best Match + Threshold
                        ↓
                Match (Label = 1 or 0)

---

##  Setup Instructions

### 1. Clone the Repo
<pre> ```
git clone https://github.com/AshAryan12104/Comsys_Hackathon
cd Comsys_Hackathon
``` </pre>

### 2. Install Dependencies
<pre>
pip install -r requirements.txt
</pre>

### 3. Train Models

#### 🔹 Task A (Gender Classification)
<pre>
python main.py --mode train --config config.yaml
</pre>

#### 🔹 Task B (Face Matching)
<pre>
python train_matcher.py
</pre>

---

##  Download Pretrained Model Weights

To run evaluation without retraining, download the pretrained model checkpoints from the link below:

 [Download Model Weights from Google Drive](https://drive.google.com/drive/folders/1IjbLg77rXdhvyadaN2vDwgEpFyu935v2?usp=sharing)

### Files Included:

- `best_model.pt` → Used for Task A (Gender Classification)
- `matcher_model.pt` → Used for Task B (Face Matching)

###  Where to Place the Files:

After downloading, place them in the following location inside the project: (This is very important)
<pre> ```
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.pt     # For Task A
│   │   └── matcher_model.pt  # For Task B
``` </pre>

>  Make sure the folder structure matches exactly, or the evaluation scripts may not find the model files.

---

##  Evaluation

### 🔹 Task A (Gender)
<pre>
python evaluate_task_a.py --val_dir path/to/your/val/folder 
</pre>

**Example :**
<pre>
python evaluate_task_a.py --val_dir data/Task_A/val 
</pre>

### 🔹 Task B (Face Recognition)
<pre>
python matcher.py --test_dir path/to/your/val/folder
</pre>
After complete running matcher.py, Run
<pre>
python evaluate_matcher_results.py
</pre>

**Example :**
<pre>
python matcher.py --test_dir data/Task_B/val
</pre>
<pre>
python evaluate_matcher_results.py
</pre>

---

##  Frontend: Web Evaluation Portal 

1. Run the backend:
<pre>
python backend.py
</pre>

2. Open your browser and visit :
http://localhost:5000

3. Enter validation folder path (e.g. data/Task_A/val)

4. Click **Evaluate Task A** or **Evaluate Task B**

5. A progress bar appears for Task B and shows % as matcher.py runs in real-time.

6. On invalid path or wrong structure, error message is now shown on the webpage.

7. View metrics on screen after evaluation of respective Task !!!

> ⏳ If the server restarts or glitches on first try, just click the button again after refresh.

---

##  Sample Metrics

###  Task A
- Accuracy  : 94.7867 %
- Precision : 97.4277 %
- Recall    : 95.5836 %
- F1-Score  : 96.4968 %

###  Task B
- Top-1 Accuracy: 100.0000 %
- Precision : 100.0000 %
- Recall    : 100.0000 %
- F1-Score  : 100.0000 %
- Macro-Averaged F1-Score: 100.0000 %

> Training And Validation Results are on training_result.md

---

##  License

This project is developed as part of a Hackathon and is intended for academic and demonstration use only.

---

##  Team BYTEBash

- **Name:** Md Aryan Rehman, Raiyan Aftab Ansari, Priyanshu Mishra.
- **GitHub:** https://github.com/AshAryan12104
- **Email:** aryanrehman12104@gmail.com
---
