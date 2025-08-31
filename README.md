# Speech Emotion Recognition (SER) using CNN-BiLSTM

This project implements a Speech Emotion Recognition (SER) model using the **TESS Toronto Emotional Speech Set dataset**.  
The model extracts **MFCC features** from speech, processes them with a **CNN-BiLSTM architecture**, and classifies emotions.

## 🎙️ Dataset
- **TESS Toronto Emotional Speech Set**
- Contains recordings of 7 emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral
- Audio files: `.wav` format
- Classes are inferred from folder names

## 🔧 Features & Preprocessing
- Audio resampled to **16kHz**
- Extracted features: **MFCC (40 coefficients)** + **Delta** + **Delta-Delta**
- Fixed-length input: **200 frames max**
- Normalization applied to raw audio
- Dataset split:  
  - Train: 70%  
  - Validation: 15%  
  - Test: 15%

## 🧠 Model Architecture
- **CNN layers**:
  - Conv2D → BatchNorm → ReLU → MaxPooling (x2)
- **BiLSTM layer**:
  - Hidden size = 128  
  - Bidirectional  
- **Dense head**:
  - Linear → ReLU → Dropout → Linear (n_classes)

## ⚙️ Training
- Optimizer: **Adam** (lr=1e-3)  
- Loss: **CrossEntropyLoss**  
- Epochs: **20**  
- Batch size: **32**  
- Early model checkpointing based on best validation accuracy  

## 📈 Evaluation
Evaluated on the test set using:
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

The script prints:
- Training/Validation accuracy per epoch
- Final classification report
- Final confusion matrix

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Download and place the TESS dataset in the project folder. Update the DATA_DIR path inside the script.

4. Run training and evaluation:
   ```bash 
   python ser_cnn_bilstm.py


5. The best model is saved as:
   ```bash
   tess_ser_best.pt

## 🔮 Future Improvements

- Data augmentation (noise injection, pitch shift, time stretch)

- More advanced architectures (transformers, wav2vec2.0)

- Hyperparameter tuning

- Deployment as a real-time application (Flask / Streamlit)
