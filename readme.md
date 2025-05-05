# Speaker Diarization & Transcription System

- **OLD**: https://chatgpt.com/share/68124798-93fc-8006-b343-51358aed110c
- **LATEST**: https://chatgpt.com/share/681678db-b944-800a-861f-947828145105
- **Compiled Google Doc**: https://docs.google.com/document/d/1StsDuGMUWbTJOWugY-tG2tn5I1lpVaO-_KTAK8wABXg/edit?usp=sharing

**Goal**: Transcribe audio while separating who said what in real time.




## System Overview

```
[ Microphone/Audio File ]
         ↓
[ Voice Activity Detection (VAD) ]
         ↓
[ Speaker Embedding Extraction ]
         ↓
[ Speaker Clustering ]
         ↓
[ Real-time Transcription (STT) ]
         ↓
[ Combine → Diarized Transcript ]
```

## Step-by-Step Implementation Guide

### 1.  Data Collection & Preprocessing

**Goal**: Collect or simulate multi-speaker audio data.

Use:

- VoxCeleb1/2 (speaker data)
    - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
    - https://academictorrents.com/details/bdd9f57a6f47aa197f502b68bc0195f5ac786ec4

- AMI Meeting Corpus (meeting-style data) [DONE]
- LibriSpeech (for STT training)


Preprocess:
- Normalize audio, split into frames (20-40 ms)
- Convert to mono, 16 kHz

### 2. Voice Activity Detection (VAD)

**Goal**: Detect where speech occurs (skip silence/noise).

Implement:
- Energy-based or spectral entropy method
- Compute Short-Time Energy or use zero-crossing rate

📌 Tip: Use a sliding window and classify each frame as "speech" or "non-speech."


### 3. Feature Extraction
**Goal**: Convert speech segments to embeddings

Features:
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Spectrograms
- Chroma features

📌 Input: speech segments → Output: vectors (13–40 dims for MFCCs)


### 4. Speaker Embedding Model
**Goal**: Build an embedding space where same-speaker segments cluster together

Build a Siamese Network or Triplet Network:
- Train on speaker verification: “Is this the same speaker?”
- Input: Pairs or triplets of MFCC features
- Loss: Contrastive or Triplet Loss

📌 Output: 128-512 dimensional speaker embeddings



### 5. Speaker Clustering
**Goal**: Group embeddings to label speakers (unsupervised)

Clustering algorithms:
- Agglomerative Hierarchical Clustering (AHC)
- Spectral Clustering
- DBSCAN (density-based, good for unknown number of speakers)

📌 Tip: Use cosine similarity to compare embeddings


### 6. ASR (Automatic Speech Recognition)
**Goal**: Train your own basic STT model

Use:
- Spectrogram + CTC Loss + BiLSTM/Transformer encoder

Dataset: LibriSpeech or Common Voice

Architecture:
- Input: Spectrogram
- Encoder: BiLSTM layers
- Decoder: CTC output for character prediction

📌 Train on clean, single-speaker data before mixing in multi-speaker


<!-- ### 7. Synchronization & Real-Time Handling
Stream from microphone using pyaudio or sounddevice

Process in small time chunks (e.g. 1–2 sec buffers)

Use queues/threads to keep VAD, diarization, and STT pipelines running in parallel -->



### 8. Post-Processing & Transcript Generation
Merge diarization and ASR output using timestamps

Format:
```
[Speaker 1] Hello, how are you?
[Speaker 2] I'm good, thanks! You?
...
```


### 9. Evaluation
- Diarization Error Rate (DER)
- Word Error Rate (WER)
<!-- - Real-time factor (RTF) for speed -->


## ⚙️ Tools & Libraries (Only for Support, Not Pre-built Models)
- numpy, scipy, librosa: audio & signal processing
- pyaudio or sounddevice: real-time mic capture
- matplotlib: visualize embeddings (e.g., t-SNE)
- scikit-learn: clustering (you can re-implement if needed)
- torch or tensorflow: model building



## 📊 Deliverables (for dissertation)
- Real-time application (demo with mic or file input)
- Trained diarization & STT models
- Evaluation results (tables, graphs)
- Full report with:
    - Data pipeline
    - Model architectures
    - Diarization algorithm
    - Real-time integration
    - Limitations & future work






