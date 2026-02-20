# Hand Gesture Recognition with Machine Learning

## Overview

This project focuses on building a machine learning model to recognize hand gestures using hand landmark data. The model processes normalized hand pose information to classify different hand gestures, enabling applications in human-computer interaction, sign language recognition, and gesture-based control systems.

## Dataset: HaGRID (Hand Gesture Recognition Image Dataset)

The project utilizes the **HaGRID (Hand Gesture Recognition Image Dataset)**, a comprehensive dataset containing hand gesture images with annotated hand landmarks. Key characteristics of the dataset:

- **Hand Landmarks**: 21 landmarks per hand, representing key points on the palm and fingers
- **Feature Format**: Each gesture is represented as a sequence of 3D coordinates (x, y, z) for each landmark
- **Raw Features**: 63 features per sample (21 landmarks × 3 coordinates)
- **Multiple Gestures**: The dataset includes various hand gesture categories for classification

The dataset is preprocessed and loaded from the `hand_landmarks_data.csv` file, which contains both feature vectors and gesture labels.

## Preprocessing Details

The preprocessing pipeline applies the following transformations to ensure robust and generalizable model training:

### 1. **Z-Coordinate Elimination**
- Removed the z-coordinate from each landmark, keeping only x and y coordinates
- **Rationale**: The z-coordinate (depth information) can introduce irrelevant variations that depend on camera distance and positioning rather than the actual hand shape
- **Result**: Reduced from 63 features to 42 features per sample (21 landmarks × 2 coordinates)

### 2. **Wrist-Based Centering**
- Recentered all landmarks relative to the wrist position (landmark 0)
- **Purpose**: Makes the gesture representation invariant to absolute position in the image
- **Benefit**: The model focuses on the relative structure of the hand, not its location

### 3. **Hand Size Normalization**
- Normalized the coordinates by the mid-finger tip distance (distance from wrist to middle finger tip)
- **Rationale**: Removes scale variations caused by different hand sizes or camera distances
- **Benefit**: Ensures the model learns the true shape and structure of gestures, independent of irrelevant size variations

### Overall Impact
These preprocessing steps make the model:
- **More Robust**: Less sensitive to camera positioning, hand size, and distance variations
- **More Generalizable**: Focuses on the intrinsic gesture shape and relative landmark positions
- **Better Performance**: Reduces overfitting to irrelevant features in the training data

## Tech Stack

- **Language**: Python 3
- **Data Processing**: 
  - `pandas` – Data loading and manipulation
  - `numpy` – Numerical computations
  - `scikit-learn` – Train/test splitting and model utilities
- **Visualization**: `matplotlib` – Hand landmark visualization
- **Development**: Jupyter Notebook for interactive experimentation
- **Experiment Tracking**: MLflow (for model tracking and versioning)

## Project Structure

```
ML1_Project/
├── README.md                      # Project documentation
├── trial.ipynb                    # Main experimentation notebook
├── utils.py                       # Utility functions (preprocessing, visualization)
├── hand_landmarks_data.csv        # Dataset containing hand landmarks and labels
└── mlruns/                        # MLflow experiment tracking directory
```

## Getting Started

*To be updated as project progresses*

## Future Updates

This README will be expanded as the project develops to include:
- Model architecture and selection details
- Training methodology and results
- Performance metrics and evaluation
- Usage examples and predictions
- Installation and setup instructions
