import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def plot_hand_gesture(X_train, y_train, idx):
    """
    Visualize a hand gesture from the training set using 2D coordinates of landmarks.
    Parameters:
    - X_train: numpy array of shape (N, 42) where 42 are features (21 landmarks , 2 coordinates [x, y])
    - y_train: pandas Series or numpy array of shape (N,) containing gesture labels
    - idx: index of the sample to visualize
    """
    # Take a sample gesture from the training set
    # sample_idx = X_train[idx]
    sample = X_train[idx]

    # Reshape the data into (x, y) coordinates for 21 landmarks
    landmarks = sample.reshape(-1, 2)  # now (21, 2)

    # Define connections between landmarks (hand skeleton)
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky finger
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]

    # Plot in 2D
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw connections (lines)
    for start, end in connections:
        x_coords = [landmarks[start, 0], landmarks[end, 0]]
        y_coords = [landmarks[start, 1], landmarks[end, 1]]
        ax.plot(x_coords, y_coords, 'g-', linewidth=1.5, alpha=0.6)

    # Plot the landmarks as points
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=45, zorder=5)

    # Label each landmark
    for i, (x, y) in enumerate(landmarks):
        ax.text(x, y-0.01, str(i), fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Hand Gesture: {y_train.iloc[idx]}')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.show()


def plot_random_hand_gestures(X_train, y_train, num_samples=5):
    """
    Visualize a random selection of hand gestures from the training set.
    Parameters:
    - X_train: numpy array of shape (N, 42) where 42 are features (21 landmarks, 2 coordinates [x, y])
    - y_train: pandas Series or numpy array of shape (N,) containing gesture labels
    - num_samples: number of random samples to plot (default: 5)
    """
    # Randomly select indices
    random_indices = np.random.choice(X_train.shape[0], size=num_samples, replace=False)
    
    # Plot each sample
    for idx in random_indices:
        plot_hand_gesture(X_train, y_train, idx)


def preprocess_data(X, wrist_idx=0, mid_finger_tip_idx=12):
    """
    Preprocess the dataset by recentering to the wrist and normalizing by the distance to the middle finger tip.
    Parameters:
    - dataset: numpy array of shape (N, 63) where 63 are features (21x3)
    - wrist_idx: index of wrist landmark
    - mid_finger_tip_idx: index of middle finger tip landmark
    Returns:
    - processed_dataset: numpy array of shape (N, 42) where 42 are features (21 landmarks × 2 coordinates [x, y])
    """
    
    # Reshape features into (N, 21, 3)
    landmarks = X.reshape(-1, 21, 3)

    # Drop z -> (N, 21, 2)
    landmarks_xy = landmarks[:, :, :2]

    # Step 1: recenter to wrist
    wrist = landmarks_xy[:, wrist_idx, :]  # shape (N, 2)
    centered = landmarks_xy - wrist[:, None, :]  # broadcast subtraction

    # Step 2: normalize by mid-finger tip distance
    mid_tip = centered[:, mid_finger_tip_idx, :]  # shape (N, 2)
    norm_factor = np.linalg.norm(mid_tip, axis=1)  # shape (N,)
    norm_factor = np.where(norm_factor == 0, 1, norm_factor)  # avoid div by zero

    normalized = centered / norm_factor[:, None, None]

    # Flatten back to (N, 42) since we now have 21 landmarks × 2 coords
    processed_dataset = normalized.reshape(-1, 42)

    return processed_dataset


def encode_labels(y):
    """
    Encode string labels into integers.
    Parameters:
    - y: pandas Series or numpy array of shape (N,) containing gesture labels
    Returns:
    - encoded_labels: numpy array of shape (N,) containing integer-encoded labels
    - label_encoder: fitted LabelEncoder instance for inverse transformation if needed
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)
    return encoded_labels, label_encoder


def log_model_to_mlflow(model, run_name, tags=None, param_list=None, metrics_list=None):
    """
    Log a trained model to MLflow with specified run name, tags, parameters, and metrics.
    
    Parameters:
    - model: the trained model object (e.g., sklearn model, keras model, etc.)
    - run_name: name for the MLflow run (string)
    - tags: dictionary of tags to log (e.g., {'model_type': 'svm', 'version': '1.0'})
    - param_list: dictionary of parameters to log (e.g., {'C': 1.0, 'kernel': 'rbf'})
    - metrics_list: dictionary of metrics to log (e.g., {'accuracy': 0.95, 'f1_score': 0.92})
    
    Returns:
    - run_id: the MLflow run ID
    """
    
    with mlflow.start_run(run_name=run_name):
        # Log tags if provided
        if tags:
            for tag_key, tag_value in tags.items():
                mlflow.set_tag(tag_key, tag_value)
        
        # Log parameters if provided
        if param_list:
            for param_key, param_value in param_list.items():
                mlflow.log_param(param_key, param_value)
        
        # Log metrics if provided
        if metrics_list:
            for metric_key, metric_value in metrics_list.items():
                mlflow.log_metric(metric_key, metric_value)
        
        # Log the model (sklearn flavor will auto-detect sklearn models)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Get the run ID
        run_id = mlflow.active_run().info.run_id
    
    return run_id
