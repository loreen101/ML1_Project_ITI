import numpy as np
import matplotlib.pyplot as plt

def plot_hand_gesture(X_train, y_train, idx):
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
        ax.plot(x_coords, y_coords, 'g-', linewidth=1, alpha=0.6)

    # Plot the landmarks as points
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=100, zorder=5)

    # Label each landmark
    for i, (x, y) in enumerate(landmarks):
        ax.text(x, y, str(i), fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Hand Gesture: {y_train.iloc[idx]}')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.show()



def preprocess_data(dataset, wrist_idx=0, mid_finger_tip_idx=12):
    """
    dataset: numpy array of shape (N, 63) where 63 are features (21x3)
    wrist_idx: index of wrist landmark
    mid_finger_tip_idx: index of middle finger tip landmark
    """
    
    # Reshape features into (N, 21, 3)
    landmarks = dataset.reshape(-1, 21, 3)

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
