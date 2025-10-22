"""
Evaluate the trained brain tumor detection model on the test set.

This script uses the same preprocessing and splitting logic as the training script
so results are comparable.
"""
import os
from tensorflow.keras.models import load_model
from brain_tumor_detection import load_data, split_data, compute_f1_score, IMG_WIDTH, IMG_HEIGHT, BEST_MODEL_PATH
# Resolve repository root (directory containing this script)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    print("Evaluate saved model on test set")
    print("--------------------------------")

    # Determine dataset path (prefer augmented data). Use absolute paths so script
    # works regardless of current working directory.
    augmented_yes = os.path.join(REPO_ROOT, 'augmented data', 'yes')
    augmented_no = os.path.join(REPO_ROOT, 'augmented data', 'no')

    if os.path.exists(augmented_yes) and os.path.exists(augmented_no):
        yes_path = augmented_yes
        no_path = augmented_no
        print("Using augmented dataset")
    else:
        yes_path = os.path.join(REPO_ROOT, 'yes')
        no_path = os.path.join(REPO_ROOT, 'no')
        print("Augmented dataset not found, using original dataset")

    # Load data
    X, y = load_data([yes_path, no_path], (IMG_WIDTH, IMG_HEIGHT))

    # Split data (same proportions as training script)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

    print(f"Loaded dataset. Test set size: {X_test.shape[0]} examples")

    # Prefer the common saved model filename used during training
    candidate_model = os.path.join(REPO_ROOT, 'models', 'brain_tumor_model.keras')
    if os.path.exists(candidate_model):
        model_path = candidate_model
    else:
        # Fall back to BEST_MODEL_PATH from the training script (may be relative)
        model_path = os.path.join(REPO_ROOT, BEST_MODEL_PATH) if not os.path.isabs(BEST_MODEL_PATH) else BEST_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Evaluate
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Compute F1
    y_test_prob = model.predict(X_test, verbose=0)
    f1 = compute_f1_score(y_test, y_test_prob)
    print(f"Test F1 Score: {f1:.4f}")


if __name__ == '__main__':
    main()
