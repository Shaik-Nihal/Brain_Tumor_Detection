"""
Brain Tumor Detection using a Convolutional Neural Network

This script performs the following tasks:
1. Loads and preprocesses brain MRI images
2. Builds and trains a CNN model for tumor detection
3. Evaluates the model performance
4. Allows prediction on new images

Usage: python brain_tumor_detection.py [--train] [--predict path_to_image]
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from os import listdir
import argparse

# Define constants
IMG_WIDTH, IMG_HEIGHT = (240, 240)
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
BEST_MODEL_PATH = 'models/brain_tumor_model.keras'

def crop_brain_contour(image, plot=False):
    """
    Crop the brain portion from an MRI image
    
    Args:
        image: The input MRI image
        plot: Whether to plot the original and cropped image
        
    Returns:
        The cropped brain image
    """
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return image
    
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Crop new image out of the original image using the four extreme points
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Image')
        
        plt.show()
    
    return new_image

def load_data(dir_list, image_size):
    """
    Read images, resize and normalize them. 
    
    Args:
        dir_list: list of strings representing file directories.
        image_size: tuple of (width, height)
        
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in listdir(directory):
            try:
                # load the image
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                
                # crop the brain and ignore the unnecessary rest part of the image
                image = crop_brain_contour(image, plot=False)
                # resize image
                image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                # normalize values
                image = image / 255.
                # convert image to numpy array and append it to X
                X.append(image)
                # append a value of 1 to the target array if the image
                # is in the folder named 'yes', otherwise append 0.
                if directory.endswith('yes'):
                    y.append([1])
                else:
                    y.append([0])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Splits data into training, development and test sets.
    
    Args:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
        test_size: Fraction of data to reserve for testing
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: Split datasets
    """
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_model(input_shape):
    """
    Build a CNN model for brain tumor detection
    
    Args:
        input_shape: A tuple representing the shape of the input of the model (image_width, image_height, #_channels)
        
    Returns:
        model: A Keras Model object
    """
    # Define the input placeholder as a tensor with shape input_shape. 
    X_input = Input(input_shape) # shape=(?, 240, 240, 3)
    
    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input) # shape=(?, 244, 244, 3)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)
    
    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool0')(X) # shape=(?, 59, 59, 32) 
    
    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool1')(X) # shape=(?, 14, 14, 32)
    
    # FLATTEN X 
    X = Flatten()(X) # shape=(?, 6272)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X) # shape=(?, 1)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')
    
    return model

def hms_string(sec_elapsed):
    """Format time in hours:minutes:seconds"""
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"

def compute_f1_score(y_true, prob):
    """Compute the F1 score"""
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)
    
    score = f1_score(y_true, y_pred)
    
    return score

def plot_metrics(history):
    """Plot training metrics"""
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

def prepare_single_image(image_path):
    """
    Prepare a single image for prediction
    
    Args:
        image_path: Path to the image file
        
    Returns:
        A numpy array ready for model prediction
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Show original image
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.show()
        
        # Crop brain contour
        image = crop_brain_contour(image, plot=True)
        
        # Resize and normalize
        image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        image = image / 255.
        
        # Convert to array and add batch dimension
        image_array = np.array([image])
        
        return image_array
        
    except Exception as e:
        print(f"Error preparing image: {e}")
        return None

def predict_tumor(model, image_array):
    """
    Predict whether an image contains a tumor
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        
    Returns:
        prediction: 0 (no tumor) or 1 (tumor)
        probability: Probability of the prediction
    """
    # Get prediction probability
    pred_prob = model.predict(image_array)[0][0]
    
    # Convert to binary prediction
    prediction = 1 if pred_prob > 0.5 else 0
    
    return prediction, pred_prob

def train_model():
    """Train the brain tumor detection model"""
    print("Starting model training...")
    
    # Define paths
    augmented_path = 'augmented data/'
    augmented_yes = os.path.join(augmented_path, 'yes')
    augmented_no = os.path.join(augmented_path, 'no')
    
    # Check if augmented data exists
    if not (os.path.exists(augmented_yes) and os.path.exists(augmented_no)):
        print("Augmented data not found! Using original dataset...")
        yes_path = 'yes'
        no_path = 'no'
        
        if not (os.path.exists(yes_path) and os.path.exists(no_path)):
            print("Error: Could not find dataset directories!")
            return
    else:
        yes_path = augmented_yes
        no_path = augmented_no
    
    # Load data
    print("Loading data...")
    X, y = load_data([yes_path, no_path], (IMG_WIDTH, IMG_HEIGHT))
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
    
    print(f"Number of training examples: {X_train.shape[0]}")
    print(f"Number of validation examples: {X_val.shape[0]}")
    print(f"Number of test examples: {X_test.shape[0]}")
    
    # Build model
    print("Building model...")
    model = build_model(IMG_SHAPE)
    model.summary()
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create model directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Set up callbacks
    log_dir = os.path.join('logs', f'brain_tumor_detection_cnn_{int(time.time())}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    tensorboard = TensorBoard(log_dir=log_dir)
    
    # Create checkpoint callback for the best model
    checkpoint = ModelCheckpoint(
        os.path.join('models', "brain_tumor_model.keras"),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    
    # Train for 24 epochs in total
    history = model.fit(
        x=X_train, 
        y=y_train, 
        batch_size=32,
        epochs=24,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, checkpoint]
    )
    
    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Training completed in {hms_string(execution_time)}")
    
    # Plot training history
    plot_metrics(history.history)
    
    # Evaluate the model on test data
    print("Evaluating model on test data...")
    loss, accuracy = model.evaluate(x=X_test, y=y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    # Calculate F1 score
    y_test_prob = model.predict(X_test)
    f1 = compute_f1_score(y_test, y_test_prob)
    print(f"Test F1 Score: {f1}")
    
    print("Training completed!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Brain Tumor Detection')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    
    args = parser.parse_args()
    
    if args.train:
        # Train the model
        train_model()
    
    elif args.predict:
        # Make sure the model file exists
        if not os.path.exists(BEST_MODEL_PATH):
            print(f"Error: Model file not found at {BEST_MODEL_PATH}")
            print("Please train the model first or make sure the model file exists.")
            return
        
        # Load the model
        print("Loading model...")
        model = load_model(BEST_MODEL_PATH)
        
        # Prepare the image
        print(f"Preparing image: {args.predict}")
        image_array = prepare_single_image(args.predict)
        
        if image_array is not None:
            # Make prediction
            prediction, probability = predict_tumor(model, image_array)
            
            # Display results
            result = "Tumor Detected!" if prediction == 1 else "No Tumor Detected"
            print(f"\nResult: {result}")
            print(f"Probability: {probability:.4f}")
            
            plt.figure(figsize=(10, 5))
            plt.imshow(image_array[0])
            plt.title(f"Prediction: {result} (Probability: {probability:.4f})")
            plt.axis('off')
            plt.show()
        
    else:
        # Default action if no arguments provided
        print("Brain Tumor Detection System")
        print("-----------------------------")
        print("Usage: python brain_tumor_detection.py [--train] [--predict path_to_image]")
        print("  --train: Train the model")
        print("  --predict: Make a prediction on an image")
        print("\nExample: python brain_tumor_detection.py --predict yes/Y1.jpg")

if __name__ == "__main__":
    main()