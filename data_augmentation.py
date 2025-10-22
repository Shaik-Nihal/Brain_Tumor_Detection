"""
Data Augmentation for Brain Tumor Detection

This script augments the brain MRI image dataset to:
1. Generate more training samples
2. Balance the dataset between tumor and non-tumor classes

Usage: python data_augmentation.py
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
import time

def hms_string(sec_elapsed):
    """Format time in hours:minutes:seconds"""
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"

def augment_data(file_dir, n_generated_samples, save_to_dir):
    """
    Arguments:
        file_dir: A string representing the directory where images that we want to augment are found.
        n_generated_samples: A string representing the number of generated samples using the given image.
        save_to_dir: A string representing the directory in which the generated images will be saved.
    """
    
    # Create data generator
    data_gen = ImageDataGenerator(
        rotation_range=10, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        shear_range=0.1, 
        brightness_range=(0.3, 1.0),
        horizontal_flip=True, 
        vertical_flip=True, 
        fill_mode='nearest'
    )

    # Make sure the save_to_dir exists
    os.makedirs(save_to_dir, exist_ok=True)
    
    total_files = len(listdir(file_dir))
    print(f"Augmenting {total_files} images from {file_dir}...")
    processed = 0
    
    # Process each image in the directory
    for filename in listdir(file_dir):
        try:
            # load the image
            image_path = os.path.join(file_dir, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            # reshape the image
            image = image.reshape((1,) + image.shape)
            # prefix of the names for the generated samples
            save_prefix = 'aug_' + filename[:-4]
            # generate 'n_generated_samples' sample images
            i = 0
            for batch in data_gen.flow(
                x=image, 
                batch_size=1, 
                save_to_dir=save_to_dir,
                save_prefix=save_prefix, 
                save_format='jpg'
            ):
                i += 1
                if i >= n_generated_samples:
                    break
            
            # Also copy the original image to the augmented folder
            cv2.imwrite(os.path.join(save_to_dir, filename), cv2.imread(image_path))
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{total_files} images")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Augmentation completed for {file_dir}")

def data_summary(main_path):
    """
    Print a summary of the dataset
    
    Args:
        main_path: Base path containing 'yes' and 'no' subdirectories
    """
    yes_path = os.path.join(main_path, 'yes')
    no_path = os.path.join(main_path, 'no')
    
    if not (os.path.exists(yes_path) and os.path.exists(no_path)):
        print(f"Error: Could not find directories at {yes_path} and {no_path}")
        return
        
    # Number of files (images) in each folder
    m_pos = len(listdir(yes_path))
    m_neg = len(listdir(no_path))
    m = (m_pos + m_neg)
    
    pos_prec = (m_pos * 100.0) / m
    neg_prec = (m_neg * 100.0) / m
    
    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec:.1f}%, number of pos examples: {m_pos}") 
    print(f"Percentage of negative examples: {neg_prec:.1f}%, number of neg examples: {m_neg}") 

def main():
    """Main function to run the data augmentation process"""
    
    print("Brain Tumor Detection - Data Augmentation")
    print("----------------------------------------")
    
    # Define paths
    yes_path = 'yes'
    no_path = 'no'
    augmented_data_path = 'augmented data'
    
    # Create output directories if they don't exist
    os.makedirs(os.path.join(augmented_data_path, 'yes'), exist_ok=True)
    os.makedirs(os.path.join(augmented_data_path, 'no'), exist_ok=True)
    
    # Check if original data exists
    if not (os.path.exists(yes_path) and os.path.exists(no_path)):
        print(f"Error: Could not find original dataset directories!")
        return
    
    # Print summary of original dataset
    print("\nOriginal Dataset Summary:")
    data_summary('.')
    
    # Perform data augmentation
    print("\nStarting data augmentation...")
    start_time = time.time()
    
    # Augment data for the examples with label equal to 'yes' (tumorous)
    augment_data(
        file_dir=yes_path,
        n_generated_samples=6,
        save_to_dir=os.path.join(augmented_data_path, 'yes')
    )
    
    # Augment data for the examples with label equal to 'no' (non-tumorous)
    augment_data(
        file_dir=no_path,
        n_generated_samples=9,
        save_to_dir=os.path.join(augmented_data_path, 'no')
    )
    
    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"\nData augmentation completed in: {hms_string(execution_time)}")
    
    # Print summary of augmented dataset
    print("\nAugmented Dataset Summary:")
    data_summary(augmented_data_path)
    
    print("\nData augmentation completed successfully!")
    print("You can now use the augmented dataset to train the model.")

if __name__ == "__main__":
    main()