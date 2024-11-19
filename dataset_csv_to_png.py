import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def create_balanced_dataset(data_path, normal_samples=1000, abnormal_samples=1000, random_state=42):
    """
    Creates a balanced dataset of normal and abnormal ECGs
    """
    df = pd.read_csv(data_path)
    normal_data = df[df.iloc[:, -1] == 0]
    abnormal_data = df[df.iloc[:, -1] != 0]
    
    normal_sample = normal_data.sample(n=normal_samples, random_state=random_state)
    abnormal_sample = abnormal_data.sample(n=abnormal_samples, random_state=random_state)
    
    balanced_dataset = pd.concat([normal_sample, abnormal_sample])
    return balanced_dataset

def create_ecg_image(signal, output_path, index, label):
    """
    Creates an ECG image with the specified style
    """

    fig = plt.figure(figsize=(8, 6))
    
    file_name = f'ECG_{index:04d}_{label}'
    plt.title(file_name, pad=20)
    
    plt.plot(signal[:-1], color='blue', linewidth=1)
    
     
    # Adjusting axis limits with padding
    x_padding = len(signal) * 0.05  # 5% padding
    y_min, y_max = min(signal[:-1]), max(signal[:-1])
    y_padding = (y_max - y_min) * 0.1  # 10% padding
    
    plt.xlim(-x_padding, len(signal) + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    
    
    plt.savefig(output_path + '/' + file_name, 
                bbox_inches='tight',
                dpi=100)
    plt.close()

def process_dataset(data_path, output_base_path):

    output_dirs = {
        'normal': os.path.join(output_base_path, 'normal'),
        'abnormal': os.path.join(output_base_path, 'abnormal')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    balanced_data = create_balanced_dataset(data_path)
    
    normal_count = 1
    abnormal_count = 1
    

    for idx, row in tqdm(enumerate(balanced_data.values), total=len(balanced_data),
                        desc="Generating ECG images"):
        signal = row[:-1]
        label = 'normal' if row[-1] == 0 else 'abnormal'
        
        if label == 'normal':
            create_ecg_image(signal, output_dirs[label], normal_count, label)
            normal_count += 1
        else:
            create_ecg_image(signal, output_dirs[label], abnormal_count, label)
            abnormal_count += 1

if __name__ == "__main__":
    input_path = r'C:\Users\Biosfer\Desktop\kaggle_dataset_ECG\data_csv\mitbih_train.csv' # change your path here
    output_base_path = r'C:\Users\Biosfer\Desktop\kaggle_dataset_ECG\images' # change your path here
    
    process_dataset(input_path, output_base_path)
    print("\nImage generation complete!")
    print(f"Images generated in  : {output_base_path}")