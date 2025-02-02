# CNN for NMR Spectra Classification Project
![CNN architeture](https://github.com/Martinfacot/CNN_Spectra_Classification/blob/main/images/Global%20Architecture%20of%20the%20CNN.png)
## Table of Contents

1. [Project Description](#project-description)
2. [Main Features](#main-features)
3. [Technologies and Frameworks](#technologies-and-frameworks)
4. [Workflow Overview](#workflow-overview-related-to-create_your_model)
   - [Step 1: Set Up Data Folder](#step-1-set-up-data-folder)
   - [Step 2: Set Up Models Folder](#step-2-set-up-models-folder)
   - [Important Note About Notebook & YAML file Placement](#important-note-about-notebook--yaml-file-placement)
5. [Dataset](#dataset)
   - [Dataset Characteristics](#dataset-characteristics)
   - [Dataset Selection](#dataset-selection)
6. [Testing the Project](#testing-the-project)
   - [Setup and Installation](#setup-and-installation)
   - [Troubleshooting](#troubleshooting)
7. [Results](#results)
8. [How the Software Works](#how-the-software-works-in-developement)
   - [Data Preparation](#data-preparation)
   - [Image Processing and Classification](#image-processing-and-classification)
   - [User Interface Features](#user-interface-features)

## Project Description
This project implements a Convolutional Neural Network (CNN) for the classification of ¹H-NMR spectra specifically focused on metabolic spectra. The CNN is designed to analyze and categorize spectral images, providing an efficient and accurate method for spectra classification.
The primary motivation behind this project is to create an automatic classification of spectra in order to reduce human error and subjective interpretation in the analysis of metabolite images.

## Main Features
1. Data Loading: Efficient loading of spectral image data.
2. Preprocessing: Image preprocessing techniques to enhance features and normalize data.
3. CNN Model: Implementation of a custom CNN architecture for spectra classification.
4. Training Pipeline: A robust training process with validation and performance metrics.
5. Prediction: Ability to classify new spectral data using the trained model.
6. Software (in developement)

## Technologies and Frameworks
- Python: Primary programming language -> version==3.12.4
- TensorFlow/Keras: For building and training the CNN model
- NumPy: For numerical computations and array operations
- Matplotlib: For data visualization
- YAML: Used for configuration management
- Tkinter: software developement

Why YAML?
Using YAML files saves time and simplifies code modifications by allowing you to change configuration settings without altering or rerunning your entire codebase


## Workflow Overview (related to create_your_model)


### Step 1: Set Up Data Folder
1. **Create a root folder** to store all your data.
Example: `your\path\here\data`

2. **For each metabolite**, create a folder named after the metabolite with two subfolders inside:

- valid: To store valid data files.
- invalid: To store invalid data files.

**Example Structure:**
- `your\path\here\data\Glutamine\valid`
- `your\path\here\data\Glutamine\invalid`


### Step 2: Set Up Models Folder
1. **Create a root folder** to store all your models.  
Example: `your\path\here\models`

2. **For each metabolite**, create a folder named after the metabolite to store model files.

**Example Structure:**
`your\path\here\models\Glutamine`


### Important Note About Notebook & YAML file Placement 
The **create_your_model.ipynb & config_[metabolite_name].yml** must be placed in the specific metabolite folder within the models directory (e.g., your\path\here\models\Glutamine\create_your_model.ipynb).
This placement is crucial as the get_metabolite_name() function relies on the folder structure to automatically detect which metabolite you're working with.
Each metabolite folder requires a YAML configuration file that is dynamically named to match the metabolite directory.


**Example Correct Placement:**
`Test_the_project/models/
                     └── Glutamine/
                            └── create_your_model.ipynb
                            └── config_[metabolite_name].yml`
    
This ensures that the notebook can automatically identify the metabolite name from its location in the folder structure, making the workflow more automated and less prone to errors.



##### Version Selection:
In the notebook, you can select which version of the configuration to use by setting the version number:

**The load_config()** function will:
1. Automatically look for the YAML file in the same directory
2. Load the specified version's configuration
3. Raise an error if the specified version doesn't exist in the config file

Make sure your YAML file exists and contains the version number you specify in the notebook, otherwise you'll receive a `Version not found` error.


## Dataset

Due to confidentiality constraints, I create a custom-generated dataset of ECG spectral images that mimic the structural characteristics of the original NMR spectra.

### Dataset Characteristics
Image Type: ECG spectral images
Purpose: Simulate NMR spectra classification workflow

### Dataset Selection
The original Heartbeat Dataset contains approximately 100,000 ECG recordings. For this project, I randomly selected:
- 1,000 Normal ECG images
- 1,000 Abnormal ECG images

This balanced subset provides a representative sample for developing and testing the classification model while maintaining computational efficiency.

The dataset images were created using the following source:
- Original Dataset: [Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data)
- Data Conversion Script: `dataset_csv_to_png.py`

[View Dataset Folder](https://github.com/Martinfacot/CNN_Spectra_Classification/tree/main/Test_the_project/data)

## Testing the Project

### Setup and Installation

1. Clone this repository to your local machine:
   
   ```bash
   git clone https://github.com/Martinfacot/CNN_Spectra_Classification.git
   ```

3. Navigate to the project directory:
   
 ```bash
cd CNN_Spectra_Classification
```

3. Create and activate a virtual environment (optional but recommended):

 ```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required Python packages:

 ```bash
pip3 install -r requirements.txt
 ```

5. Unzip the image folders and ensure the following directory structure:
- `your/path/here/Test_the_project/data/ECG/normal/`
- `your/path/here/Test_the_project/data/ECG/abnormal/`

6. Open and run the Jupyter Notebook located at:
- `your/path/here/Test_the_project/models/ECG/create_your_model_ECG.ipynb`

### Troubleshooting

- Ensure all dependencies are correctly installed
- Verify the exact Python version (3.12)
- Check that image folders are correctly unzipped and placed in the specified directories


## Results

If you want to explore the model performance without retraining, you can examine the detailed results in the following Excel files:

1. **Alanine Model Results** 
  - Example of a simple metabolite classification
  - [View Alanine Results](https://github.com/Martinfacot/CNN_Spectra_Classification/blob/main/models_results_Alanine.xlsx)

2. **ECG Model Results**
  - Results from the test project using ECG data
  - [View ECG Results](https://github.com/Martinfacot/CNN_Spectra_Classification/blob/main/Test_the_project/models/ECG/models_results_ECG.xlsx)

These files contain comprehensive metrics and performance evaluations for their respective models.

## How the Software Works
The software implements a comprehensive workflow for metabolite spectra classification (the final version cannot be provided due to confidentiality policies):

### Data Preparation
- Users select a patient folder containing metabolite spectra images
- The application automatically scans specific subdirectories for metabolite images
- Supports multiple metabolites including Alanine, 3-HB, Acetone, Glutamine, and others

### Image Processing and Classification
1. **Image Preprocessing**
   - Crops and resizes images to standardized dimensions
   - Normalizes image data for consistent analysis
   - Prepares images for neural network input

2. **Machine Learning Classification**
   - Utilizes pre-trained Convolutional Neural Network (CNN) models for each metabolite
   - Generates probability scores for image classification
   - Provides confidence levels: High, Medium, and Low confidence classifications

### User Interface Features
- Interactive image display with probability visualization
- Manual classification validation
- Results export to Excel for further analysis
- Overview page for comprehensive results review

