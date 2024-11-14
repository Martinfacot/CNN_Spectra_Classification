# CNN for NMR Spectra Classification Project

## Project Description
This project implements a Convolutional Neural Network (CNN) for the classification of ¹H-NMR spectra specifically focused on metabolic spectra. The CNN is designed to analyze and categorize spectral images, providing an efficient and accurate method for spectra classification.

## Main Features
1. Data Loading: Efficient loading of spectral image data.
2. Preprocessing: Image preprocessing techniques to enhance features and normalize data.
3. CNN Model: Implementation of a custom CNN architecture for spectra classification.
4. Training Pipeline: A robust training process with validation and performance metrics.
5. Prediction: Ability to classify new spectral data using the trained model.

## Technologies and Frameworks
- Python: Primary programming language
- TensorFlow/Keras: For building and training the CNN model
- NumPy: For numerical computations and array operations
- Matplotlib: For data visualization
- YAML: Used for configuration management

Why YAML?
Using YAML files saves time and simplifies code modifications by allowing you to change configuration settings without altering or rerunning your entire codebase

## Project Structure
1. CNN :
   - Alanine_model.ipynb --> main file to run the CNN
   - config_alanine.yml --> config preprocess parameters and model versions
   - test_your_saved_model.ipynb --> input your new images you want to test OR use the Sofware

2. Old_YAML_&_Model :  if you are curious..

3. Sofware : created with Tkinter 
--> spectra_analysis_app.py
(update on going)

## Workflow Overview (related to create_your_model)

### Step 1: Set Up Data Folder
1. **Create a root folder** to store all your data.
Example: your\path\here\data

2. **For each metabolite**, create a folder named after the metabolite with two subfolders inside:

- valid: To store valid data files.
- invalid: To store invalid data files.

**Example Structure:**
- 'your\path\here\data\Glutamine\valid'
- 'your\path\here\data\Glutamine\invalid'


### Step 2: Set Up Models Folder
1. **Create a root folder** to store all your models.  
Example: `your\path\here\models`

2. **For each metabolite**, create a folder named after the metabolite to store model files.

**Example Structure:**
` your\path\here\models\Glutamine'

### Important
- **YAML File:** Place the YAML configuration file in the same directory as your data and model folders for consistent versioning and configuration.
 

## Dataset
I am currently working on providing the dataset used for this project. Due to some necessary adjustments and privacy concerns, the dataset is not publicly available at the moment. I aim to make a sample dataset or a processed version available in the near future to aid in understanding and replicating the project results.

For any questions or if you need more information about the data used, please feel free to open an issue in this repository.
