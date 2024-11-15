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
- Python: Primary programming language -> version==3.12.4
- TensorFlow/Keras: For building and training the CNN model
- NumPy: For numerical computations and array operations
- Matplotlib: For data visualization
- YAML: Used for configuration management
- Tkinte: sofware developement

Why YAML?
Using YAML files saves time and simplifies code modifications by allowing you to change configuration settings without altering or rerunning your entire codebase


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

### Important Note About Notebook Placement
The **create_your_model.ipynb** notebook must be placed in the specific metabolite folder within the models directory (e.g., your\path\here\models\Glutamine\create_your_model.ipynb).
This placement is crucial as the get_metabolite_name() function relies on the folder structure to automatically detect which metabolite you're working with.

##### Example Correct Placement:
models/
└── Glutamine/
    └── create_your_model.ipynb
This ensures that the notebook can automatically identify the metabolite name from its location in the folder structure, making the workflow more automated and less prone to errors.

### YAML Configuration File Setup
Each metabolite folder must contain a YAML configuration file named config_[metabolite].yml (e.g., config_glutamine.yml). 
This file must be placed in the same directory as the notebook.

#### File Naming Convention:

- The file must be named in lowercase
- Format: config_[metabolite_name].yml
- Example: config_glutamine.yml

##### Example Directory Structure:
Copymodels/
└── Glutamine/
    ├── create_your_model.ipynb
    └── config_glutamine.yml
    
##### Version Selection:
In the notebook, you can select which version of the configuration to use by setting the version number:

**The load_config()** function will:
1. Automatically look for the YAML file in the same directory
2. Load the specified version's configuration
3. Raise an error if the specified version doesn't exist in the config file

Make sure your YAML file exists and contains the version number you specify in the notebook, otherwise you'll receive a "Version not found" error.

## Dataset
I am currently working on providing the dataset used for this project. Due to some necessary adjustments and privacy concerns, the dataset is not publicly available at the moment. I aim to make a sample dataset or a processed version available in the near future to aid in understanding and replicating the project results.

For any questions or if you need more information about the data used, please feel free to open an issue in this repository.
