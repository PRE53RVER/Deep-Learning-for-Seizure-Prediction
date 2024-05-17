# Deep Learning for Seizure Prediction


This project focuses on classifying electroencephalogram (EEG) signals using deep learning techniques. The goal is to build a convolutional neural network (CNN) model capable of distinguishing between different classes of EEG signals.

## Dataset

The project uses a private dataset containing EEG signals recorded from various subjects. Due to privacy concerns, the dataset cannot be shared publicly. However, the dataset is expected to be in CSV format, with each row representing a sample of EEG data, and the last column indicating the class label.

## Prerequisites

- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

The project consists of a single Python script that handles data preprocessing, model creation, training, and evaluation.

```
├── best_model.h5 (Saved model weights)
└── ipython file
```

## Usage

1. Install the required dependencies:

```
pip install tensorflow pandas numpy matplotlib scikit-learn
```

2. Run the `main.py` script:

```
run the notebook
```

The script will perform the following steps:

1. Load the EEG dataset from the CSV file.
2. Preprocess the data by scaling and splitting into training and testing sets.
3. Define the CNN model architecture.
4. Train the model on the training data and evaluate it on the testing data.
5. Save the best model weights based on the validation loss.

## Model Architecture

The CNN model architecture consists of the following layers:

- Convolutional layers with varying kernel sizes and strides
- Batch normalization layers
- Dropout layers for regularization
- Dense layers with ReLU activation
- Output layer with softmax activation

The model is compiled with the Adam optimizer and categorical cross-entropy loss function. The top-k categorical accuracy, AUC, precision, and recall metrics are tracked during training.

## Results

The trained model's performance metrics, including accuracy, loss, and other evaluation metrics, will be printed to the console. Additionally, the best model weights will be saved in the `best_model.h5` file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

