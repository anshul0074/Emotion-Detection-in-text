This project demonstrates emotion classification using BERT (Bidirectional Encoder Representations from Transformers) for text data. It involves preprocessing the data, tokenization using a pre-trained BERT model, creating a classification model architecture, training the model, and evaluating its performance.

Getting Started:

To run this project, follow the steps below:

Prerequisites:

Python 3.x
Jupyter Notebook or any Python IDE

Usage:

Open the Emotion.ipynb notebook in Jupyter Notebook or any Python IDE.
Execute the cells in sequential order to run the project.

Data:

The dataset used for this project is divided into three files: train.txt, test.txt, and val.txt. These files contain labeled text data for emotion classification. The dataset is combined into a dataframe and preprocessed before training the model.

Preprocessing:

The preprocess function in the notebook applies various preprocessing steps to the input data, including cleaning HTML tags, removing digits, removing stopwords, removing links, removing special characters, converting to lowercase, and more. The preprocessed data is stored in the preprocessed_df dataframe.

Tokenization:

The text data is tokenized using the BERT tokenizer from the Hugging Face transformers library. The tokenizer converts the input text into tokenized sequences suitable for input to the BERT model. The tokenization parameters such as maximum sequence length are set, and the tokenized data is stored in X_train and X_test tensors.

Model Architecture:

The classification model architecture is created using the TensorFlow Keras API. It consists of an input layer, BERT embedding layer, pooling layer, dense layers, and a softmax output layer. The model is compiled with the Adam optimizer and Categorical Cross-Entropy loss function.

Training:

The model is trained on the preprocessed and tokenized data. The training is performed for a specified number of epochs and with a batch size of 32. The model's performance on the test set is evaluated after training.

Performance Analysis:

The model's performance is analyzed by plotting the accuracy graph during training and printing the classification report. Additionally, the trained model is saved as emotion_detector.h5.

Results:

The final results include the test loss, test accuracy, and the classification report showing precision, recall, and F1-score ( more than 75% for all cases) for each emotion class.


