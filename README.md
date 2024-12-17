# Topic Modeling (Text Classification)

## Project Overview  
The **Topic Modeling (Text Classification)** project focuses on building a system to classify news articles into different categories. The system classifies articles into predefined topics such as **Politics**, **Health**, **Emotion**, **Financial**, **Sport**, and **Science**. This project demonstrates the use of advanced deep learning techniques to automatically categorize text data, making it a useful tool for news aggregation and content analysis.

## Key Features  
- Classifies news articles into multiple categories such as Politics, Health, Emotion, Financial, Sport, and Science.  
- Uses Bidirectional LSTM (Long Short-Term Memory) networks to process textual data.  
- Enhances model performance with an Attention Mechanism.  
- Pretrained word embeddings from **GloVe** for better representation of text data.  
- High accuracy in topic classification for real-world use cases.

## Technologies Used  
- **Python**: For the backend logic and data processing.  
- **TensorFlow**: For building, training, and deploying the deep learning model.  
- **Bidirectional LSTM**: For capturing both past and future contexts in text sequences.  
- **Attention Mechanism**: To enhance the focus on important parts of the text.  
- **GloVe Pretrained Embeddings**: To provide semantic representation of words in the text.  

## How It Works  
1. **Data Preprocessing**: The raw news articles are processed, tokenized, and transformed into sequences of words using GloVe pretrained word embeddings.  
2. **Model Architecture**: A Bidirectional LSTM network is used to process text sequences, with an attention mechanism that highlights key portions of the text.  
3. **Classification**: The model classifies the processed text into one of the predefined categories (Politics, Health, Emotion, Financial, Sport, Science).  
4. **Model Evaluation**: The system's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score, ensuring high classification accuracy.

## Files Included  
- **`my_model.keras`**: The trained Keras model for text classification.  
- **`tokenizer.pickle`**: Tokenizer used to preprocess the input text and convert it into sequences.  
- **`ixtoword.pickle`**: Mapping of integer indices to words (used for reverse lookup).  
- **`wordtoix.pickle`**: Mapping of words to integer indices (used for tokenization).  
- **`text-classification.ipynb`**: Jupyter notebook used for training and testing the classification model. This file contains the full pipeline of data preprocessing, model training, and evaluation.  

## Installation and Setup  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/Topic-Modeling.git  
   cd Topic-Modeling
2. Load the pre-trained model and tokenizer files:
   The pre-trained model and tokenization files (my_model.keras, tokenizer.pickle, ixtoword.pickle, wordtoix.pickle) are included in the repository and can be loaded directly in the code
3. Run the classification system:
   ```bash
   text-classification.ipynb
## Example Interaction and Results  
### **Example Input:**  
A user submits a news article with the headline "Global Health Crisis: Experts Discuss the Future of Healthcare."

### **Example Output:**  
The article is classified into the "Health" category with high confidence.
