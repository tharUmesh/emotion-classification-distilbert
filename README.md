# Emotion Classification with DistilBERT

This repository contains a Jupyter Notebook that fine-tunes a DistilBERT model to classify emotions in tweets using the [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion). The project demonstrates a complete NLP pipeline, including data preprocessing, model training, evaluation, and inference.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Results](#results)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

## Project Overview
This project focuses on emotion recognition in tweets, a key task in natural language processing (NLP) with applications in sentiment analysis, social media analytics, and mental health monitoring. Using the HuggingFace Transformers library, a DistilBERT model is fine-tuned to classify tweets into six emotions: sadness, joy, love, anger, fear, and surprise.

The notebook follows this workflow:
1. Load and explore the dair-ai/emotion dataset.
2. Preprocess the data using DistilBERT's tokenizer.
3. Fine-tune a DistilBERT model for sequence classification.
4. Evaluate the model using accuracy, F1-score, and a confusion matrix.
5. Perform inference on new text inputs.
6. Visualize attention weights for interpretability.

## Dataset
The [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) contains tweets labeled with six emotions:
- **Sadness**: 4666 samples
- **Joy**: 5362 samples
- **Love**: 1304 samples
- **Anger**: 2159 samples
- **Fear**: 1937 samples
- **Surprise**: 572 samples

The dataset is imbalanced, with joy and sadness being the most frequent emotions, which impacts model performance on underrepresented classes like surprise and love.

## Results
The fine-tuned DistilBERT model achieved the following on the test set:
- **Accuracy**: 92.7%
- **F1-Score**: 92.7% (weighted average)
- **Test Loss**: 0.1699

### Classification Report
| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Sadness   | 0.97      | 0.96   | 0.96     | 581     |
| Joy       | 0.96      | 0.94   | 0.95     | 695     |
| Love      | 0.79      | 0.91   | 0.85     | 159     |
| Anger     | 0.93      | 0.93   | 0.93     | 275     |
| Fear      | 0.87      | 0.90   | 0.89     | 224     |
| Surprise  | 0.75      | 0.68   | 0.71     | 66      |

The model performs best on sadness and joy due to their larger sample sizes, while surprise and love show lower performance due to dataset imbalance.

## Setup Instructions
To run the notebook locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/emotion-classification-distilbert.git
   cd emotion-classification-distilbert
    ```

2. **Install Dependencies**: Ensure Python 3.7+ is installed. Install the required libraries:   
    ```bash
    pip install transformers datasets accelerate scikit-learn pandas numpy seaborn matplotlib torch
    ```
3. **Run the Notebook**: Open Jupyter Notebook:
    ```bash
    jupyter notebook emotion_classification_distilbert.ipynb
    ```
Execute the cells sequentially. A GPU is recommended for faster training but is optional.

## Usage
The notebook (Emotion-Classification-Distilbert.ipynb) includes all code for data loading, preprocessing, training, evaluation, and inference.
To test the model on new text, use the inference section with inputs like:
```python   
    text = "I'm so happy to see you again!"
    input_encoded = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_encoded)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    print(classes[pred])  # Outputs: joy
```

## Future Work
Future improvements could include:
- Address dataset imbalance using oversampling, undersampling, or class weighting.
- Experiment with other transformer models (e.g., BERT, RoBERTa) for comparison.
- Optimize hyperparameters (e.g., learning rate, epochs) for better performance.
- Deploy the model as an API using services like xAI's API.

## References
- dair-ai/emotion dataset
- HuggingFace Transformers
- Vaswani, A., et al. (2017). "Attention is All You Need." Advances in Neural Information           Processing Systems.