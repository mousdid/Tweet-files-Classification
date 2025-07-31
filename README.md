# Tweet-files Gender Classification (CS585 Project)

**What This Is:**  
A machine learning pipeline for classifying gender (male/female) based on tweet text. Developed as part of a course project to experiment with classical NLP-based text classification methods on a balanced tweet dataset.

**Dataset Used:**  
- [Tweet-files for Gender Guessing (Kaggle)](https://www.kaggle.com/datasets/aharless/tweet-files-for-gender-guessing)  
- 55,510 tweets (balanced: 27,755 male / 27,755 female)  
- Includes tweet text and gender labels based on user profiles

**Approach Taken:**  
- Preprocessed tweets using:
  - HTML decoding
  - Lowercasing
  - Emoji, email, and mention token replacement
  - Tokenization
- Converted tweets into Bag-of-Words features
- Implemented Naïve Bayes from scratch  
- Also tested Logistic Regression using `scikit-learn`
- Designed the pipeline to be modular and extensible

**Evaluation & Results:**  
- Performance plateau observed despite data size  
- Achieved a tie on F₁-score between both models  
- Accuracy, Precision, Recall, and F-score used for evaluation  
- Found that gender cues in tweets are subtle and sparse

**What I Learned:**  
- Implementing Naïve Bayes from scratch offered deep insights into probabilistic models  
- Preprocessing and vectorization choices significantly impact results  
- Text classification is highly sensitive to vocabulary quality and data noise  
- Designing modular, extensible code improves reproducibility and experimentation

**Tools Used:**  
Python, NumPy, Pandas, scikit-learn, Matplotlib

---

*Note: This project was completed in April 2025 as part of the CS585 course at Illinois Tech.*
