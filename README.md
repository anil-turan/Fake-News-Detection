# Fake-News-Detection
The project includes a machine learning model and a PyQt5-based interface to detect fake news. Users can enter news text to check whether the text is fake or not. The model is a Logistic Regression model trained with datasets containing real and fake news.


Code Explanation

main.py

main.py creates a PyQt5 user interface for entering news text and checking whether it is fake or real using a pre-trained model.

Technologies Used:

PyQt5: For creating the graphical user interface (GUI).

Pickle: To load the pre-trained model and vectorizer.

train_and_save_model.py

train_and_save_model.py trains a Logistic Regression model to detect fake news using a dataset of fake and real news articles. It then saves the trained model and the TF-IDF vectorizer for later use.

Technologies Used:

Pandas: For data manipulation and analysis.

Scikit-learn: For machine learning, including training the model and transforming the text data.

Pickle: To save the trained model and vectorizer.
