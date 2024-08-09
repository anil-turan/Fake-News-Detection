import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import pickle

class FakeNewsDetector(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        # Load the model and vectorizer
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
    def initUI(self):
        layout = QVBoxLayout()
        
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText('Enter the news text here')
        layout.addWidget(self.text_input)
        
        self.button = QPushButton('Check')
        self.button.clicked.connect(self.check_news)
        layout.addWidget(self.button)
        
        self.result_label = QLabel('', self)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.result_icon = QLabel('', self)
        self.result_icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_icon)
        
        self.setLayout(layout)
        self.setWindowTitle('Fake News Detection')
        self.setGeometry(100, 100, 600, 400)  # Set the window size
        self.show()
    
    def check_news(self):
        text = self.text_input.text()
        if text:
            # Transform the text using the vectorizer
            text_tfidf = self.vectorizer.transform([text])
            prediction = self.model.predict(text_tfidf)
            
            if prediction[0] == 1:
                self.result_label.setText('Real News')
                pixmap = QPixmap('path/to/real_icon.png')  # Update 'path/to/real_icon.png' to the actual file path
                pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio)  # Resize the icon
                self.result_icon.setPixmap(pixmap)
            else:
                self.result_label.setText('Fake News')
                pixmap = QPixmap('path/to/fake_icon.png')  # Update 'path/to/fake_icon.png' to the actual file path
                pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio)  # Resize the icon
                self.result_icon.setPixmap(pixmap)
        else:
            self.result_label.setText('Please enter some text')
            self.result_icon.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FakeNewsDetector()
    sys.exit(app.exec_())
