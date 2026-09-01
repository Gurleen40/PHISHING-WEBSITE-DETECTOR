**# 🎣 Phishing URL Detector**



**A machine learning web application to detect phishing URLs.**



**## Project Overview**



**This project predicts whether a URL is \*\*phishing\*\* or \*\*legitimate\*\* using a Random Forest machine learning model trained on 20 URL features.**



**## Project Structure**



**phishing-url-detector/**

**├── phishing\_model.py              # Backend (Flask API)**

**├── index\_ph.html                  # Frontend (Web Interface)**

**├── Random\_Forest.pk               # Trained ML Model**

**├── scaler.pk                      # Feature Scaler**

**├── url.csv                        # Training Dataset**

**├── Phishing.ipynb                 # Model Training Notebook**

**├── requirements.txt               # Python Dependencies**

**├── .gitignore                     # Git Ignore File**

**└── README.md                      # This File**



**## How to Use Locally**



**### 1. Clone the repository**

**git clone https://github.com/YOUR-USERNAME/phishing-detector.git**

**cd phishing-detector**



**### 2. Install dependencies**

**pip install -r requirements.txt**



**### 3. Run the backend**

**python phishing\_model.py**



**The backend will run on http://localhost:5000**



**### 4. Open the frontend**

**- Open index\_ph.html in your browser**

**- Update the BACKEND\_URL in the JavaScript code to http://localhost:5000**



**## API Endpoint**



**POST /predict**



**Send 20 URL features and get a prediction: "Phishing" or "Legitimate"**



**## Features Used**



**- URLLength**

**- DomainLength**

**- IsDomainIP**

**- CharContinuationRate**

**- TLDLegitimateProb**

**- URLCharProb**

**- TLDLength**

**- NoOfSubDomain**

**- HasObfuscation**

**- NoOfObfuscatedChar**

**- ObfuscationRatio**

**- NoOfLettersInURL**

**- LetterRatioInURL**

**- NoOfDegitsInURL**

**- DegitRatioInURL**

**- NoOfEqualsInURL**

**- NoOfQMarkInURL**

**- NoOfAmpersandInURL**

**- NoOfOtherSpecialCharsInURL**

**- SpacialCharRatioInURL**



**## Deployment**



**This app is deployed on Render: \[INSERT YOUR RENDER URL HERE]**



**Live Demo URL: \[INSERT YOUR FRONTEND URL HERE]**



**## Technologies**



**- Backend: Flask (Python)**

**- ML Model: scikit-learn (Random Forest)**

**- Frontend: HTML, CSS, JavaScript**

**- Deployment: Render**



**## License**



**MIT License**



**---**



**Author: \[Your Name]**

**GitHub: https://github.com/YOUR-USERNAME**

