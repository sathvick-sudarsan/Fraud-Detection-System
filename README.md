# Shipping Fraud Detection System

A comprehensive machine learning system for detecting fraudulent shipping activities, designed for logistics and e-commerce companies like FedEx.

## Project Overview

This project implements an end-to-end shipping fraud detection system that:

1. Processes shipping transaction data
2. Engineers relevant features for fraud detection
3. Trains and evaluates multiple machine learning models
4. Implements graph-based analysis to detect fraud rings
5. Provides a production-ready API for real-time fraud detection

## Key Features

- **Advanced Data Processing**: Handles missing values, outliers, and imbalanced data
- **Feature Engineering**: Creates 30+ features from raw shipping data
- **Graph Analysis**: Detects fraud rings and suspicious patterns using network analysis
- **Ensemble Models**: Combines XGBoost, Random Forest, and other algorithms
- **Real-time API**: Flask-based API for integration with shipping systems
- **Comprehensive Testing**: Unit and integration tests for all components
- **Detailed Documentation**: Jupyter notebooks and code documentation

## Project Structure

```
Shipping-Fraud-Detection/
├── data/
│   └── raw/                  # Raw shipping transaction data
├── models/                   # Saved trained models
├── notebooks/                # Jupyter notebooks for exploration and demos
├── results/                  # Evaluation results and visualizations
├── src/
│   ├── api/                  # Flask API for fraud detection
│   ├── data/                 # Data loading and preprocessing
│   ├── features/             # Feature engineering
│   └── models/               # ML models and graph analysis
├── tests/                    # Unit and integration tests
├── Detection.py              # Main script to run the pipeline
└── README.md                 # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/sathvick-sudarsan/Fraud-Detection-System.git
cd Fraud-Detection-System

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

```bash
python Detection.py
```

### Using the Demo Script

```bash
cd notebooks
python fraud_detection_demo.py
```

### API Usage

Start the API server:

```bash
python src/api/app.py
```

Make a prediction:

```python
import requests
import json

# Sample shipment data
shipment_data = {
    'ShipmentID': 1000001,
    'ShipmentTimestamp': 86400,
    'ShipmentValue': 500.0,
    'ShipmentType': 'W',
    'SenderID': 12345,
    # ... other fields
}

# Send prediction request
response = requests.post(
    'http://localhost:5000/predict',
    json=shipment_data
)

# Get prediction result
result = response.json()
print(f"Fraud Probability: {result['fraud_probability']}")
print(f"Risk Level: {result['risk_level']}")
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.9542 | 0.9231 | 0.8976 | 0.9102 | 0.9812 |
| Random Forest | 0.9487 | 0.9103 | 0.8854 | 0.8977 | 0.9756 |
| Ensemble | 0.9578 | 0.9312 | 0.9021 | 0.9164 | 0.9834 |

## Graph Analysis

The system uses graph-based analysis to:

- Identify connections between shipping transactions
- Detect fraud rings and coordinated fraud attempts
- Calculate risk scores for entities (senders, recipients, addresses)
- Visualize suspicious patterns

## Future Improvements

- Implement deep learning models for sequence analysis
- Add real-time monitoring and alerting
- Integrate with external data sources for enhanced detection
- Deploy as a containerized microservice

## License

MIT

## Contact

For questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com). 