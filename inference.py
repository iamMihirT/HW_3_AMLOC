from flask import Flask, request, jsonify
from flask_cors import CORS 
import torch
import torch.nn as nn
import numpy as np
import joblib
import yfinance as yf

app = Flask(__name__)
CORS(app)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load model and scaler
model = LSTM().to(device)
model.load_state_dict(torch.load('models/lstm_stock.pth', map_location=device))
model.eval()

scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', 'AAPL')
    df = yf.download(ticker, period='40d')
    data = df['Close'].values.reshape(-1, 1)

    if len(data) < 30:
        return jsonify({'error': 'Not enough data to make prediction.'})

    scaled_data = scaler.transform(data[-30:])
    X_input = torch.tensor(scaled_data.reshape(1, 30, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction = model(X_input)
        predicted_price = scaler.inverse_transform(prediction.cpu().numpy())[0][0]

    return jsonify({'ticker': ticker, 'predicted_close': float(predicted_price)})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
