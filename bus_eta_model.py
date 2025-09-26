"""
ML Model Prototype for Real-Time Bus ETA Prediction
Part 2: LSTM Model for Smart India Hackathon 2025
Team: Matthews - Travelo Project
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
import random
from typing import Dict, Tuple, Optional

# For production, uncomment these imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class BusETAPredictor:
    """
    LSTM-based ETA prediction model for public transport tracking
    """
    
    def __init__(self, model_path: str = "models/eta_model.h5"):
        """
        Initialize the ETA predictor
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False
        
        # Feature configuration
        self.feature_names = [
            'distance_km',
            'time_of_day',
            'day_of_week',
            'is_peak_hour',
            'weather_code',
            'traffic_density',
            'num_stops',
            'avg_stop_time'
        ]
        
    def generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic training data for prototype
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(42)
        
        data = []
        for _ in range(n_samples):
            # Generate realistic bus route data
            distance_km = np.random.uniform(2, 25)  # 2-25 km routes
            time_of_day = np.random.randint(0, 24)  # Hour of day
            day_of_week = np.random.randint(0, 7)   # Day of week
            
            # Peak hours: 8-10 AM and 5-7 PM
            is_peak_hour = 1 if time_of_day in [8, 9, 17, 18, 19] else 0
            
            # Weather: 0=Clear, 1=Rain, 2=Heavy Rain
            weather_code = np.random.choice([0, 0, 0, 1, 1, 2], p=[0.5, 0.2, 0.1, 0.1, 0.08, 0.02])
            
            # Traffic density (0-1 scale)
            traffic_base = 0.3 if not is_peak_hour else 0.7
            traffic_density = min(1.0, traffic_base + np.random.uniform(-0.2, 0.3))
            
            # Number of stops
            num_stops = int(distance_km * 1.5) + np.random.randint(-2, 3)
            num_stops = max(2, num_stops)
            
            # Average stop time (seconds)
            avg_stop_time = 30 + np.random.randint(-10, 20)
            
            # Calculate realistic ETA (in seconds)
            base_speed = 25  # km/h base speed
            
            # Adjust speed based on conditions
            speed_factor = 1.0
            if is_peak_hour:
                speed_factor *= 0.6
            if weather_code == 1:
                speed_factor *= 0.85
            elif weather_code == 2:
                speed_factor *= 0.7
            
            speed_factor *= (1 - traffic_density * 0.5)
            
            effective_speed = base_speed * speed_factor
            travel_time = (distance_km / effective_speed) * 3600  # Convert to seconds
            stop_time = num_stops * avg_stop_time
            
            # Add some random variation
            eta_seconds = travel_time + stop_time + np.random.uniform(-120, 180)
            
            data.append({
                'distance_km': distance_km,
                'time_of_day': time_of_day,
                'day_of_week': day_of_week,
                'is_peak_hour': is_peak_hour,
                'weather_code': weather_code,
                'traffic_density': traffic_density,
                'num_stops': num_stops,
                'avg_stop_time': avg_stop_time,
                'eta_seconds': eta_seconds
            })
        
        return pd.DataFrame(data)
    
    def prepare_sequences(self, data: pd.DataFrame, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Input DataFrame
            seq_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        # For prototype, we'll simulate sequence data
        X_list = []
        y_list = []
        
        for i in range(len(data) - seq_length):
            X_seq = data[self.feature_names].iloc[i:i+seq_length].values
            y_val = data['eta_seconds'].iloc[i+seq_length]
            
            X_list.append(X_seq)
            y_list.append(y_val)
        
        return np.array(X_list), np.array(y_list)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture
        
        Args:
            input_shape: Shape of input data (seq_length, n_features)
        """
        # For production, uncomment this code:
        """
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        """
        
        # For prototype, we'll use a mock model
        self.model = "MockLSTMModel"
        print(f"Model built with input shape: {input_shape}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # For production, uncomment this code:
        """
        # Initialize scalers
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Reshape and scale data
        n_samples, seq_len, n_features = X_train.shape
        X_reshaped = X_train.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        
        # Train model
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        self.is_trained = True
        return history.history
        """
        
        # Mock training for prototype
        print(f"Training model with {len(X_train)} samples...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Simulate training metrics
        history = {
            'loss': [0.5 - i*0.008 for i in range(epochs)],
            'mae': [0.3 - i*0.004 for i in range(epochs)]
        }
        
        self.is_trained = True
        return history
    
    def save_model(self, save_dir: str = "models") -> None:
        """
        Save the trained model and scalers
        
        Args:
            save_dir: Directory to save model files
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # For production:
        # self.model.save(os.path.join(save_dir, 'eta_model.h5'))
        # pickle.dump(self.scaler_X, open(os.path.join(save_dir, 'scaler_X.pkl'), 'wb'))
        # pickle.dump(self.scaler_y, open(os.path.join(save_dir, 'scaler_y.pkl'), 'wb'))
        
        # Mock save for prototype
        model_info = {
            'model_type': 'LSTM',
            'features': self.feature_names,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model saved to {save_dir}/")
    
    def load_model(self, load_dir: str = "models") -> None:
        """
        Load a pre-trained model and scalers
        
        Args:
            load_dir: Directory containing model files
        """
        # For production:
        # self.model = load_model(os.path.join(load_dir, 'eta_model.h5'))
        # self.scaler_X = pickle.load(open(os.path.join(load_dir, 'scaler_X.pkl'), 'rb'))
        # self.scaler_y = pickle.load(open(os.path.join(load_dir, 'scaler_y.pkl'), 'rb'))
        
        # Mock load for prototype
        import os
        model_info_path = os.path.join(load_dir, 'model_info.json')
        
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            self.is_trained = model_info.get('is_trained', False)
            print(f"Model loaded from {load_dir}/")
        else:
            print(f"No model found in {load_dir}/")
    
    def predict_eta(self, 
                   origin: str,
                   destination: str,
                   current_conditions: Optional[Dict] = None) -> Dict:
        """
        Main prediction function - returns ETA in seconds
        
        Args:
            origin: Starting point
            destination: End point
            current_conditions: Optional dict with current conditions
            
        Returns:
            Dictionary with prediction results
        """
        # Extract or generate features
        if current_conditions:
            distance_km = current_conditions.get('distance_km', 10)
            weather_code = current_conditions.get('weather_code', 0)
            traffic_density = current_conditions.get('traffic_density', 0.5)
            num_stops = current_conditions.get('num_stops', 8)
        else:
            # Generate mock values for prototype
            distance_km = random.uniform(5, 20)
            weather_code = random.choice([0, 0, 1])
            traffic_density = random.uniform(0.3, 0.8)
            num_stops = int(distance_km * 1.5)
        
        # Get current time features
        now = datetime.now()
        time_of_day = now.hour
        day_of_week = now.weekday()
        is_peak_hour = 1 if time_of_day in [8, 9, 17, 18, 19] else 0
        avg_stop_time = 35
        
        # For production with real model:
        """
        # Create feature sequence (using last 10 time steps)
        features = np.array([[
            distance_km, time_of_day, day_of_week, is_peak_hour,
            weather_code, traffic_density, num_stops, avg_stop_time
        ]])
        
        # Create sequence (repeat for simplicity in prototype)
        X = np.repeat(features, 10, axis=0).reshape(1, 10, 8)
        
        # Scale and predict
        X_scaled = self.scaler_X.transform(X.reshape(-1, 8)).reshape(1, 10, 8)
        y_pred_scaled = self.model.predict(X_scaled)
        eta_seconds = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
        """
        
        # Mock prediction for prototype
        base_time = (distance_km / 20) * 3600  # 20 km/h average
        traffic_factor = 1 + traffic_density * 0.5
        weather_factor = 1 + weather_code * 0.15
        peak_factor = 1.3 if is_peak_hour else 1.0
        stop_time = num_stops * avg_stop_time
        
        eta_seconds = int(base_time * traffic_factor * weather_factor * peak_factor + stop_time)
        
        # Add some randomness for realism
        eta_seconds += random.randint(-60, 120)
        
        # Calculate confidence based on conditions
        confidence = 0.95
        if weather_code > 0:
            confidence -= 0.05
        if traffic_density > 0.7:
            confidence -= 0.1
        if is_peak_hour:
            confidence -= 0.05
        
        confidence = max(0.6, confidence)
        
        return {
            'eta_seconds': eta_seconds,
            'eta_minutes': round(eta_seconds / 60, 1),
            'confidence': confidence,
            'origin': origin,
            'destination': destination,
            'distance_km': round(distance_km, 2),
            'predicted_at': datetime.now().isoformat(),
            'factors': {
                'is_peak_hour': bool(is_peak_hour),
                'weather_condition': ['Clear', 'Light Rain', 'Heavy Rain'][weather_code],
                'traffic_level': 'High' if traffic_density > 0.7 else 'Medium' if traffic_density > 0.4 else 'Low',
                'number_of_stops': num_stops
            }
        }

# Wrapper function for easy integration with FastAPI
def predict_eta(origin: str, destination: str, conditions: Optional[Dict] = None) -> Dict:
    """
    Wrapper function for API integration
    
    Args:
        origin: Starting point
        destination: End point
        conditions: Optional conditions dictionary
        
    Returns:
        ETA prediction results
    """
    predictor = BusETAPredictor()
    return predictor.predict_eta(origin, destination, conditions)

from apis import get_complete_route_weather_data
import random

def predict_eta_from_api_data(origin: str, destination: str) -> dict:
    """
    Given origin and destination, get API data and return ETA prediction.
    """
    # Step 1: Get API data
    api_data = get_complete_route_weather_data(origin, destination)

    # Step 2: Initialize predictor
    predictor = BusETAPredictor()

    # Step 3: Train mock model with synthetic data
    data = predictor.generate_synthetic_data(n_samples=1000)
    X, y = predictor.prepare_sequences(data, seq_length=10)
    predictor.build_model(input_shape=(10, 8))
    predictor.train(X, y, epochs=5, batch_size=32)
    predictor.save_model()

    # Step 4: Prepare conditions from API data
    distance_km = api_data["route"]["distance_meters"] / 1000
    precip_mm = api_data["weather"]["precip_mm"]
    weather_code = 0 if precip_mm < 0.1 else 1
    traffic_density = random.uniform(0.3, 0.8)
    num_stops = max(2, int(distance_km * 1.5))

    # Step 5: Predict ETA
    result = predictor.predict_eta(
        origin=origin,
        destination=destination,
        current_conditions={
            "distance_km": distance_km,
            "weather_code": weather_code,
            "traffic_density": traffic_density,
            "num_stops": num_stops
        }
    )

    return {
        "api_data": api_data,
        "eta_prediction": result
    }

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = BusETAPredictor()
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    data = predictor.generate_synthetic_data(n_samples=1000)
    print(f"Generated {len(data)} samples")
    
    # Prepare sequences
    X, y = predictor.prepare_sequences(data, seq_length=10)
    print(f"Prepared sequences: X shape={X.shape}, y shape={y.shape}")
    
    # Build and train model
    predictor.build_model(input_shape=(10, 8))
    history = predictor.train(X, y, epochs=5, batch_size=32)
    
    # Save model
    predictor.save_model()
    
    # Test prediction
    print("\n=== Testing Predictions ===")
    
    # Test case 1: Short distance, good conditions
    result1 = predictor.predict_eta(
        origin="BITS Pilani Main Gate",
        destination="Pilani Bus Stand",
        current_conditions={
            'distance_km': 3.5,
            'weather_code': 0,
            'traffic_density': 0.3,
            'num_stops': 5
        }
    )
    print(f"\nTest 1 - Short Route:")
    print(f"  Route: {result1['origin']} -> {result1['destination']}")
    print(f"  ETA: {result1['eta_minutes']} minutes")
    print(f"  Confidence: {result1['confidence']*100:.1f}%")
    
    # Test case 2: Long distance, peak hour
    result2 = predictor.predict_eta(
        origin="Pilani Railway Station",
        destination="BITS Pilani",
        current_conditions={
            'distance_km': 15,
            'weather_code': 1,
            'traffic_density': 0.75,
            'num_stops': 12
        }
    )
    print(f"\nTest 2 - Long Route (Peak Hour):")
    print(f"  Route: {result2['origin']} -> {result2['destination']}")
    print(f"  ETA: {result2['eta_minutes']} minutes")
    print(f"  Confidence: {result2['confidence']*100:.1f}%")
    
    # Test case 3: Random conditions
    result3 = predictor.predict_eta(
        origin="City Center",
        destination="Airport"
    )
    print(f"\nTest 3 - Random Conditions:")
    print(f"  Route: {result3['origin']} -> {result3['destination']}")
    print(f"  ETA: {result3['eta_minutes']} minutes")
    print(f"  Distance: {result3['distance_km']} km")
    print(f"  Factors: {result3['factors']}")