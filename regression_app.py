"""
Simple Linear Regression App
Predicts housing prices based on features using scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt


def load_data():
    """Load the California housing dataset."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    return df


def train_model(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return {
        'predictions': predictions,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def plot_results(y_test, predictions):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Housing Prices')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()


def plot_location_vs_price(df):
    """Plot a map showing how location (lat/long) relates to housing price."""
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df['Longitude'],
        df['Latitude'],
        c=df['Price'],
        cmap='viridis',
        alpha=0.5,
        s=10
    )
    plt.colorbar(scatter, label='Price ($100k)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('California Housing Prices by Location')
    plt.tight_layout()
    plt.savefig('location_vs_price.png')
    plt.show()


def main():
    # Load data
    print("Loading California Housing dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"\nFeatures: {df.columns.tolist()[:-1]}")
    print(f"\nDataset preview:\n{df.head()}")

    # Prepare features and target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train model
    print("\nTraining Linear Regression model...")
    model = train_model(X_train, y_train)

    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test)

    print(f"\nModel Performance:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  R² Score: {results['r2']:.4f}")

    # Show feature importance
    print("\nFeature Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")

    # Plot results
    plot_results(y_test, results['predictions'])

    # Example prediction
    print("\n--- Example Prediction ---")
    sample = X_test.iloc[0:1]
    prediction = model.predict(sample)[0]
    actual = y_test.iloc[0]
    print(f"Sample features:\n{sample.to_string()}")
    print(f"\nPredicted price: ${prediction * 100000:.2f}")
    print(f"Actual price: ${actual * 100000:.2f}")


if __name__ == "__main__":
    main()
