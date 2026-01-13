"""
Comprehensive unit tests for regression_app.py

Tests cover:
- Data loading functionality
- Model training
- Model evaluation metrics
- Plotting functions (with mocked matplotlib)
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

import regression_app


class TestLoadData:
    """Test suite for the load_data function."""

    @patch('regression_app.fetch_california_housing')
    def test_load_data_returns_dataframe(self, mock_fetch):
        """Test that load_data returns a pandas DataFrame."""
        # Arrange: Create mock housing data
        mock_housing = MagicMock()
        mock_housing.data = np.array([[1, 2, 3], [4, 5, 6]])
        mock_housing.feature_names = ['Feature1', 'Feature2', 'Feature3']
        mock_housing.target = np.array([10, 20])
        mock_fetch.return_value = mock_housing

        # Act
        result = regression_app.load_data()

        # Assert
        assert isinstance(result, pd.DataFrame)
        mock_fetch.assert_called_once()

    @patch('regression_app.fetch_california_housing')
    def test_load_data_has_correct_columns(self, mock_fetch):
        """Test that the DataFrame has all feature columns plus Price."""
        # Arrange
        mock_housing = MagicMock()
        mock_housing.data = np.array([[1, 2], [3, 4]])
        mock_housing.feature_names = ['MedInc', 'HouseAge']
        mock_housing.target = np.array([5, 6])
        mock_fetch.return_value = mock_housing

        # Act
        result = regression_app.load_data()

        # Assert
        expected_columns = ['MedInc', 'HouseAge', 'Price']
        assert list(result.columns) == expected_columns

    @patch('regression_app.fetch_california_housing')
    def test_load_data_price_column_matches_target(self, mock_fetch):
        """Test that the Price column contains the target values."""
        # Arrange
        mock_housing = MagicMock()
        mock_housing.data = np.array([[1, 2], [3, 4]])
        mock_housing.feature_names = ['Feature1', 'Feature2']
        target_values = np.array([100, 200])
        mock_housing.target = target_values
        mock_fetch.return_value = mock_housing

        # Act
        result = regression_app.load_data()

        # Assert
        np.testing.assert_array_equal(result['Price'].values, target_values)

    @patch('regression_app.fetch_california_housing')
    def test_load_data_correct_shape(self, mock_fetch):
        """Test that the DataFrame has the correct shape."""
        # Arrange
        mock_housing = MagicMock()
        mock_housing.data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mock_housing.feature_names = ['A', 'B', 'C']
        mock_housing.target = np.array([10, 20, 30])
        mock_fetch.return_value = mock_housing

        # Act
        result = regression_app.load_data()

        # Assert: 3 rows, 4 columns (3 features + 1 Price)
        assert result.shape == (3, 4)

    @patch('regression_app.fetch_california_housing')
    def test_load_data_with_actual_california_housing_structure(self, mock_fetch):
        """Test with realistic California housing dataset structure."""
        # Arrange: Use actual dataset structure
        actual_housing = fetch_california_housing()
        mock_fetch.return_value = actual_housing

        # Act
        result = regression_app.load_data()

        # Assert
        assert 'Price' in result.columns
        assert len(result.columns) == len(actual_housing.feature_names) + 1
        assert result.shape[0] == actual_housing.data.shape[0]


class TestTrainModel:
    """Test suite for the train_model function."""

    def test_train_model_returns_linear_regression(self):
        """Test that train_model returns a LinearRegression instance."""
        # Arrange
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([10, 20, 30])

        # Act
        model = regression_app.train_model(X_train, y_train)

        # Assert
        assert isinstance(model, LinearRegression)

    def test_train_model_fits_data(self):
        """Test that the model is properly fitted with training data."""
        # Arrange
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([2, 4, 6, 8])

        # Act
        model = regression_app.train_model(X_train, y_train)

        # Assert: For perfect linear relationship y = 2x, coefficient should be ~2
        assert model.coef_ is not None
        assert model.intercept_ is not None
        np.testing.assert_almost_equal(model.coef_[0], 2.0, decimal=10)
        np.testing.assert_almost_equal(model.intercept_, 0.0, decimal=10)

    def test_train_model_with_multiple_features(self):
        """Test training with multiple features."""
        # Arrange
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([5, 8, 11, 14])  # y = x1 + 2*x2 + 1

        # Act
        model = regression_app.train_model(X_train, y_train)

        # Assert
        assert len(model.coef_) == 2
        predictions = model.predict(X_train)
        np.testing.assert_allclose(predictions, y_train, rtol=1e-10)

    def test_train_model_with_pandas_input(self):
        """Test that train_model works with pandas DataFrames."""
        # Arrange
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_train = pd.Series([10, 20, 30])

        # Act
        model = regression_app.train_model(X_train, y_train)

        # Assert
        assert isinstance(model, LinearRegression)
        assert model.coef_ is not None

    def test_train_model_with_single_sample(self):
        """Test training with minimum number of samples."""
        # Arrange: Single sample with single feature
        X_train = np.array([[5]])
        y_train = np.array([10])

        # Act
        model = regression_app.train_model(X_train, y_train)

        # Assert: Model should still be created
        assert isinstance(model, LinearRegression)


class TestEvaluateModel:
    """Test suite for the evaluate_model function."""

    def test_evaluate_model_returns_dict(self):
        """Test that evaluate_model returns a dictionary."""
        # Arrange
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])
        X_test = np.array([[4], [5]])
        y_test = np.array([8, 10])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert
        assert isinstance(results, dict)

    def test_evaluate_model_has_required_keys(self):
        """Test that the results dictionary has all required keys."""
        # Arrange
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([10, 20, 30])
        X_test = np.array([[4]])
        y_test = np.array([40])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert
        expected_keys = {'predictions', 'mse', 'rmse', 'r2'}
        assert set(results.keys()) == expected_keys

    def test_evaluate_model_perfect_predictions(self):
        """Test evaluation metrics with perfect predictions."""
        # Arrange: Create a perfect linear relationship
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([2, 4, 6, 8])
        X_test = np.array([[5], [6]])
        y_test = np.array([10, 12])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert: Perfect predictions should have MSE=0, R²=1
        np.testing.assert_allclose(results['predictions'], y_test, rtol=1e-10)
        assert results['mse'] < 1e-20  # Essentially 0
        assert results['rmse'] < 1e-10  # Essentially 0
        np.testing.assert_almost_equal(results['r2'], 1.0, decimal=10)

    def test_evaluate_model_rmse_is_sqrt_of_mse(self):
        """Test that RMSE is correctly calculated as square root of MSE."""
        # Arrange
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([1.5, 3.5, 5.5])
        X_test = np.array([[4], [5]])
        y_test = np.array([7, 9])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert
        expected_rmse = np.sqrt(results['mse'])
        np.testing.assert_almost_equal(results['rmse'], expected_rmse)

    def test_evaluate_model_predictions_shape(self):
        """Test that predictions have the correct shape."""
        # Arrange
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([5, 8, 11])
        X_test = np.array([[4, 5], [5, 6], [6, 7]])
        y_test = np.array([14, 17, 20])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert
        assert results['predictions'].shape == y_test.shape

    def test_evaluate_model_r2_range(self):
        """Test that R² score is in valid range (can be negative for poor models)."""
        # Arrange
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([1, 2, 3, 4])
        X_test = np.array([[5], [6]])
        y_test = np.array([5, 6])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert: R² should be close to 1 for this good fit
        assert results['r2'] <= 1.0
        assert results['r2'] > 0.9  # Should be very close to 1

    def test_evaluate_model_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        # Arrange
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])
        X_test = np.array([[4], [5]])
        y_test = np.array([9, 11])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert
        assert results['mse'] >= 0
        assert results['rmse'] >= 0

    def test_evaluate_model_with_pandas_input(self):
        """Test evaluation with pandas DataFrame and Series."""
        # Arrange
        X_train = pd.DataFrame({'feature': [1, 2, 3]})
        y_train = pd.Series([10, 20, 30])
        X_test = pd.DataFrame({'feature': [4, 5]})
        y_test = pd.Series([40, 50])
        model = LinearRegression().fit(X_train, y_train)

        # Act
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert
        assert isinstance(results, dict)
        assert 'predictions' in results
        assert len(results['predictions']) == len(y_test)


class TestPlotResults:
    """Test suite for the plot_results function."""

    @patch('regression_app.plt')
    def test_plot_results_creates_figure(self, mock_plt):
        """Test that plot_results creates a matplotlib figure."""
        # Arrange
        y_test = np.array([1, 2, 3, 4, 5])
        predictions = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        # Act
        regression_app.plot_results(y_test, predictions)

        # Assert
        mock_plt.figure.assert_called_once_with(figsize=(10, 6))

    @patch('regression_app.plt')
    def test_plot_results_creates_scatter_plot(self, mock_plt):
        """Test that a scatter plot is created."""
        # Arrange
        y_test = np.array([1, 2, 3])
        predictions = np.array([1.1, 2.1, 3.1])

        # Act
        regression_app.plot_results(y_test, predictions)

        # Assert
        mock_plt.scatter.assert_called_once()
        call_args = mock_plt.scatter.call_args
        np.testing.assert_array_equal(call_args[0][0], y_test)
        np.testing.assert_array_equal(call_args[0][1], predictions)

    @patch('regression_app.plt')
    def test_plot_results_adds_diagonal_line(self, mock_plt):
        """Test that the ideal prediction line is plotted."""
        # Arrange
        y_test = np.array([1, 2, 3, 4, 5])
        predictions = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        # Act
        regression_app.plot_results(y_test, predictions)

        # Assert
        mock_plt.plot.assert_called_once()
        call_args = mock_plt.plot.call_args[0]
        # The line should go from min to max of y_test
        assert call_args[0] == [y_test.min(), y_test.max()]
        assert call_args[1] == [y_test.min(), y_test.max()]

    @patch('regression_app.plt')
    def test_plot_results_adds_labels(self, mock_plt):
        """Test that axis labels and title are added."""
        # Arrange
        y_test = np.array([1, 2, 3])
        predictions = np.array([1.1, 2.1, 3.1])

        # Act
        regression_app.plot_results(y_test, predictions)

        # Assert
        mock_plt.xlabel.assert_called_once_with('Actual Price')
        mock_plt.ylabel.assert_called_once_with('Predicted Price')
        mock_plt.title.assert_called_once_with('Actual vs Predicted Housing Prices')

    @patch('regression_app.plt')
    def test_plot_results_saves_figure(self, mock_plt):
        """Test that the plot is saved to a file."""
        # Arrange
        y_test = np.array([1, 2, 3])
        predictions = np.array([1.1, 2.1, 3.1])

        # Act
        regression_app.plot_results(y_test, predictions)

        # Assert
        mock_plt.savefig.assert_called_once_with('prediction_results.png')

    @patch('regression_app.plt')
    def test_plot_results_calls_show(self, mock_plt):
        """Test that plt.show() is called to display the plot."""
        # Arrange
        y_test = np.array([1, 2, 3])
        predictions = np.array([1.1, 2.1, 3.1])

        # Act
        regression_app.plot_results(y_test, predictions)

        # Assert
        mock_plt.show.assert_called_once()

    @patch('regression_app.plt')
    def test_plot_results_with_pandas_series(self, mock_plt):
        """Test plotting with pandas Series input."""
        # Arrange
        y_test = pd.Series([1.0, 2.0, 3.0])
        predictions = pd.Series([1.1, 2.1, 3.1])

        # Act
        regression_app.plot_results(y_test, predictions)

        # Assert
        mock_plt.scatter.assert_called_once()
        mock_plt.plot.assert_called_once()


class TestPlotLocationVsPrice:
    """Test suite for the plot_location_vs_price function."""

    @patch('regression_app.plt')
    def test_plot_location_vs_price_creates_figure(self, mock_plt):
        """Test that a figure is created with correct size."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23, -122.22, -122.24],
            'Latitude': [37.88, 37.86, 37.85],
            'Price': [4.5, 3.8, 5.0]
        })

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        mock_plt.figure.assert_called_once_with(figsize=(12, 8))

    @patch('regression_app.plt')
    def test_plot_location_vs_price_creates_scatter(self, mock_plt):
        """Test that a scatter plot is created with correct data."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23, -122.22],
            'Latitude': [37.88, 37.86],
            'Price': [4.5, 3.8]
        })
        mock_scatter_obj = MagicMock()
        mock_plt.scatter.return_value = mock_scatter_obj

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        mock_plt.scatter.assert_called_once()
        call_args = mock_plt.scatter.call_args

        # Check that correct columns are used
        pd.testing.assert_series_equal(
            pd.Series(call_args[0][0]),
            df['Longitude'],
            check_names=False
        )
        pd.testing.assert_series_equal(
            pd.Series(call_args[0][1]),
            df['Latitude'],
            check_names=False
        )

    @patch('regression_app.plt')
    def test_plot_location_vs_price_uses_price_for_color(self, mock_plt):
        """Test that Price column is used for color mapping."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23, -122.22, -122.24],
            'Latitude': [37.88, 37.86, 37.85],
            'Price': [4.5, 3.8, 5.0]
        })
        mock_scatter_obj = MagicMock()
        mock_plt.scatter.return_value = mock_scatter_obj

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        call_kwargs = mock_plt.scatter.call_args[1]
        pd.testing.assert_series_equal(
            pd.Series(call_kwargs['c']),
            df['Price'],
            check_names=False
        )
        assert call_kwargs['cmap'] == 'viridis'

    @patch('regression_app.plt')
    def test_plot_location_vs_price_adds_colorbar(self, mock_plt):
        """Test that a colorbar is added."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23, -122.22],
            'Latitude': [37.88, 37.86],
            'Price': [4.5, 3.8]
        })
        mock_scatter_obj = MagicMock()
        mock_plt.scatter.return_value = mock_scatter_obj

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        mock_plt.colorbar.assert_called_once_with(mock_scatter_obj, label='Price ($100k)')

    @patch('regression_app.plt')
    def test_plot_location_vs_price_adds_labels(self, mock_plt):
        """Test that axis labels and title are added."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23],
            'Latitude': [37.88],
            'Price': [4.5]
        })

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        mock_plt.xlabel.assert_called_once_with('Longitude')
        mock_plt.ylabel.assert_called_once_with('Latitude')
        mock_plt.title.assert_called_once_with('California Housing Prices by Location')

    @patch('regression_app.plt')
    def test_plot_location_vs_price_saves_figure(self, mock_plt):
        """Test that the plot is saved to a file."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23],
            'Latitude': [37.88],
            'Price': [4.5]
        })

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        mock_plt.savefig.assert_called_once_with('location_vs_price.png')

    @patch('regression_app.plt')
    def test_plot_location_vs_price_calls_show(self, mock_plt):
        """Test that plt.show() is called."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23],
            'Latitude': [37.88],
            'Price': [4.5]
        })

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        mock_plt.show.assert_called_once()

    @patch('regression_app.plt')
    def test_plot_location_vs_price_with_multiple_rows(self, mock_plt):
        """Test with a larger dataset."""
        # Arrange
        df = pd.DataFrame({
            'Longitude': [-122.23, -122.22, -122.24, -122.21, -122.25],
            'Latitude': [37.88, 37.86, 37.85, 37.87, 37.84],
            'Price': [4.5, 3.8, 5.0, 4.2, 3.5]
        })
        mock_scatter_obj = MagicMock()
        mock_plt.scatter.return_value = mock_scatter_obj

        # Act
        regression_app.plot_location_vs_price(df)

        # Assert
        mock_plt.scatter.assert_called_once()
        assert mock_plt.colorbar.called


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch('regression_app.fetch_california_housing')
    def test_complete_workflow(self, mock_fetch):
        """Test the complete machine learning workflow."""
        # Arrange: Create realistic mock data
        np.random.seed(42)
        n_samples = 100
        n_features = 3

        mock_housing = MagicMock()
        mock_housing.data = np.random.rand(n_samples, n_features)
        mock_housing.feature_names = ['Feature1', 'Feature2', 'Feature3']
        mock_housing.target = np.random.rand(n_samples) * 5
        mock_fetch.return_value = mock_housing

        # Act: Run through the workflow
        # 1. Load data
        df = regression_app.load_data()
        X = df.drop('Price', axis=1)
        y = df['Price']

        # 2. Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 3. Train model
        model = regression_app.train_model(X_train, y_train)

        # 4. Evaluate model
        results = regression_app.evaluate_model(model, X_test, y_test)

        # Assert: Verify all components work together
        assert isinstance(df, pd.DataFrame)
        assert isinstance(model, LinearRegression)
        assert isinstance(results, dict)
        assert 'predictions' in results
        assert 'mse' in results
        assert 'rmse' in results
        assert 'r2' in results
        assert len(results['predictions']) == len(y_test)

    @patch('regression_app.plt')
    @patch('regression_app.fetch_california_housing')
    def test_workflow_with_plotting(self, mock_fetch, mock_plt):
        """Test workflow including plotting functions."""
        # Arrange
        mock_housing = MagicMock()
        mock_housing.data = np.array([[1, 2, -122.23, 37.88],
                                      [3, 4, -122.22, 37.86]])
        mock_housing.feature_names = ['Feature1', 'Feature2', 'Longitude', 'Latitude']
        mock_housing.target = np.array([4.5, 3.8])
        mock_fetch.return_value = mock_housing

        # Act
        df = regression_app.load_data()
        X_train = df[['Feature1', 'Feature2']].values
        y_train = df['Price'].values
        model = regression_app.train_model(X_train, y_train)
        results = regression_app.evaluate_model(model, X_train, y_train)

        regression_app.plot_results(y_train, results['predictions'])
        regression_app.plot_location_vs_price(df)

        # Assert
        assert mock_plt.figure.call_count == 2
        assert mock_plt.savefig.call_count == 2
