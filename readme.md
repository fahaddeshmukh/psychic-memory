# Bayesian Linear Regression Project

This project implements Bayesian Linear Regression using the evidence approximation method. The code provides a framework for performing Bayesian inference in linear regression settings, optimizing the hyperparameters alpha (precision of the weight prior) and beta (precision of the noise) through evidence approximation.

## Project Structure

The project is divided into three main files:

1. `bayes_linear_regression.py`: Contains the implementation of the `BayesianLinearRegression` class.
2. `load_plot.py`: Contains functions for loading temperature data and plotting regression results.
3. `main.py`: The main script that loads data, fits the model, and plots the results.

## Dependencies

The project requires the following Python packages:

- numpy
- pandas
- matplotlib

These dependencies are listed in the `requirements.txt` file.

## Installation

1. **Clone the repository** (or download the project files):
    ```sh
    git clone <repository_url>
    cd your_project_directory
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv env
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        .\env\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source env/bin/activate
        ```

4. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Ensure the CSV file** `GM000003342.csv` is in the project directory. This file contains the temperature data.

2. **Run the main script**:
    ```sh
    python main.py
    ```

    This script will load the temperature data, fit the Bayesian Linear Regression model, and plot the regression results.

## Explanation of Files

- **`bayes_linear_regression.py`**: 
  - Implements the Bayesian Linear Regression model with methods for fitting the model (`fit`), computing the posterior distribution (`posterior`), and making predictions (`predict`).

- **`load_plot.py`**:
  - Contains functions to load temperature data from a CSV file (`load_temperature_data`) and to plot the regression results (`plot_regression`).

- **`main.py`**:
  - The main entry point for the project. It sets up the environment, loads the data, initializes the model, fits the model to the data, and plots the results.

## Example

Here's an example of how to run the project:

```sh
# Navigate to your project directory
cd your_project_directory

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate

# Install the required dependencies
pip install -r requirements.txt

# Run the main script
python main.py
