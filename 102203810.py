import pandas as pd
import numpy as np

# Function to normalize the matrix
def normalize_matrix(matrix, weights):
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    return weighted_matrix

# Function to calculate TOPSIS
def topsis(data_file, weights, impacts):
    try:
        # Read the CSV file
        data = pd.read_csv(data_file)

        # Extract fund names and criteria matrix
        fund_names = data.iloc[:, 0]
        matrix = data.iloc[:, 1:].values

        # Check if columns are numeric
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError("All criteria columns must contain numeric values.")

        # Parse weights and impacts
        weights = np.array([float(w) for w in weights.split(',')])
        impacts = impacts.split(',')

        if len(weights) != matrix.shape[1] or len(impacts) != matrix.shape[1]:
            raise ValueError("Number of weights and impacts must match the number of criteria.")

        if not all(impact in ['+', '-'] for impact in impacts):
            raise ValueError("Impacts must be either '+' or '-'.")

        # Normalize and weight the matrix
        weighted_matrix = normalize_matrix(matrix, weights)

        # Identify ideal best and worst
        ideal_best = np.max(weighted_matrix, axis=0) * (np.array(impacts) == '+') + \
                     np.min(weighted_matrix, axis=0) * (np.array(impacts) == '-')
        ideal_worst = np.min(weighted_matrix, axis=0) * (np.array(impacts) == '+') + \
                      np.max(weighted_matrix, axis=0) * (np.array(impacts) == '-')

        # Calculate distances to ideal best and worst
        dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

        # Calculate TOPSIS score
        topsis_score = dist_worst / (dist_best + dist_worst)

        # Rank based on TOPSIS score
        rank = topsis_score.argsort()[::-1] + 1

        # Create results DataFrame
        result = data.copy()
        result['Topsis Score'] = topsis_score
        result['Rank'] = rank

        return result

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")