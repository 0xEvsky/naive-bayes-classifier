import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

def encode_features(data: DataFrame) -> DataFrame:
    encoded_data = data.drop('class', axis=1).copy()
    encoder = LabelEncoder()
    
    for col in encoded_data.columns:
        encoded_data[col] = encoder.fit_transform(encoded_data[col])
    
    return encoded_data

class PCA:
    def __init__(self, df: DataFrame):
        self.df: DataFrame = df
        self.data_values = self.df.values
    
    def means(self):
        means = np.mean(self.data_values, axis=0)
        return means

    def shift_values(self) -> np.ndarray:
        data_values = self.data_values
        means = self.means()
        
        shifted_values = data_values - means
        return shifted_values
    
    def compute_cov_matrix(self):
        shifted_values: np.ndarray = self.shift_values()
        n = self.data_values.shape[0]
        
        cov_matirx = (1 / n) * shifted_values.T @ shifted_values
        return cov_matirx
    
    
    def compute_fprime(self, k=3):
        cov_matrix = self.compute_cov_matrix()
        eigenvals, eigenvectors = np.linalg.eigh(cov_matrix)
        
        sorted_idx = np.argsort(eigenvals)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        shifted_values = self.shift_values()
        
        Q = eigenvectors[:, :k].T
        F = shifted_values.T
        
        return Q @ F


class FeatureSelection:
    pass