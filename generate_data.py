import pandas as pd
import numpy as np

def generate_water_quality_data(n_samples=1000):
    """Generates a synthetic water quality dataset."""
    np.random.seed(42)
    
    data = {
        'ph': np.random.uniform(0, 14, n_samples),
        'Hardness': np.random.normal(196, 33, n_samples),
        'Solids': np.random.normal(22014, 8768, n_samples),
        'Chloramines': np.random.normal(7.1, 1.5, n_samples),
        'Sulfate': np.random.normal(333, 41, n_samples),
        'Conductivity': np.random.normal(426, 80, n_samples),
        'Organic_carbon': np.random.normal(14, 3, n_samples),
        'Trihalomethanes': np.random.normal(66, 16, n_samples),
        'Turbidity': np.random.normal(3.9, 0.7, n_samples),
        'Potability': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]) # 0: Unsafe, 1: Safe
    }
    
    # Introduce some correlations/logic for Potability (simplified)
    # e.g., extreme pH or high turbidity makes it less likely to be potable
    df = pd.DataFrame(data)
    
    # Adjust Potability based on rules to make it learnable
    mask_unsafe = (df['ph'] < 6.5) | (df['ph'] > 8.5) | (df['Turbidity'] > 5) | (df['Sulfate'] > 400)
    
    # Randomly flip some to unsafe based on mask, but keep some noise
    df.loc[mask_unsafe & (np.random.random(n_samples) > 0.3), 'Potability'] = 0
    
    return df

if __name__ == "__main__":
    print("Generating synthetic water quality data...")
    df = generate_water_quality_data(2000)
    
    # Introduce some missing values to simulate real-world data
    for col in df.columns[:-1]: # Skip target
        df.loc[df.sample(frac=0.05).index, col] = np.nan
        
    output_file = 'water_quality.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(df.head())
    print(df['Potability'].value_counts())
