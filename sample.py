import os
import random
import pandas as pd

# Function to sample 0.2% of the content
def sample_data(file_path, sample_size=20000):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    sampled_lines = random.sample(lines, sample_size)
    
    return sampled_lines


def sample_csv(file_path, sample_size=200000):
    df = pd.read_csv(file_path)  # Read CSV properly
    df = df.drop(columns=['inchi_key_1'])
    df.dropna(inplace=True)  # Drop rows with missing values
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)  # Sample rows safely

    return sampled_df


if __name__ == '__main__':
    # random seed
    random.seed(42)
    
    pubchem_file = 'biostructures.csv'
    data_folder = 'myopic-mces-data'

    pubchem_file = os.path.join(data_folder, pubchem_file)

    sampled_pubchem_data = sample_csv(pubchem_file, 200000)

    # Save the sampled data
    sampled_pubchem_file = 'biostructures_200k.csv'
    sampled_pubchem_file = os.path.join(data_folder, sampled_pubchem_file)
    # with open(sampled_pubchem_file, 'w') as file:
    #     file.writelines(sampled_pubchem_data)

    sampled_pubchem_data.to_csv(sampled_pubchem_file, index=False)

    

