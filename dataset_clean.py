import os
import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdBase
import sys

# Set RDKit error output to be more verbose for debugging
rdBase.LogToPythonStderr()

def clean_smiles_data_with_column(input_file, output_file, smiles_column='smiles', remove_header=True):
    """
    Clean a CSV file containing SMILES strings by removing rows with invalid SMILES.
    Uses a specific column name instead of assuming SMILES is in the last column.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to the output CSV file
    smiles_column : str
        Name of the column containing SMILES strings
    remove_header : bool
        Whether the CSV file has a header (should be True if using column names)
    
    Returns:
    --------
    tuple
        (total_count, valid_count, invalid_count, invalid_smiles_list)
    """
    # Try to read the CSV file as a pandas DataFrame
    try:
        df = pd.read_csv(input_file)
        # Check if the specified column exists
        if smiles_column not in df.columns:
            print(f"Error: Column '{smiles_column}' not found in the CSV file.")
            print(f"Available columns: {', '.join(df.columns)}")
            return 0, 0, 0, []
    except Exception as e:
        print(f"Error reading CSV file with pandas: {e}")
        print("Falling back to CSV reader...")
        # Fall back to regular CSV reading if pandas fails
        return clean_smiles_data(input_file, output_file, remove_header)
    
    # Track invalid SMILES for reporting
    invalid_smiles = []
    
    # Check which SMILES are valid
    valid_indices = []
    
    for i, smiles in enumerate(df[smiles_column]):
        # Skip NaN values
        if pd.isna(smiles):
            invalid_smiles.append((i+1, str(smiles), "NaN or empty value"))
            continue

        smiles_str = str(smiles)
        mol = Chem.MolFromSmiles(smiles_str)
        
        if mol is not None:
            # Additional check: Try to get atoms and bonds
            try:
                num_atoms = mol.GetNumAtoms()
                num_bonds = mol.GetNumBonds()
                # If we get here, the molecule is valid
                valid_indices.append(i)
            except Exception as e:
                invalid_smiles.append((i+1, smiles_str, f"Failed structure check: {str(e)}"))
        else:
            invalid_smiles.append((i+1, smiles_str, "Failed to parse"))
    
    # Create a new DataFrame with only valid rows
    df_cleaned = df.iloc[valid_indices].copy()
    
    # Write to the output file
    df_cleaned.to_csv(output_file, index=False)
    
    return len(df), len(valid_indices), len(invalid_smiles), invalid_smiles

def clean_smiles_data(input_file, output_file, remove_header=False):
    """
    Clean a CSV file containing SMILES strings by removing rows with invalid SMILES.
    Assumes SMILES is in the last column.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to the output CSV file
    remove_header : bool
        Whether to skip the first row of the input file
    
    Returns:
    --------
    tuple
        (total_count, valid_count, invalid_count, invalid_smiles_list)
    """
    # Read the raw data with all rows
    smiles_data = []
    all_rows = []
    
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        # Handle header if present
        header = None
        if remove_header:
            header = next(csv_reader)
        
        for i, row in enumerate(csv_reader):
            all_rows.append(row)
            smiles_data.append(row[-1])
    
    # Check which SMILES are valid
    valid_indices = []
    invalid_smiles = []
    
    for i, smiles in enumerate(smiles_data):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Additional check: Try to get atoms and bonds
            try:
                num_atoms = mol.GetNumAtoms()
                num_bonds = mol.GetNumBonds()
                # If we get here, the molecule is valid
                valid_indices.append(i)
            except Exception as e:
                invalid_smiles.append((i+1, smiles, f"Failed structure check: {str(e)}"))
        else:
            invalid_smiles.append((i+1, smiles, "Failed to parse"))
    
    # Write valid rows to the output file
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        
        # Write header if it exists
        if header:
            csv_writer.writerow(header)
        
        # Write valid rows
        for idx in valid_indices:
            csv_writer.writerow(all_rows[idx])
    
    return len(smiles_data), len(valid_indices), len(invalid_smiles), invalid_smiles

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.getcwd()
    
    # Prompt for input file
    input_file = "myopic-mces-data/biostructures.csv"
    
    # Default output file name
    output_file = os.path.splitext(input_file)[0] + "_cleaned.csv"
    

    column_input = "smiles"
    
    # Ask about header
    if not column_input:
        header_input = input("Does your CSV file have a header row to skip? (y/n): ").lower()
        remove_header = True if header_input == 'y' else False
        
        # Clean the data
        total, valid, invalid, invalid_list = clean_smiles_data(input_file, output_file, remove_header)
    else:
        # Use column name approach (assumes header exists)
        total, valid, invalid, invalid_list = clean_smiles_data_with_column(input_file, output_file, column_input)
    
    print(f"\nCleaning complete!")
    print(f"Total rows processed: {total}")
    print(f"Valid SMILES: {valid}")
    print(f"Invalid SMILES removed: {invalid}")
    
    # Display some examples of invalid SMILES
    if invalid > 0:
        print("\nSample of invalid SMILES entries:")
        for i, (row_num, smiles, reason) in enumerate(invalid_list[:10]):  # Show first 10 invalid entries
            print(f"  Row {row_num}: '{smiles}' - {reason}")
        
        if invalid > 10:
            print(f"  ... and {invalid - 10} more invalid entries")
            
        # Ask if user wants to save the list of invalid entries
        save_invalid = input("\nDo you want to save the full list of invalid SMILES to a file? (y/n): ").lower()
        if save_invalid == 'y':
            invalid_file = os.path.splitext(input_file)[0] + "_invalid_smiles.txt"
            with open(invalid_file, 'w') as f:
                f.write("Row,SMILES,Reason\n")
                for row_num, smiles, reason in invalid_list:
                    f.write(f"{row_num},\"{smiles}\",{reason}\n")
            print(f"Invalid SMILES list saved to {invalid_file}")
    
    print(f"\nCleaned data saved to {output_file}")