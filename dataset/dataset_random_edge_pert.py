# File: New folder/dataset/dataset_laplacian5.py
# Implements Random Edge Feature Perturbation (removes Laplacian/Eigenvalue calc)

import os
import csv
import math
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
# Removed Laplacian/linalg imports as they are no longer used
# from torch_geometric.utils import get_laplacian, to_dense_adj
# import torch.linalg

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

# Define constants for atom and bond types
ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
# Still using 3 features: Bond type, Bond direction, Perturbed Weight
NUM_EDGE_FEATURES = 3


def read_smiles(data_path, remove_header=False):
    """Reads SMILES strings from a CSV file."""
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        if remove_header:
            _ = next(csv_reader) # Skip header
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            if Chem.MolFromSmiles(smiles) is not None:
                smiles_data.append(smiles)
    return smiles_data


# Renamed function slightly to reflect the change
def random_edge_feature_perturbation(data, perturbation_scale=0.25):
    """
    Applies Random Perturbation to an edge feature channel.
    Adds random noise as a new feature to edge attributes.
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    # Use data.edge_attr.device if available, otherwise default to CPU or data.x.device
    device = data.edge_attr.device if hasattr(data.edge_attr, 'device') else (data.x.device if hasattr(data, 'x') else 'cpu')


    if num_nodes == 0: # Handle empty graph nodes
         pert_edge_attr = torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float, device=device)
         return Data(x=data.x, edge_index=data.edge_index, edge_attr=pert_edge_attr)

    if num_edges == 0: # Handle graphs with nodes but no edges
         pert_edge_attr = torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float, device=device)
         return Data(x=data.x, edge_index=data.edge_index, edge_attr=pert_edge_attr)

    # --- Generate Perturbation (No Laplacian Calculation Needed) ---
    perturbation = torch.randn(num_edges, 1, device=device) * perturbation_scale

    # --- Combine Features ---
    # Ensure original attributes are float and on the correct device
    original_edge_attr_float = data.edge_attr.float().to(device)

    # Handle case where original edge_attr might have been empty tensor with wrong shape
    if original_edge_attr_float.numel() == 0 and num_edges > 0:
         # This shouldn't happen if MoleculeDataset creates edge_attr correctly, but safety check
         original_edge_attr_float = torch.zeros((num_edges, NUM_EDGE_FEATURES - 1), device=device)
         print(f"Warning: Had to create placeholder original edge attributes.")


    # Check dimensions before concatenation
    if original_edge_attr_float.size(0) != perturbation.size(0):
        raise RuntimeError(f"Dimension mismatch before concat: edge_attr ({original_edge_attr_float.shape}) vs perturbation ({perturbation.shape})")

    pert_edge_attr = torch.cat([original_edge_attr_float, perturbation], dim=1)

    # --- Final Dimension Check ---
    if pert_edge_attr.size(1) != NUM_EDGE_FEATURES:
        # If original only had 2 dims, this should now be 3. If it's still not 3, something's wrong.
        print(f"Warning: Final edge attribute dimension is {pert_edge_attr.size(1)}, expected {NUM_EDGE_FEATURES}. Adjusting.")
        missing_dims = NUM_EDGE_FEATURES - pert_edge_attr.size(1)
        if missing_dims > 0:
             padding = torch.zeros((pert_edge_attr.size(0), missing_dims), dtype=torch.float, device=device)
             pert_edge_attr = torch.cat([pert_edge_attr, padding], dim=1)
        elif missing_dims < 0: # Too many dimensions? Trim.
             pert_edge_attr = pert_edge_attr[:, :NUM_EDGE_FEATURES]


    data_perturbed = Data(x=data.x.to(device), edge_index=data.edge_index.to(device), edge_attr=pert_edge_attr)

    return data_perturbed


class MoleculeDataset(Dataset):
    # Pass perturbation_scale, remove eigenvalue_weighting
    def __init__(self, data_path, remove_header=False, perturbation_scale=0.1):
        super(Dataset, self).__init__()
        print(f"Reading SMILES from: {data_path}")
        self.smiles_data = read_smiles(data_path, remove_header)
        print(f"Read {len(self.smiles_data)} valid SMILES.")
        self.perturbation_scale = perturbation_scale


    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        N = mol.GetNumAtoms()

        if N == 0:
             # print(f"Warning: Empty molecule generated for SMILES: {self.smiles_data[index]}")
             empty_data = Data(x=torch.empty((0, 2), dtype=torch.long),
                               edge_index=torch.empty((2, 0), dtype=torch.long),
                               edge_attr=torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float))
             return empty_data, empty_data

        M = mol.GetNumBonds()

        # --- Node Features ---
        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num not in ATOM_LIST: type_idx.append(0)
            else: type_idx.append(ATOM_LIST.index(atomic_num))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        # --- Edge Features (Initial - Type and Direction) ---
        row, col, edge_feat_base = [], [], []
        if M > 0:
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]; col += [end, start]
                bond_type_rdkit = bond.GetBondType()
                bond_type = BOND_LIST.index(bond_type_rdkit) if bond_type_rdkit in BOND_LIST else 0
                bond_dir = BONDDIR_LIST.index(bond.GetBondDir())
                edge_feat_base.append([bond_type, bond_dir])
                edge_feat_base.append([bond_type, bond_dir])
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr_base = torch.tensor(np.array(edge_feat_base), dtype=torch.long)
            # Handle single bond case which might result in 1D tensor
            if edge_attr_base.dim() == 1 and edge_attr_base.numel() == 2:
                 edge_attr_base = edge_attr_base.unsqueeze(0) # Shape [1, 2]
            # Handle case where edge_feat_base was empty list -> results in size [0] tensor
            elif M > 0 and edge_attr_base.numel() == 0 :
                 edge_attr_base = torch.empty((2*M, 2), dtype=torch.long) # Recreate with correct shape if empty
                 print(f"Warning: Corrected empty edge_attr_base for M={M}")


        else: # M == 0
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr_base = torch.empty((0, 2), dtype=torch.long)


        # --- Create Base Data Object ---
        # Ensure edge_attr_base has 2 dimensions before creating Data object
        if edge_attr_base.dim() != 2 and edge_attr_base.numel() > 0 :
             # This case should be less likely now, but as a safeguard
             print(f"Warning: Reshaping edge_attr_base with dim {edge_attr_base.dim()} before Data creation.")
             try:
                  edge_attr_base = edge_attr_base.view(-1, 2)
             except RuntimeError: # If reshape fails (e.g., odd number of elements)
                  edge_attr_base = torch.empty((0, 2), dtype=torch.long) # Fallback to empty


        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_base)

        # --- Apply Random Edge Feature Perturbation for two views ---
        data_i = random_edge_feature_perturbation(data, perturbation_scale=self.perturbation_scale)
        data_j = random_edge_feature_perturbation(data, perturbation_scale=self.perturbation_scale)

        return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    # Remove eigenvalue_weighting parameter
    def __init__(self, batch_size, num_workers, valid_size, data_path, remove_header=False, perturbation_scale=0.1):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.remove_header = remove_header
        self.perturbation_scale = perturbation_scale # Keep scale

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(
            data_path=self.data_path,
            remove_header=self.remove_header,
            perturbation_scale=self.perturbation_scale # Pass scale
        )
        # (rest of the function is unchanged)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # --- Unchanged ---
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader( train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers, drop_last=True )
        valid_loader = DataLoader( train_dataset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=self.num_workers, drop_last=True )
        return train_loader, valid_loader

# Corrected if __name__ == "__main__": block for dataset_laplacian5.py (Random Perturbation version)

if __name__ == "__main__":
    dummy_file = 'dummy_smiles.csv'
    # (Create dummy file - unchanged)
    with open(dummy_file, 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['id', 'smiles'])
        writer.writerow([1, 'CCO']); writer.writerow([2, 'C1=CC=CC=C1']); writer.writerow([3, 'CC(=O)O'])
        writer.writerow([4, 'C']); writer.writerow([6, 'O=C=O']); writer.writerow([7, '[Na+].[Cl-]']) # Salt example
    print("Dummy CSV created.")

    # Config using only random perturbation
    config = {
        'batch_size': 2,
        'num_workers': 0,
        'valid_size': 0.5,
        'dataset': {
            'data_path': dummy_file,
            'remove_header': True,
            'perturbation_scale': 0.1
            }
        }

    # Wrapper call using only random perturbation scale
    dataset_wrapper = MoleculeDatasetWrapper(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        valid_size=config['valid_size'],
        data_path=config['dataset']['data_path'],
        remove_header=config['dataset']['remove_header'],
        perturbation_scale=config['dataset']['perturbation_scale']
    )

    train_loader, valid_loader = dataset_wrapper.get_data_loaders()

    print("\nTesting Train Loader...")
    batch_count = 0
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        data_i = None # Initialize to None
        data_j = None
        try:
            if batch is None or not batch:
                print("  Warning: Received empty or None batch.")
                continue
            if len(batch) != 2:
                print(f"  Warning: Batch does not contain two elements. Contains {len(batch)}.")
                continue

            # --- Assign only if checks pass ---
            data_i, data_j = batch

            # --- Checks moved inside try block after assignment ---
            if hasattr(data_i, 'edge_attr') and hasattr(data_j, 'edge_attr'):
                print(f"  Data_i batch graphs: {data_i.num_graphs}, edge_attr shape: {data_i.edge_attr.shape}")
                print(f"  Data_j batch graphs: {data_j.num_graphs}, edge_attr shape: {data_j.edge_attr.shape}")
                if data_i.edge_attr.numel() > 0: assert data_i.edge_attr.shape[1] == NUM_EDGE_FEATURES, f"Data_i edge_attr dim: {data_i.edge_attr.shape[1]}"
                if data_j.edge_attr.numel() > 0: assert data_j.edge_attr.shape[1] == NUM_EDGE_FEATURES, f"Data_j edge_attr dim: {data_j.edge_attr.shape[1]}"
            else:
                print(f"  Batch {i} structure invalid (missing edge_attr). data_i type: {type(data_i)}, data_j type: {type(data_j)}")

        except (TypeError, ValueError, AssertionError) as e:
             print(f"  Error processing batch {i}: {e}")
             continue # Skip to next batch if error occurs

        batch_count += 1;
        if batch_count >= 2: break # Limit printed batches

    print("\nTesting Validation Loader...")
    batch_count = 0
    for i, batch in enumerate(valid_loader):
        print(f"Validation Batch {i}:")
        data_i = None
        data_j = None
        try:
            if batch is None or not batch:
                print("  Warning: Received empty or None batch.")
                continue
            if len(batch) != 2:
                print(f"  Warning: Batch does not contain two elements. Contains {len(batch)}.")
                continue

            data_i, data_j = batch

            if hasattr(data_i, 'edge_attr') and hasattr(data_j, 'edge_attr'):
                print(f"  Data_i batch graphs: {data_i.num_graphs}, edge_attr shape: {data_i.edge_attr.shape}")
                print(f"  Data_j batch graphs: {data_j.num_graphs}, edge_attr shape: {data_j.edge_attr.shape}")
                if data_i.edge_attr.numel() > 0: assert data_i.edge_attr.shape[1] == NUM_EDGE_FEATURES
                if data_j.edge_attr.numel() > 0: assert data_j.edge_attr.shape[1] == NUM_EDGE_FEATURES
            else:
                print(f"  Validation Batch {i} structure invalid.")

        except (TypeError, ValueError, AssertionError) as e:
             print(f"  Error processing validation batch {i}: {e}")
             continue

        batch_count += 1
        if batch_count >= 2: break # Limit printed batches

    # (Cleanup remains the same)
    try: os.remove(dummy_file); print(f"\nRemoved dummy file: {dummy_file}")
    except OSError as e: print(f"Error removing dummy file {dummy_file}: {e}")