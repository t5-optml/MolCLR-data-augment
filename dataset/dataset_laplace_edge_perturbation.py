# File: New folder/dataset/dataset_laplacian5.py
# Implements Random Edge Feature Perturbation + Bond Direction Perturbation

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
NUM_BOND_DIR = len(BONDDIR_LIST) # Should be 3

# Still using 3 features: Bond type, Bond direction, Perturbed Weight
NUM_EDGE_FEATURES = 3


def read_smiles(data_path, remove_header=False):
    """Reads SMILES strings from a CSV file."""
    # (Implementation unchanged)
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        if remove_header: _ = next(csv_reader)
        for i, row in enumerate(csv_reader):
            smiles = row[-1];
            if Chem.MolFromSmiles(smiles) is not None: smiles_data.append(smiles)
    return smiles_data


# Renamed function to reflect combined augmentations
def combined_edge_perturbation(data, direction_perturb_ratio=0.15, noise_perturbation_scale=0.15):
    """
    Applies augmentations to edge features:
    1. Randomly perturbs the Bond Direction feature for a fraction of edges.
    2. Adds random Gaussian noise as a new (third) edge feature.
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1) # Number of directed edges (2*M)

    device = data.edge_attr.device if hasattr(data.edge_attr, 'device') else (data.x.device if hasattr(data, 'x') else 'cpu')

    # Handle empty/no-edge graphs
    if num_nodes == 0 or num_edges == 0:
         pert_edge_attr = torch.empty((num_edges, NUM_EDGE_FEATURES), dtype=torch.float, device=device)
         # (Fill with zeros or original data if applicable - simplified here)
         if data.edge_attr.numel() > 0: pert_edge_attr[:, :data.edge_attr.size(1)] = data.edge_attr.float()
         if pert_edge_attr.size(1) > data.edge_attr.size(1): pert_edge_attr[:, data.edge_attr.size(1):] = 0.0
         if pert_edge_attr.size(1) < NUM_EDGE_FEATURES:
             missing_dims = NUM_EDGE_FEATURES - pert_edge_attr.size(1); padding = torch.zeros((num_edges, missing_dims), dtype=torch.float, device=device)
             pert_edge_attr = torch.cat([pert_edge_attr, padding], dim=1)
         return Data(x=data.x.to(device) if hasattr(data, 'x') else None, edge_index=data.edge_index.to(device), edge_attr=pert_edge_attr)

    # --- 1. Perturb Bond Direction ---
    perturbed_edge_attr_base = data.edge_attr.clone().long() # Work with integer indices first
    num_undirected_edges = num_edges // 2
    num_edges_to_perturb = int(num_undirected_edges * direction_perturb_ratio)

    if num_edges_to_perturb > 0 and num_undirected_edges > 0:
        # Sample indices of undirected edges to perturb
        perturb_indices_undirected = random.sample(range(num_undirected_edges), num_edges_to_perturb)
        # Get indices for both directions in the edge_attr tensor
        perturb_indices_directed = []
        for idx in perturb_indices_undirected:
            perturb_indices_directed.extend([2 * idx, 2 * idx + 1])

        # Generate new random directions for the selected edges
        # Ensure new direction is chosen from available indices [0, 1, 2]
        new_directions = torch.randint(0, NUM_BOND_DIR, (num_edges_to_perturb,), device=device)

        # Apply the new direction to both corresponding directed edges
        for i, base_idx in enumerate(perturb_indices_undirected):
            dir_idx1 = 2 * base_idx
            dir_idx2 = 2 * base_idx + 1
            # Update BondDir (column 1)
            perturbed_edge_attr_base[dir_idx1, 1] = new_directions[i]
            perturbed_edge_attr_base[dir_idx2, 1] = new_directions[i] # Apply same change to reverse edge

    # --- 2. Generate Random Noise Perturbation ---
    noise_perturbation = torch.randn(num_edges, 1, device=device) * noise_perturbation_scale

    # --- 3. Combine Features ---
    # Concatenate the (potentially perturbed) base features with the noise
    final_edge_attr = torch.cat([perturbed_edge_attr_base.float(), noise_perturbation], dim=1)

    # --- Final Dimension Check ---
    if final_edge_attr.size(1) != NUM_EDGE_FEATURES:
        print(f"Warning: Final edge attribute dimension is {final_edge_attr.size(1)}, expected {NUM_EDGE_FEATURES}. Adjusting.")
        missing_dims = NUM_EDGE_FEATURES - final_edge_attr.size(1)
        if missing_dims > 0:
             padding = torch.zeros((num_edges, missing_dims), dtype=torch.float, device=device)
             final_edge_attr = torch.cat([final_edge_attr, padding], dim=1)
        elif missing_dims < 0:
             final_edge_attr = final_edge_attr[:, :NUM_EDGE_FEATURES]

    data_perturbed = Data(x=data.x.to(device), edge_index=data.edge_index.to(device), edge_attr=final_edge_attr)

    return data_perturbed


class MoleculeDataset(Dataset):
    # Add direction_perturb_ratio, rename noise scale param
    def __init__(self, data_path, remove_header=False, direction_perturb_ratio=0.1, noise_perturbation_scale=0.1):
        super(Dataset, self).__init__()
        print(f"Reading SMILES from: {data_path}")
        self.smiles_data = read_smiles(data_path, remove_header)
        print(f"Read {len(self.smiles_data)} valid SMILES.")
        self.direction_perturb_ratio = direction_perturb_ratio
        self.noise_perturbation_scale = noise_perturbation_scale


    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        N = mol.GetNumAtoms()
        if N == 0:
             empty_data = Data(x=torch.empty((0, 2), dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float))
             return empty_data, empty_data
        M = mol.GetNumBonds()

        # --- Node Features (unchanged) ---
        type_idx = []; chirality_idx = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum(); type_idx.append(ATOM_LIST.index(atomic_num) if atomic_num in ATOM_LIST else 0)
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        x = torch.cat([torch.tensor(type_idx, dtype=torch.long).view(-1, 1), torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)], dim=-1)

        # --- Edge Features (Initial - Type and Direction) (unchanged) ---
        row, col, edge_feat_base = [], [], []
        if M > 0:
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); row += [start, end]; col += [end, start]
                bond_type_rdkit = bond.GetBondType(); bond_type = BOND_LIST.index(bond_type_rdkit) if bond_type_rdkit in BOND_LIST else 0
                bond_dir = BONDDIR_LIST.index(bond.GetBondDir())
                edge_feat_base.append([bond_type, bond_dir]); edge_feat_base.append([bond_type, bond_dir])
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr_base = torch.tensor(np.array(edge_feat_base), dtype=torch.long)
            if edge_attr_base.dim() == 1 and edge_attr_base.numel() == 2: edge_attr_base = edge_attr_base.unsqueeze(0)
            elif M > 0 and edge_attr_base.numel() == 0 : edge_attr_base = torch.empty((2*M, 2), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long); edge_attr_base = torch.empty((0, 2), dtype=torch.long)

        if edge_attr_base.dim() != 2 and edge_attr_base.numel() > 0 :
             try: edge_attr_base = edge_attr_base.view(-1, 2)
             except RuntimeError: edge_attr_base = torch.empty((0, 2), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_base)

        # --- Apply Combined Edge Perturbation for two views ---
        # Apply potentially different random perturbations in each call
        data_i = combined_edge_perturbation(
            data,
            direction_perturb_ratio=self.direction_perturb_ratio,
            noise_perturbation_scale=self.noise_perturbation_scale
        )
        data_j = combined_edge_perturbation(
            data,
            direction_perturb_ratio=self.direction_perturb_ratio,
            noise_perturbation_scale=self.noise_perturbation_scale
        )

        return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    # Add direction_perturb_ratio parameter
    def __init__(self, batch_size, num_workers, valid_size, data_path, remove_header=False, direction_perturb_ratio=0.1, noise_perturbation_scale=0.1):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.remove_header = remove_header
        self.direction_perturb_ratio = direction_perturb_ratio # Store ratio
        self.noise_perturbation_scale = noise_perturbation_scale

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(
            data_path=self.data_path,
            remove_header=self.remove_header,
            direction_perturb_ratio=self.direction_perturb_ratio, # Pass params
            noise_perturbation_scale=self.noise_perturbation_scale
        )
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # --- Unchanged ---
        num_train = len(train_dataset); indices = list(range(num_train)); np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train)); train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx); valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader( train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers, drop_last=True )
        valid_loader = DataLoader( train_dataset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=self.num_workers, drop_last=True )
        return train_loader, valid_loader

# --- Example Usage (__main__) - update config/call ---
if __name__ == "__main__":
    dummy_file = 'dummy_smiles.csv'
    # (Create dummy file - unchanged)
    with open(dummy_file, 'w', newline='') as f: writer = csv.writer(f); writer.writerow(['id', 'smiles']); writer.writerow([1, 'CCO']); writer.writerow([2, 'C1=CC=CC=C1']); writer.writerow([3, 'CC(=O)O']); writer.writerow([4, 'C']); writer.writerow([6, 'O=C=O']); writer.writerow([7, '[Na+].[Cl-]'])
    print("Dummy CSV created.")

    # Add direction_perturb_ratio to config
    config = { 'batch_size': 2, 'num_workers': 0, 'valid_size': 0.5, 'dataset': { 'data_path': dummy_file, 'remove_header': True, 'direction_perturb_ratio': 0.15, 'noise_perturbation_scale': 0.1 } }

    # Add direction_perturb_ratio to wrapper call
    dataset_wrapper = MoleculeDatasetWrapper( batch_size=config['batch_size'], num_workers=config['num_workers'], valid_size=config['valid_size'], data_path=config['dataset']['data_path'], remove_header=config['dataset']['remove_header'], direction_perturb_ratio=config['dataset']['direction_perturb_ratio'], noise_perturbation_scale=config['dataset']['noise_perturbation_scale'] )

    train_loader, valid_loader = dataset_wrapper.get_data_loaders()
    print("\nTesting Train Loader...")
    # (Test loop is unchanged)
    batch_count = 0
    for i, batch in enumerate(train_loader):
        # ... (rest of test loop) ...
        if batch is None or not batch: continue; 
        try: data_i, data_j = batch; assert len(batch)==2
        
        except Exception as e: continue
        print(f"Batch {i}:");
        if hasattr(data_i, 'edge_attr') and hasattr(data_j, 'edge_attr'):
             print(f"  Data_i batch graphs: {data_i.num_graphs}, edge_attr shape: {data_i.edge_attr.shape}")
             print(f"  Data_j batch graphs: {data_j.num_graphs}, edge_attr shape: {data_j.edge_attr.shape}")
             if data_i.edge_attr.numel() > 0: assert data_i.edge_attr.shape[1] == NUM_EDGE_FEATURES
             if data_j.edge_attr.numel() > 0: assert data_j.edge_attr.shape[1] == NUM_EDGE_FEATURES
        else: print(f"  Batch {i} structure invalid.")
        batch_count += 1;
        if batch_count >= 2: break
    # (Validation loop remains the same)
    try: os.remove(dummy_file); print(f"\nRemoved dummy file: {dummy_file}")
    except OSError as e: print(f"Error removing dummy file {dummy_file}: {e}")
