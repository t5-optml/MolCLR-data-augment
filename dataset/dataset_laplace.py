import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolDrawOptions
import matplotlib.pyplot as plt

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path, remove_header=False):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if remove_header:
            next(csv_reader)
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


class MoleculeDataset(Dataset):
    def __init__(self, data_path, remove_header=False):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path, remove_header)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        # mol = Chem.AddHs(mol)
        
        if mol is None:
            return Data(), Data()  # Return empty data objects for invalid SMILES

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        
        type_idx = []
        chirality_idx = []
        atomic_number = []
        # aromatic = []
        # sp, sp2, sp3, sp3d = [], [], [], []
        # num_hs = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
 
            # aromatic.append(1 if atom.GetIsAromatic() else 0)
            # hybridization = atom.GetHybridization()
            # sp.append(1 if hybridization == HybridizationType.SP else 0)
            # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
        
        # z = torch.tensor(atomic_number, dtype=torch.long)
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
        #                     dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        
        degrees = torch.zeros(N, dtype=torch.long)
        for u, v in edge_index.t():
            degrees[u] += 1
            degrees[v] += 1  # Undirected edges are bidirectional
        
        bond_sum_degrees = []
        for bond_idx in range(M):
            bond = mol.GetBondWithIdx(bond_idx)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_sum_degrees.append(degrees[u].item() + degrees[v].item())
        if M > 0:
            bond_sum_degrees = torch.tensor(bond_sum_degrees, dtype=torch.float)
            probabilities = 1.0 / (bond_sum_degrees + 1e-6)
            # Add check for valid probabilities
            if not torch.isfinite(probabilities).all():
                probabilities = torch.ones_like(probabilities) / len(probabilities)
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = None

        # Determine number of edges to mask (25% of bonds)
        num_mask_edges = max(0, math.floor(0.25 * M))

        # Generate masked views for data_i and data_j
        def create_masked_view(probabilities, num_mask, edge_attr):
            if num_mask == 0 or len(probabilities) == 0:
                return deepcopy(x), edge_index.clone(), edge_attr.clone()
            
            # Edge type weights
            edge_type_weights = torch.tensor([
                1.0,   # Single bond
                1.2,   # Double bond
                1.5,   # Triple bond
                1.3    # Aromatic bond
            ], dtype=torch.float)
            
            # Ensure probabilities matches the number of unique bonds
            if probabilities is not None and len(probabilities) != edge_attr.shape[0] // 2:
                probabilities = torch.ones(edge_attr.shape[0] // 2, dtype=torch.float) / (edge_attr.shape[0] // 2)
            
            # Combine degree-based and edge type probabilities
            if probabilities is not None:
                # Select unique bond indexes and their corresponding weights
                unique_bond_indexes = edge_attr[::2, 0]
                enhanced_probabilities = probabilities * edge_type_weights[unique_bond_indexes]
                enhanced_probabilities /= enhanced_probabilities.sum()
            else:
                # If no probabilities, use uniform distribution with edge type weights
                unique_bond_indexes = edge_attr[::2, 0]
                enhanced_probabilities = edge_type_weights[unique_bond_indexes]
                enhanced_probabilities /= enhanced_probabilities.sum()
            
            # Sample bonds to mask with enhanced probabilities
            mask_bonds = torch.multinomial(enhanced_probabilities, num_mask, replacement=False).tolist()
            mask_edges = []
            for bond in mask_bonds:
                mask_edges.extend([2 * bond, 2 * bond + 1])  # Mask both directions

            # Create masked edge_index and edge_attr
            remaining_edges = [i for i in range(2*M) if i not in mask_edges]
            masked_edge_index = edge_index[:, remaining_edges]
            masked_edge_attr = edge_attr[remaining_edges]
            
            return deepcopy(x), masked_edge_index, masked_edge_attr

        # Create two perturbed views
        x_i, edge_index_i, edge_attr_i = create_masked_view(probabilities, num_mask_edges, edge_attr)
        x_j, edge_index_j, edge_attr_j = create_masked_view(probabilities, num_mask_edges, edge_attr)

        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)

        return data_i, data_j
    
        
    def visualize_sample(self, index, seed=42, save_path="molecule_augmentation.png"):
        """Visualizes and saves original vs augmented molecule as an image file."""
        random.seed(seed)
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        if mol is None:
            return None
        
        # Generate original graph data
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        
        # --- Original molecule processing ---
        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        # Original edge processing
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([BOND_LIST.index(bond.GetBondType()),
                            BONDDIR_LIST.index(bond.GetBondDir())])
            edge_feat.append([BOND_LIST.index(bond.GetBondType()),
                            BONDDIR_LIST.index(bond.GetBondDir())])
        edge_index = torch.tensor([row, col], dtype=torch.long)
        
        # --- Augmentation logic ---
        # Compute node degrees for edge masking
        degrees = torch.zeros(N, dtype=torch.long)
        for u, v in edge_index.t():
            degrees[u] += 1
            degrees[v] += 1

        # Calculate bond-based degree sums
        bond_sum_degrees = []
        for bond_idx in range(M):
            bond = mol.GetBondWithIdx(bond_idx)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_sum_degrees.append(degrees[u].item() + degrees[v].item())

        # Get edge masking probabilities
        if M > 0:
            probabilities = 1.0 / (torch.tensor(bond_sum_degrees, dtype=torch.float) + 1e-6)
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = None

        # Determine edges to mask (25% of bonds)
        num_mask_edges = max(0, math.floor(0.25 * M))
        mask_bonds = torch.multinomial(probabilities, num_mask_edges, replacement=False).tolist() if M > 0 else []

        # --- Visualization preparation ---
        # Create highlighted original molecule
        orig_mol = Chem.Mol(mol)
        highlight_bonds = []
        for bond_idx in mask_bonds:
            if bond_idx < len(orig_mol.GetBonds()):
                highlight_bonds.append(orig_mol.GetBondWithIdx(bond_idx).GetIdx())

        # Create augmented molecule (with edges removed)
        aug_mol = Chem.RWMol(mol)
        bonds_to_remove = sorted([orig_mol.GetBondWithIdx(bond_idx) for bond_idx in mask_bonds], 
                                key=lambda x: x.GetIdx(), reverse=True)
        for bond in bonds_to_remove:
            aug_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        # Configure drawing options
        draw_options = MolDrawOptions()
        draw_options.highlightBondWidthMultiplier = 20
        
        # Generate images
        orig_img = Draw.MolToImage(orig_mol, size=(400,400), 
                                 highlightBonds=highlight_bonds,
                                 drawOptions=draw_options)
        aug_img = Draw.MolToImage(aug_mol, size=(400,400))

        # Create a combined image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
        # Display original with highlights
        ax1.imshow(orig_img)
        ax1.set_title("Original Molecule\n(Red = Masked Edges)", pad=20)
        ax1.axis('off')
        
        # Display augmented version
        ax2.imshow(aug_img)
        ax2.set_title("Augmented Molecule", pad=20)
        ax2.axis('off')
        
        # Modify visualize_sample() save section:
        plt.tight_layout()

        # Ensure directory exists
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Also return the image objects in case you want to display in notebook
        return orig_img, aug_img, save_path

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path, remove_header=False):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.remove_header = remove_header

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path, remove_header=self.remove_header)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader
