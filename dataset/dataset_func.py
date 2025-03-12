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
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from dataset.abbrs import ABBREVIATIONS, ABBREVIATIONS_VOCAB, ABBREVIATION_PATTERNS
# from visualize import visualize_molecule


ATOM_DICT = {atom: idx for idx, atom in enumerate(range(1, 119))}

CHIRALITY_DICT = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3
}

BOND_DICT = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1, 
    BT.TRIPLE: 2,
    BT.AROMATIC: 3
}
BONDDIR_DICT = {
    Chem.rdchem.BondDir.NONE: 0, 
    Chem.rdchem.BondDir.ENDUPRIGHT: 1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2
}

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

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_DICT[atom.GetAtomicNum()])
            chirality_idx.append(CHIRALITY_DICT[atom.GetChiralTag()])
            atomic_number.append(atom.GetAtomicNum())
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_DICT[bond.GetBondType()],
                BONDDIR_DICT[bond.GetBondDir()]
            ])
            edge_feat.append([
                BOND_DICT[bond.GetBondType()],
                BONDDIR_DICT[bond.GetBondDir()]
            ])
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # --- NEW: Identify functional groups using the functional group (abbreviation) patterns ---
        func_groups = []
        for abbr, pattern in ABBREVIATION_PATTERNS.items():
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                func_groups.append(set(match))
        # Build a mapping from an atom index to its functional group (if any)
        atom_to_group = {}
        for group in func_groups:
            for atom_idx in group:
                atom_to_group[atom_idx] = group

        # Randomly mask a subgraph of the molecule with functional group consistency
        num_mask_nodes = max(1, math.floor(0.25 * N))
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        # Expand: if an atom in a functional group is chosen, mask the entire group.
        mask_nodes_i_expanded = set()
        for node in mask_nodes_i:
            if node in atom_to_group:
                mask_nodes_i_expanded.update(atom_to_group[node])
            else:
                mask_nodes_i_expanded.add(node)
        mask_nodes_i = list(mask_nodes_i_expanded)

        mask_nodes_j_expanded = set()
        for node in mask_nodes_j:
            if node in atom_to_group:
                mask_nodes_j_expanded.update(atom_to_group[node])
            else:
                mask_nodes_j_expanded.add(node)
        mask_nodes_j = list(mask_nodes_j_expanded)
        
        num_mask_edges = max(0, math.floor(0.25 * M))
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

        x_i = deepcopy(x)
        for atom_idx in mask_nodes_i:
            x_i[atom_idx, :] = torch.tensor([len(ATOM_DICT), 0])
        edge_index_i = torch.zeros((2, 2*(M - num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2*(M - num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:, count] = edge_index[:, bond_idx]
                edge_attr_i[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx, :] = torch.tensor([len(ATOM_DICT), 0])
        edge_index_j = torch.zeros((2, 2*(M - num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2*(M - num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:, count] = edge_index[:, bond_idx]
                edge_attr_j[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        
        return data_i, data_j

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


if __name__ == "__main__":
    # python dataset/dataset_abbre.py

    # data_path = 'data/pubchem-20k-sample.txt'
    # dataset = MoleculeDataset(data_path=data_path)
    # print(dataset.__getitem__(0))

    
    # mol = Chem.MolFromSmiles('CCC(C)CC(NN)C1CCC(C)O1') 
    # mol = Chem.MolFromSmiles('CCOc1ccccc1N1C(=O)C(=O)N(CC(=O)c2cc(C)c(C)cc2C)C1=O') #1
    mol = Chem.MolFromSmiles('Cc1ccnc(N2CCC(C)C2C(=O)[O-])c1[N+](=O)[O-]') #2
    # mol = Chem.MolFromSmiles('CCOC(=O)C(N=[N+]=[N-])C(c1ccccc1)[NH+]1CCOCC1') #3
    # mol = Chem.MolFromSmiles('Cc1cc(N2C(=O)C3CC=CC(C)C3C2=O)no1') #4

    print(ABBREVIATIONS_VOCAB)
    
    # img = visualize_molecule(mol, "original molecule")
    # save image
    # img.save("original_molecule.png")

    start = time.time()
    # mol = apply_functional_group_abbreviations(mol)
    # print("Mol:", Chem.MolToSmiles(mol))

    # mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles("C=CC"), Chem.MolFromSmiles("[119*]"), replaceAll=False)[0]
    # # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace(".", ""))
    # print("Mol:", Chem.MolToSmiles(mol))
    # img = visualize_molecule(mol, "molecule with abbreviations")

    # mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles("[119*]"), Chem.MolFromSmiles("C=CC"), replaceAll=True)[0]
    # # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace(".", ""))
    # print("Mol:", Chem.MolToSmiles(mol))
    # # remove . from the smiles of molecule
    # img = visualize_molecule(mol, "molecule with undo")


    end = time.time()
    print("Time taken to apply functional group abbreviations:", end-start)

