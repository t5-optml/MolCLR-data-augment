import csv
import math
import random
import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.data import Data, Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def add_random_atoms(mol, atom_symbol="C", num_new_atoms=1):
    """
    Add a random atom to the molecule and connect it to a random existing atom
    """
    new_mol = Chem.RWMol(mol)

    # Add extra atoms
    for _ in range(num_new_atoms):
        # Get indices of carbons that are not saturated (valence < 4)
        carbon_indices = [i for i, atom in enumerate(mol.GetAtoms()) 
                        if atom.GetAtomicNum() == 6 and len(atom.GetNeighbors()) < 4]

        # If no carbons available, use all atoms
        existing_atom_indices = carbon_indices if carbon_indices else range(mol.GetNumAtoms())

        # Sample atom
        existing_atom_idx = random.choice(existing_atom_indices)

        # Add atom
        atom_idx = new_mol.AddAtom(Chem.Atom(atom_symbol))
        if new_mol.GetBondBetweenAtoms(atom_idx, existing_atom_idx) is None:
            new_mol.AddBond(atom_idx, existing_atom_idx, Chem.BondType.SINGLE)

    return new_mol.GetMol()


def read_smiles(data_path, remove_header=False):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
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

        # add random atoms
        num_new_atoms = max([1, math.floor(0.25 * N)])
        mol_i = add_random_atoms(mol, num_new_atoms=num_new_atoms)
        mol_j = add_random_atoms(mol, num_new_atoms=num_new_atoms)

        # get atom features for mol i
        type_idx_i, chirality_idx_i, atomic_number_i = [], [], []
        for atom in mol_i.GetAtoms():
            type_idx_i.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx_i.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number_i.append(atom.GetAtomicNum())
        x_i = torch.cat(
            [
                torch.tensor(type_idx_i, dtype=torch.long).view(-1, 1),
                torch.tensor(chirality_idx_i, dtype=torch.long).view(-1, 1),
            ],
            dim=-1,
        )
        row_i, col_i, edge_feat_i = [], [], []
        for bond in mol_i.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row_i += [start, end]
            col_i += [end, start]
            edge_feat_i.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
            edge_feat_i.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        # get atom features for mol j
        type_idx_j, chirality_idx_j, atomic_number_j = [], [], []
        for atom in mol_j.GetAtoms():
            type_idx_j.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx_j.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number_j.append(atom.GetAtomicNum())
        x_j = torch.cat(
            [
                torch.tensor(type_idx_j, dtype=torch.long).view(-1, 1),
                torch.tensor(chirality_idx_j, dtype=torch.long).view(-1, 1),
            ],
            dim=-1,
        )
        row_j, col_j, edge_feat_j = [], [], []
        for bond in mol_j.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row_j += [start, end]
            col_j += [end, start]
            edge_feat_j.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
            edge_feat_j.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)

        return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(
        self, batch_size, num_workers, valid_size, data_path, remove_header=False
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.remove_header = remove_header

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(
            data_path=self.data_path, remove_header=self.remove_header
        )
        train_loader, valid_loader = self.get_train_validation_data_loaders(
            train_dataset
        )
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return train_loader, valid_loader


if __name__ == "__main__":
    """
    Tests
    """
    from visualize import visualize_molecule

    mol = Chem.MolFromSmiles("CCC")
    visualize_molecule(mol, name="before_simple")
    new_mol = add_random_atoms(mol)
    visualize_molecule(new_mol, name="after_simple")

    mol = Chem.MolFromSmiles("O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C")
    visualize_molecule(mol, name="before_complex")
    new_mol = add_random_atoms(mol, num_new_atoms=5)
    visualize_molecule(new_mol, name="after_complex")

    # N = mol.GetNumAtoms()
    # M = mol.GetNumBonds()
    # print(N, M)

    # type_idx = []
    # chirality_idx = []
    # atomic_number = []
    # for atom in mol.GetAtoms():
    #     type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
    #     chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
    #     atomic_number.append(atom.GetAtomicNum())
    # print(type_idx)
    # print(chirality_idx)
    # print(atomic_number)

    # x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    # x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    # x = torch.cat([x1, x2], dim=-1)
    # print(x1)
    # print(x2)

    # row, col, edge_feat = [], [], []
    # for bond in mol.GetBonds():
    #     start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    #     row += [start, end]
    #     col += [end, start]
    #     edge_feat.append(
    #         [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
    #     )
    #     edge_feat.append(
    #         [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
    #     )
    # print(row)
    # print(col)
    # print(edge_feat)

    # edge_index = torch.tensor([row, col], dtype=torch.long)
    # edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    # print()
    # print("x", x)
    # print("edge_index", edge_index)
    # print("edge_attr", edge_attr)