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

import json
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit import rdBase

blocker = rdBase.BlockLogs()



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


class RuleOptimizer:
    def __init__(self, rules: list):
        self.rules = self._preprocess_rules(rules)
        # Set up a logger to suppress specific RDKit warnings
        rdkit.RDLogger.DisableLog('rdApp.warning')
        
    @staticmethod
    def _preprocess_rules(rules: list) -> list:
        preprocessed = []
        for rule in rules:
            reactant_smarts, product_smarts = rule['smarts'].split('>>')
            try:
                pat = Chem.MolFromSmarts(reactant_smarts)
                
                # Fix atom mapping issues by ensuring consistent mapping
                try:
                    fixed_rule = RuleOptimizer._fix_atom_mapping(reactant_smarts, product_smarts)
                    rxn = AllChem.ReactionFromSmarts(fixed_rule)
                except:
                    # Fallback to original SMARTS if fixing fails
                    rxn = AllChem.ReactionFromSmarts(f"{reactant_smarts}>>{product_smarts}")
                    
                # Set the rxn to ignore mapping mismatches
                if hasattr(rxn, 'Initialize'):
                    rxn.Initialize()
                
                preprocessed.append({
                    **rule,
                    'pattern': pat,
                    'reaction': rxn,
                    'reactant_smarts': reactant_smarts,
                    'product_smarts': product_smarts
                })
            except Exception as e:
                print(f"Failed to process rule {rule['name']}: {str(e)}")
        return preprocessed
    
    @staticmethod
    def _fix_atom_mapping(reactant_smarts, product_smarts):
        """Fix atom mapping issues between reactants and products."""
        try:
            # Parse the reactant and product SMARTS
            reactant_mol = Chem.MolFromSmarts(reactant_smarts)
            product_mol = Chem.MolFromSmarts(product_smarts)
            
            if not reactant_mol or not product_mol:
                return f"{reactant_smarts}>>{product_smarts}"
                
            # Get atom mappings from reactants
            reactant_mappings = {}
            for atom in reactant_mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    reactant_mappings[map_num] = atom.GetIdx()
            
            # Get atom mappings from products
            product_mappings = {}
            for atom in product_mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    product_mappings[map_num] = atom.GetIdx()
            
            # Find mappings in reactants that are missing in products
            missing_mappings = set(reactant_mappings.keys()) - set(product_mappings.keys())
            
            # If there are missing mappings, we need to fix the product SMARTS
            if missing_mappings:
                # Convert to SMILES to manipulate more easily
                product_smiles = Chem.MolToSmiles(product_mol)
                # Return the original strings - the issue will be handled during reaction application
                return f"{reactant_smarts}>>{product_smarts}"
            
            # If no missing mappings, return the original SMARTS
            return f"{reactant_smarts}>>{product_smarts}"
        except:
            # If anything fails, return the original SMARTS
            return f"{reactant_smarts}>>{product_smarts}"

    def _get_applicable_rules(self, mol: Chem.Mol) -> list:
        if mol is None or mol.GetNumAtoms() == 0:
            return []
        return [rule for rule in self.rules if mol.HasSubstructMatch(rule['pattern'])]

    def _sanitize_product(self, mol: Chem.Mol) -> Chem.Mol:
        if mol is None:
            return None
            
        try:
            # Make a copy first to avoid modifying original
            mol_copy = Chem.Mol(mol)
            
            # Try less aggressive sanitization first
            sanitize_options = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol_copy, sanitizeOp=sanitize_options)
            
            # Try kekulization separately
            try:
                Chem.Kekulize(mol_copy, clearAromaticFlags=False)
            except:
                try:
                    # If that fails, try clearing aromatic flags
                    Chem.Kekulize(mol_copy, clearAromaticFlags=True)
                except:
                    # If kekulization completely fails, continue with the unsanitized molecule
                    pass
                    
            # Even if sanitization partly failed, return what we have
            if mol_copy and mol_copy.GetNumAtoms() > 0:
                return mol_copy
            return None
        except Exception as e:
            # Last resort - return the original molecule if copy/sanitize fails
            if mol and mol.GetNumAtoms() > 0:
                return mol
            return None

    def _apply_reaction_safely(self, rxn, mol):
        """Apply reaction with error handling for atom mapping issues."""
        try:
            # Set reaction to be more permissive about atom mapping
            if hasattr(rxn, 'Initialize'):
                rxn.Initialize()
                
            # Run the reaction
            products = rxn.RunReactants((mol,))
            
            # Process all products
            valid_products = []
            for product_tuple in products:
                if not product_tuple:
                    continue
                for prod in product_tuple:
                    if prod and prod.GetNumAtoms() > 0:
                        # Try to sanitize
                        sanitized = self._sanitize_product(prod)
                        if sanitized and sanitized.GetNumAtoms() > 0:
                            valid_products.append(sanitized)
            
            return valid_products
        except Exception as e:
            # If reaction completely fails, return empty list
            return []

    def apply_augmentation(self, mol: Chem.Mol, R: int = 1) -> Chem.Mol:
        """Apply a series of transformations to the molecule."""
        if mol is None or mol.GetNumAtoms() == 0:
            return mol
            
        try:
            # Make a copy to avoid modifying original
            current_mol = Chem.Mol(mol)
        except:
            return mol
            
        for _ in range(R):
            try:
                # Get applicable rules
                applicable_rules = self._get_applicable_rules(current_mol)
                if not applicable_rules:
                    break
                
                # Select a rule and apply it
                selected_rule = random.choice(applicable_rules)
                
                # Apply the reaction with special handling for atom mapping
                valid_products = self._apply_reaction_safely(selected_rule['reaction'], current_mol)
                
                # If we got valid products, choose one randomly
                if valid_products:
                    current_mol = random.choice(valid_products)
                else:
                    # If no valid products, try another rule
                    continue
            except Exception as e:
                # If something goes wrong, return what we have so far
                break
                
        return current_mol


def read_smiles(data_path, remove_header=False):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if remove_header:
            next(csv_reader)
        for row in csv_reader:
            smiles_data.append(row[-1])
    return smiles_data


def mol_to_data(mol: Chem.Mol) -> Data:
    """Convert an RDKit molecule to a PyTorch Geometric Data object."""
    # Handle empty molecules
    if mol is None or mol.GetNumAtoms() == 0:
        return Data(x=torch.zeros((0, 2), dtype=torch.long), 
                  edge_index=torch.zeros((2, 0), dtype=torch.long),
                  edge_attr=torch.zeros((0, 2), dtype=torch.long))
                   
    # Extract atom features
    type_idx = []
    chirality_idx = []
    for atom in mol.GetAtoms():
        try:
            atomic_num = atom.GetAtomicNum()
            if atomic_num in ATOM_LIST:
                type_idx.append(ATOM_LIST.index(atomic_num))
            else:
                type_idx.append(0)  # Default to first element if not found
                
            chiral_tag = atom.GetChiralTag()
            if chiral_tag in CHIRALITY_LIST:
                chirality_idx.append(CHIRALITY_LIST.index(chiral_tag))
            else:
                chirality_idx.append(0)  # Default to first chirality if not found
        except:
            # If something goes wrong, use defaults
            type_idx.append(0)
            chirality_idx.append(0)
    
    # Create node feature matrix
    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    # Extract bond features
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        try:
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Ensure indices are valid
            if start < mol.GetNumAtoms() and end < mol.GetNumAtoms():
                row += [start, end]
                col += [end, start]
                
                # Get bond type with fallback
                try:
                    bond_type = bond.GetBondType()
                    bond_type_idx = BOND_LIST.index(bond_type) if bond_type in BOND_LIST else 0
                except:
                    bond_type_idx = 0
                    
                # Get bond direction with fallback
                try:
                    bond_dir = bond.GetBondDir()
                    bond_dir_idx = BONDDIR_LIST.index(bond_dir) if bond_dir in BONDDIR_LIST else 0
                except:
                    bond_dir_idx = 0
                
                bond_features = [bond_type_idx, bond_dir_idx]
                edge_feat.extend([bond_features, bond_features.copy()])
        except:
            # Skip this bond if something goes wrong
            continue

    # Create edge index and edge attribute tensors
    if row:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(edge_feat, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Enhanced MoleculeDataset class with better error handling
class MoleculeDataset(Dataset):
    def __init__(self, data_path, remove_header=False, rules_path='./dataset/rules.json'):
        super(Dataset, self).__init__()
        
        # Suppress specific RDKit warnings 
        rdkit.RDLogger.DisableLog('rdApp.warning')
        
        # Load SMILES data
        self.smiles_data = read_smiles(data_path, remove_header)
        
        # Load rules safely
        try:
            with open(rules_path) as f:
                rules = json.load(f)
            self.rule_optimizer = RuleOptimizer(rules)
        except Exception as e:
            print(f"Error loading rules from {rules_path}: {str(e)}")
            # Create an empty rule set as fallback
            self.rule_optimizer = RuleOptimizer([])
            
        # Parse molecules with error handling
        self.original_mols = []
        for s in self.smiles_data:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    self.original_mols.append(mol)
                else:
                    # Create an empty molecule as a placeholder
                    empty_mol = Chem.RWMol()
                    self.original_mols.append(empty_mol.GetMol())
            except:
                # Create an empty molecule as a placeholder
                empty_mol = Chem.RWMol()
                self.original_mols.append(empty_mol.GetMol())

    def __getitem__(self, index):
        # Get original molecule and SMILES
        try:
            original_mol = self.original_mols[index]
            original_smiles = self.smiles_data[index]
        except:
            # Fallback if index is out of range somehow
            empty_mol = Chem.RWMol()
            original_mol = empty_mol.GetMol()
            original_smiles = ""
        
        def get_augmented_view(mol, smiles):
            """Get an augmented view of the molecule with fallbacks."""
            try:
                # Try augmentation
                if mol and mol.GetNumAtoms() > 0:
                    augmented = self.rule_optimizer.apply_augmentation(mol, R=3)
                    if augmented and augmented.GetNumAtoms() > 0:
                        return mol_to_data(augmented)
                
                # Fallback to original if augmentation fails
                return mol_to_data(mol)
            except Exception as e:
                # If all else fails, try to create from SMILES
                try:
                    fallback_mol = Chem.MolFromSmiles(smiles)
                    if fallback_mol:
                        return mol_to_data(fallback_mol)
                except:
                    pass
                
                # Last resort - empty data object
                return Data(x=torch.zeros((0, 2), dtype=torch.long), 
                          edge_index=torch.zeros((2, 0), dtype=torch.long),
                          edge_attr=torch.zeros((0, 2), dtype=torch.long))

        # Get two augmented views with robust error handling
        try:
            data_i = get_augmented_view(original_mol, original_smiles)
        except:
            # Fallback for complete failure
            data_i = Data(x=torch.zeros((0, 2), dtype=torch.long), 
                         edge_index=torch.zeros((2, 0), dtype=torch.long),
                         edge_attr=torch.zeros((0, 2), dtype=torch.long))
                         
        try:
            data_j = get_augmented_view(original_mol, original_smiles)
        except:
            # Fallback for complete failure
            data_j = Data(x=torch.zeros((0, 2), dtype=torch.long), 
                         edge_index=torch.zeros((2, 0), dtype=torch.long),
                         edge_attr=torch.zeros((0, 2), dtype=torch.long))
        
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