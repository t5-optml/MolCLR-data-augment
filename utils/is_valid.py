from rdkit import Chem


def is_valid_molecule(mol):
    """
    Check if a molecule is valid.
    """
    try:
        Chem.SanitizeMol(mol)
        return True
    except Chem.rdchem.KekulizeException:
        return False
    except ValueError:
        return False
