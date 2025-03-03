from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

def visualize_molecule(mol, name="Molecule"):
    """
    Visualizes an RDKit molecule with custom labels for atoms that have an "abbr" property.
    
    Parameters:
        mol (Chem.Mol): The RDKit molecule.
        name (str): Title for the molecule image.
    
    Returns:
        PIL.Image.Image: An image of the molecule with custom labels for atoms with "abbr" property.
    """
    # Create a drawing object with the desired size
    drawer = rdMolDraw2D.MolDraw2DCairo(720, 720)
    
    # Prepare drawing options
    opts = drawer.drawOptions()
    
    # Collect atoms with "abbr" property
    highlight_atoms = []
    atom_labels = {}
    
    for atom in mol.GetAtoms():
        if atom.HasProp("abbr"):
            idx = atom.GetIdx()
            highlight_atoms.append(idx)
            atom_labels[idx] = atom.GetProp("abbr")
    
    # Set custom atom labels before drawing
    for idx, label in atom_labels.items():
        opts.atomLabels[idx] = label
    
    # Draw the molecule with highlighted atoms
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, legend=name)
    drawer.FinishDrawing()
    
    # Convert to PIL Image
    png_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png_data))
    
    # Display the image
    img.show()
    
    return img


def visualize_smiles(mol, name="Molecule"):
    """
    Visualizes a SMILES string with annotations for atoms that have an "abbr" property.
    
    Parameters:
        mol (str): The SMILES string.
    
    Returns:
        PIL.Image.Image: An image of the molecule with highlighted dummy atoms and their abbreviation labels.
    """
    mol = Chem.MolFromSmiles(mol)
    return visualize_molecule(mol, name=name)

if __name__ == "__main__":
    # Test the visualization function with a dummy molecule
    # debugging
    smiles = "CC1=C2C(=CC(=C1C(=O)OC)O)OC(=O)C3=C(Cl=C(C=C3O2)O)C"


    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    visualize_molecule(mol, name="Molecule")


    if mol:
        try:
            Chem.SanitizeMol(mol)
            print("Molecule sanitized successfully")
        except Exception as e:
            print("Sanitization failed:", e)
