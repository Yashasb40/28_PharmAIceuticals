import requests
import random
from rdkit import Chem
from rdkit.Chem import AllChem

# Define the disease of interest
disease = "respiratory"  # Replace with your specific disease of interest

# Define the ChEMBL API endpoint for compound search
chembl_api_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/"

# Define the search parameters for ChEMBL API
chembl_params = {
    "q": f"biotherapeutic_flag==False;disease_efo_id:{disease}",
    "format": "json",
}

# Make an HTTP GET request to the ChEMBL API
chembl_response = requests.get(chembl_api_url, params=chembl_params)

# Check if the ChEMBL API request was successful
if chembl_response.status_code == 200:
    # Parse the JSON response from ChEMBL
    chembl_data = chembl_response.json()

    # Extract SMILES data from the ChEMBL API response
    chembl_smiles_dataset = [entry["molecule_structures"]["canonical_smiles"] for entry in chembl_data["molecules"]]

    # Define a chemical reaction to convert C-C to C-O
    reaction_smarts = "[C:1]-[C:2]>>[C:1]-[O:2]"

    # Function to generate a new compound
    def generate_new_compound(dataset, max_attempts=10):
        for _ in range(max_attempts):
            # Randomly select a SMILES string from the dataset
            random_smiles = random.choice(dataset)

            # Create an RDKit molecule object from the selected SMILES
            molecule = Chem.MolFromSmiles(random_smiles)

            if molecule is not None:
                # Apply the chemical reaction to the molecule
                reaction = AllChem.ReactionFromSmarts(reaction_smarts)
                products = reaction.RunReactants((molecule,))
                if products:
                    # Extract the product molecule from the list of products
                    product_molecule = products[0][0]
                    return Chem.MolToSmiles(product_molecule)

        return None

    # Perform the compound generation process 10 times
    for _ in range(5):
        new_compound = generate_new_compound(chembl_smiles_dataset)
        if new_compound is not None:
            print(f"Generated compound for {disease}: {new_compound}")
        else:
            print(f"No valid compounds found for {disease}")

else:
    print(f"Failed to retrieve data from ChEMBL. Status code: {chembl_response.status_code}")
