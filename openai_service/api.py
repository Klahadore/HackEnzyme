from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from pydantic import BaseModel
from chembl_webresource_client.new_client import new_client

# Create a ChEMBL client
molecule_client = new_client.molecule

# Get the molecule by name
name = "aspirin"
molecules = molecule_client.search(name)

# Get the SMILES string for the first molecule
if molecules:
    smiles = molecules[0]['molecule_structures']['canonical_smiles']
    print(smiles)
else:
    print("Molecule not found.")
client = OpenAI()
class SMILES_Reaction(BaseModel):
    substrates: str
    products: str
    
class CHEMBL_Names(BaseModel):
    substrates: list(str)
    products: list(str)
    

def generate(prompt):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate the CHEMBL names associated with the enzyme catalyzed reaction. Generate the substrates and products. "},
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=SMILES_Reaction,

    )
    
print(generate("breaks down glucose"))
