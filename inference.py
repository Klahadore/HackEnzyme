import json
import random

# Function to load the sequence data (including embeddings and SMILES strings)
def load_sequence_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        sequence_data = json.load(f)
    return sequence_data

# Function to return a random enzyme sequence and its SMILES string
def get_random_enzyme_sequence_and_smiles(sequence_data):
    # Get all sequences from the dataset
    sequences = list(sequence_data.keys())
    
    # Randomly select a sequence
    random_sequence = random.choice(sequences)
    
    # Retrieve the SMILES string associated with the sequence
    smiles_string = sequence_data[random_sequence]['smiles']
    
    return random_sequence, smiles_string

# Example usage
if __name__ == "__main__":
    # Load the sequence data from your dataset
    sequence_data = load_sequence_data("sequence_to_embedding.json")
    
    # Check if the dataset is not empty
    if not sequence_data:
        print("The sequence dataset is empty. Please provide a valid dataset.")
    else:
        # Get a random enzyme sequence and its SMILES string
        random_sequence, smiles_string = get_random_enzyme_sequence_and_smiles(sequence_data)
        
        # Print the random enzyme sequence and SMILES string
        print(f"Random Enzyme Sequence:\n{random_sequence}")
        print(f"Corresponding SMILES String:\n{smiles_string}")
