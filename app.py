from flask import Flask, jsonify
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import random
import json

app = Flask(__name__)

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

# Define your SequenceEmbeddingDataset class (as provided earlier)
class SequenceEmbeddingDataset(Dataset):
    def __init__(self, tokenizer, json_file):
        """
        Args:
            json_file (str): Path to the JSON file containing the sequence to embedding mapping.
        """
        self.tokenizer = tokenizer
        # Load the data from the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            self.sequence_to_embedding = json.load(f)
        
        # Convert the data into a list of tuples for easier indexing
        self.data = list(self.sequence_to_embedding.items())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence, embedding = self.data[idx]

        # Replace U, Z, O, B with 'X' and add spaces between amino acids
        sequence_replaced = re.sub(r"[UZOB]", "X", sequence)
        sequence_spaced = " ".join(sequence_replaced)

        # Tokenize the sequence
        tokenized = self.tokenizer(
            sequence_spaced,
            return_tensors='pt'
        )

        # Convert the embedding list to a PyTorch tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

        # Get input IDs from tokenizer output
        sequence_tensor = tokenized['input_ids'].squeeze(0)  # Remove batch dimension

        return embedding_tensor, sequence_tensor

def collate_fn(batch):
    # Separate the embeddings and sequences
    embeddings, sequences = zip(*batch)
    
    # Stack embeddings (assumed to be of the same size)
    embeddings = torch.stack(embeddings)
    
    # Pad sequences to the same length using tokenizer's pad token ID
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return embeddings, sequences_padded

# Initialize the dataset
dataset = SequenceEmbeddingDataset(tokenizer, "sequence_to_embedding.json")

# Endpoint to return a random enzyme sequence
@app.route('/random_sequence', methods=['GET'])
def get_random_enzyme_sequence():
    # Check if the dataset is not empty
    if len(dataset) == 0:
        return jsonify({'error': 'The enzyme dataset is empty.'}), 404

    # Get a random index
    random_idx = random.randint(0, len(dataset) - 1)

    # Get the sequence and embedding at the random index
    embedding_tensor, sequence_tensor = dataset[random_idx]

    # Decode the sequence tensor back to the amino acid sequence
    # Remove padding tokens if any (not needed here since sequences are not padded)
    sequence = tokenizer.decode(sequence_tensor, skip_special_tokens=True)
    # Remove spaces between amino acids
    sequence = re.sub(r" ", "", sequence)

    # Return the sequence as JSON
    return jsonify({'sequence': sequence})

if __name__ == '__main__':
    app.run(debug=True)
