import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from torch.nn.utils.rnn import pad_sequence
import re
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# Load the tokenizer and the model
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
base_model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')

# Set up LoRA configuration with reduced rank to decrease parameter count
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # We are doing feature extraction
    inference_mode=False,
    r=4,  # Reduced rank of LoRA matrices to decrease parameter count
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
)

# Apply LoRA to the model
base_model = get_peft_model(base_model, peft_config)
base_model.train()  # Set model to training mode

# Define the custom model that includes a linear layer for dimensionality mapping
class CustomModel(torch.nn.Module):
    def __init__(self, base_model, output_dim):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(base_model.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Get the last hidden state and average over the sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
        # Map to the desired output dimension
        output = self.linear(embeddings)  # Shape: (batch_size, output_dim)
        return output

# Instantiate the custom model
model = CustomModel(base_model, output_dim=384)  # Set output_dim to match your embedding size

# Define the dataset class
class SequenceEmbeddingDataset(Dataset):
    def __init__(self, tokenizer, json_file, max_length=512):
        """
        Args:
            json_file (str): Path to the JSON file containing the sequence to embedding mapping.
            max_length (int): Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
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

        # Tokenize the sequence with truncation
        tokenized = self.tokenizer(
            sequence_spaced,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=False  # We'll handle padding in the collate_fn
        )

        # Convert the embedding list to a PyTorch tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

        # Get input IDs and attention mask from tokenizer output
        sequence_tensor = tokenized['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = tokenized['attention_mask'].squeeze(0)

        return embedding_tensor, sequence_tensor, attention_mask

# Define the collate function
def collate_fn(batch):
    # Separate the embeddings, sequences, and attention masks
    embeddings, sequences, attention_masks = zip(*batch)
    
    # Stack embeddings
    embeddings = torch.stack(embeddings)
    
    # Pad sequences and attention masks
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return embeddings, sequences_padded, attention_masks_padded

# Training parameters
batch_size = 1
num_epochs = 3
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model.to(device)

# Prepare the dataset and dataloader
dataset = SequenceEmbeddingDataset(tokenizer, "sequence_to_embedding.json", max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the optimizer (include parameters from LoRA and the linear layer)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize the gradient scaler for mixed precision
scaler = torch.cuda.amp.GradScaler()

# Training loop with mixed precision
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # Moved progress bar inside the loop
    for batch_idx, (embeddings, sequences, attention_masks) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # Move data to device
        embeddings = embeddings.to(device, non_blocking=True)
        sequences = sequences.to(device, non_blocking=True)
        attention_masks = attention_masks.to(device, non_blocking=True)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            model_output = model(input_ids=sequences, attention_mask=attention_masks)
            # Compute the loss
            loss = torch.nn.functional.mse_loss(model_output, embeddings)

        # Backward pass and optimization with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Delete unnecessary variables and empty cache
        del sequences, attention_masks, model_output
        torch.cuda.empty_cache()

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:  # Update progress every 10 batches
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {total_loss / (batch_idx + 1):.4f}")

        del loss

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
    torch.cuda.empty_cache()

# Save the finetuned model
# Save the custom model's state_dict
torch.save(model.state_dict(), "finetuned_model/enzyme_model.bin")
# Save the tokenizer
tokenizer.save_pretrained("finetuned_prot_t5_with_lora")
