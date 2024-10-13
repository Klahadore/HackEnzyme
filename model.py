import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence
import re
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# Load the tokenizer and the model
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
base_model = T5ForConditionalGeneration.from_pretrained('Rostlab/prot_t5_xl_uniref50')

# Enable gradient checkpointing
base_model.gradient_checkpointing_enable()

# Set up LoRA configuration with reduced rank to decrease parameter count
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # We are doing sequence-to-sequence language modeling
    inference_mode=False,
    r=4,  # Reduced rank of LoRA matrices to decrease parameter count
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
)

# Apply LoRA to the model
base_model = get_peft_model(base_model, peft_config)
base_model.train()  # Set model to training mode

# Define the custom model that includes a projection layer for reaction embeddings
class CustomModel(torch.nn.Module):
    def __init__(self, base_model, embedding_dim):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.projection = torch.nn.Linear(embedding_dim, base_model.config.d_model)
    
    def forward(self, reaction_embedding, labels=None, decoder_attention_mask=None):
        # Project reaction embedding to match model's expected input size
        inputs_embeds = self.projection(reaction_embedding).unsqueeze(1)  # Shape: (batch_size, 1, d_model)
        
        # Create attention mask for the encoder (since input length is 1)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)
        
        # Forward pass through the model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs

# Instantiate the custom model
embedding_dim = 384  # Replace with your reaction embedding dimension
model = CustomModel(base_model, embedding_dim)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the dataset class (adjusted)
class ReactionEmbeddingDataset(Dataset):
    def __init__(self, tokenizer, json_file):
        self.tokenizer = tokenizer
        # Load the data from the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            self.sequence_to_embedding = json.load(f)
        
        # Create a list of tuples (embedding, sequence)
        self.data = [(embedding, sequence) for sequence, embedding in self.sequence_to_embedding.items()]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        reaction_embedding, sequence = self.data[idx]

        # Convert the reaction embedding list to a PyTorch tensor
        reaction_embedding_tensor = torch.tensor(reaction_embedding, dtype=torch.float32)

        # Replace U, Z, O, B with 'X' and add spaces between amino acids
        sequence_replaced = re.sub(r"[UZOB]", "X", sequence)
        sequence_spaced = " ".join(sequence_replaced)

        # Tokenize the sequence with max_length and truncation
        tokenized = self.tokenizer(
            sequence_spaced,
            return_tensors='pt',
            padding='longest',
            max_length=512,  # Limit the sequence length
            truncation=True
        )

        # Get labels and decoder attention mask
        labels = tokenized['input_ids'].squeeze(0)
        decoder_attention_mask = tokenized['attention_mask'].squeeze(0)

        return reaction_embedding_tensor, labels, decoder_attention_mask

# Define the collate function
def collate_fn(batch):
    # Separate the reaction embeddings, labels, and decoder attention masks
    reaction_embeddings, labels, decoder_attention_masks = zip(*batch)
    
    # Stack reaction embeddings
    reaction_embeddings = torch.stack(reaction_embeddings)
    
    # Pad labels and decoder attention masks
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    decoder_attention_masks_padded = pad_sequence(decoder_attention_masks, batch_first=True, padding_value=0)
    
    return {
        'reaction_embedding': reaction_embeddings,
        'labels': labels_padded,
        'decoder_attention_mask': decoder_attention_masks_padded,
    }

# Training parameters
batch_size = 1
num_epochs = 2
learning_rate = 1e-4

# Prepare the dataset and dataloader
dataset = ReactionEmbeddingDataset(tokenizer, "sequence_to_embedding.json")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the optimizer (include parameters from LoRA and the projection layer)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop with bf16
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # Move data to device
        reaction_embedding = batch['reaction_embedding'].to(device)
        labels = batch['labels'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)
    
        # Zero the gradients
        optimizer.zero_grad()
    
        # Forward pass with bf16 autocast
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(
                reaction_embedding=reaction_embedding,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask
            )
            loss = outputs.loss
            print(loss)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()
        
        # Clear unnecessary variables and cache
        del reaction_embedding, labels, decoder_attention_mask, outputs, loss
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    # Clear cache after each epoch
    torch.cuda.empty_cache()

# Save the finetuned model
# Save only the LoRA adapters
torch.save(model.state_dict(), "finetuned_model/enzyme_model.bin")
# Save the tokenizer
tokenizer.save_pretrained("finetuned_prot_t5_with_lora")