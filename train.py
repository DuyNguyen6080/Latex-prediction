from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import torch
import math
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
import yaml
import os
import sys

ds = load_dataset("deepcopy/MathWriting-human")
print(ds)
print(ds["train"][0])

# === Data loading & preprocessing ===
# Trying out padded resizing of handwritten equations to avoid warping them
class PaddedResize:
    def __init__(self, h, w):
        self.target_h = h
        self.target_w = w

    def __call__(self, img):
        img = img.convert("L")
        w, h = img.size

        scale = min(self.target_w / w, self.target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = img.resize((new_w, new_h), Image.BICUBIC)

        # Create padded canvas
        canvas = Image.new("L", (self.target_w, self.target_h), color=0)
        offset_x = (self.target_w - new_w) // 2
        offset_y = (self.target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        return transforms.ToTensor()(canvas)

img_transform = transforms.Compose([PaddedResize(128, 512)])

class NotesLatexDataset(Dataset):
    def __init__(self, data, tokenizer, transform=None):
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"]
        if self.transform:
            img = self.transform(img)

        token_ids = self.tokenizer.encode(item["latex"], add_special_tokens=True)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        return img, token_ids

    def __len__(self): return len(self.data)



# Try using coding tokenizer to preserve non english characters
tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

train_ds = NotesLatexDataset(ds["train"], tokenizer, img_transform)
val_ds = NotesLatexDataset(ds["val"], tokenizer, img_transform)
test_ds = NotesLatexDataset(ds["test"], tokenizer, img_transform)
def collate_fn(batch):
    imgs, seqs = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    # Padded matrix for tokens
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :lengths[i]] = s

    return imgs, padded, torch.tensor(lengths)


train_ds = Subset(train_ds, range(2000))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

train_features, train_labels, c = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"c: {c.size()}")








# === Model definition ===
class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=3,
        dim_feedforward=512,
        pad_id=0,
    ):
        super().__init__()
        self.d_model = d_model

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Transformer for generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        return mask

    def encode_images(self, images):
        feat = self.cnn(images)
        batch_len, channels, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1)
        feat = feat.reshape(batch_len, h * w, channels)
        return feat

    def forward(self, images, tgt_input):
        src = self.encode_images(images)
        embedding = self.token_embed(tgt_input)
        tgt_mask = self.make_tgt_mask(tgt_input)
        decoded = self.decoder(tgt=embedding, memory=src, tgt_mask=tgt_mask)
        logits = self.output(decoded)
        return logits
    
def train (model: nn.Module, criterion, optimizer, epochs, epoch_loss_arr, losses):
  for ep in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    total_batches = 0

    for images, target_seq, lengths in train_loader:
        images = images.to(device)
        target_seq = target_seq.to(device)

        tgt_input = target_seq[:, :-1]
        gt = target_seq[:, 1:]

        logits = model(images, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), gt.reshape(-1),)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    epoch_loss = running_loss / max(1, total_batches)
    epoch_loss_arr.append(epoch_loss)
    print(f"Epoch {ep}   Loss = {epoch_loss:.4f}")

#---------loading configured model---------
def load_config(path):
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)
# === Training ===

import numpy as np
import matplotlib.pyplot as plt

# re-initialize model and analasis variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0



config = load_config("model_config.yaml")
print(config)

model_name = config["model_name"]
model_epochs = config["num_epochs"]
model_learning_rate = config["learning_rate"]

model = Model(vocab_size=tokenizer.vocab_size, pad_id=pad_id).to(device)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = optim.Adam(model.parameters(), lr=model_learning_rate)
epochs = 1

epoch_loss_arr = []
losses_arr = []
train (model, criterion, optimizer, model_epochs, epoch_loss_arr, losses_arr)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "pad_id": pad_id,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_name": "huggingface/CodeBERTa-small-v1",
        "n_epochs": epochs,
        "epoch_loss_arr": epoch_loss_arr,
        "losses_arr": losses_arr,
        
    },
    # trained_model.pt trained on subset
    # trained_model_2.pt trained on full dataset
    model_name,
)

#plot the loss over time


numpy_losses = np.array(losses_arr)
plt.plot(numpy_losses)
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.title("Loss over each iteration")
plt.show()

epoch_loss = np.array(epoch_loss_arr)
plt.plot(epoch_loss)
plt.xlabel("eposch")
plt.ylabel("Loss")
plt.title("Loss over each eposch")
plt.show()