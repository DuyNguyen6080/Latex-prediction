import argparse
import os
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import time
import sys
import matplotlib.pyplot as plt
from torch.amp import GradScaler
import yaml

torch.set_float32_matmul_precision("medium")
ds = load_dataset("deepcopy/MathWriting-human")

# Fix random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

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

img_transform = transforms.Compose([PaddedResize(96, 384)])

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

# === Model definition ===
class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=3,
        dim_feedforward=1024,
        pad_id=0,
    ):
        super().__init__()
        self.d_model = d_model

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.feature_norm = nn.LayerNorm(d_model)

        # Transformer for generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
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
        feat = self.feature_norm(feat)
        return feat

    def forward(self, images, tgt_input):
        src = self.encode_images(images)
        embedding = self.token_embed(tgt_input)
        tgt_mask = self.make_tgt_mask(tgt_input)
        decoded = self.decoder(tgt=embedding, memory=src, tgt_mask=tgt_mask)
        logits = self.output(decoded)
        return logits
    

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

#---------loading configured model---------
def load_config(path):
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config):
    # Try using coding tokenizer to preserve non english characters
    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

    train_ds = NotesLatexDataset(ds["train"], tokenizer, img_transform)

    
    config_details = load_config(config)
    model_name = config_details["model_name"]
    model_epochs = config_details["num_epochs"]
    model_learning_rate = config_details["learning_rate"]
    save_dir = config_details["save_dir"]
    samples = config_details["num_samples"]
    epochs = model_epochs
    learning_rate = float(model_learning_rate)

    if samples:
        if len(train_ds) < samples:
            print("Number of samplel is greater than dataset\n")
            exit() 
        train_ds = Subset(train_ds, torch.randperm(len(train_ds))[:int(samples)].tolist())
    else: 
        print("In config file sample must > 0\n")
        exit()
    
    if not os.path.exists(save_dir):
        # Use os.makedirs() to create the directory(ies). 
        # 'exist_ok=True' prevents an error if another part of 
        # your code might also try to create it.
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    print("config learning_rate: {learning_rate}\n")

    train_loader = DataLoader(train_ds,
                            batch_size=64,
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=2,
                            pin_memory=True,
                            persistent_workers=True)

    
    print("Training model...")
    # === Training ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = Model(vocab_size=tokenizer.vocab_size, pad_id=pad_id).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=0.01)

    

    track_loss = []

    scaler = GradScaler("cuda") if device.type == "cuda" else None
    model.train()

    for ep in range(1, epochs + 1):
        last_ep = time.time()
        running_loss = 0.0
        total_batches = 0
        for images, target_seq, _ in train_loader:
            images = images.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()

            tgt_input = target_seq[:, :-1]
            gt = target_seq[:, 1:]

            logits = model(images, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), gt.reshape(-1),)

            if not scaler:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()
            total_batches += 1

        epoch_loss = running_loss / max(1, total_batches)
        track_loss.append(epoch_loss)


        if not (ep % 10) :
            print(f"Epoch {ep}   Loss = {epoch_loss:.4f}   Time: {time.time() - last_ep}")

        last_ep = time.time()

    # Graph loss over training
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), track_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs")
    plt.grid(True)
    plt.savefig("loss_graphs/" + model_name + ".png", dpi=300, bbox_inches="tight")
    plt.close()

    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pad_id": pad_id,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_name": "huggingface/CodeBERTa-small-v1",
        },
        "models/" + model_name + ".pt",
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <model file name (Will be saved in /models directory)> [training set size (Must be <229,864), selects a random subset of training set]")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        
        args = parser.parse_args()
        config_file = args.config
        
        if len(sys.argv) == 3:
            sample = 1
            print(f"default num sample set is {sample}")
            main(config_file)
        