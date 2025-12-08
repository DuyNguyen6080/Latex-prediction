from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import torch
import sys
import torch.nn as nn
from datasets import load_dataset
import nltk
nltk.download("punkt")
from nltk.translate.bleu_score import corpus_bleu
torch.set_float32_matmul_precision('medium')

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

def main(model):
    # Try using coding tokenizer to preserve non english characters
    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

    test_ds = NotesLatexDataset(ds["test"], tokenizer, img_transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

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
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("models/" + model, map_location=device, weights_only=False)
    pad_id = checkpoint["pad_id"]
    vocab_size = checkpoint["vocab_size"]

    model = Model(vocab_size=vocab_size, pad_id=pad_id).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_tokens = 0
    correct_tokens = 0
    refs = []
    pred_tokens = []

    with torch.no_grad():
        for images, target_seq, _ in test_loader:
            images = images.to(device)
            target_seq = target_seq.to(device)

            tgt_input = target_seq[:, :-1]
            gt = target_seq[:, 1:]

            logits = model(images, tgt_input)
            preds = logits.argmax(dim=-1)

            mask = gt != pad_id
            correct_tokens += (preds[mask] == gt[mask]).sum().item()
            total_tokens += mask.sum().item()

            gt_cpu = gt.cpu().tolist()
            preds_cpu = preds.cpu().tolist()

            # Takes out special tokens to get actual token
            for ref_seq, pred_seq in zip(gt_cpu, preds_cpu):
                ref_tex = tokenizer.decode(ref_seq, skip_special_tokens=True)
                pred_tex = tokenizer.decode(pred_seq, skip_special_tokens=True)

                if len(ref_tex) == 0 or len(pred_tex) == 0:
                    continue
                refs.append([ref_tex])
                pred_tokens.append(pred_tex)

    accuracy = correct_tokens / total_tokens
    bleu_score = corpus_bleu(refs, pred_tokens)
 
    print(f"Token level accuracy: {accuracy * 100:.2f}%")
    print(f"BLEU score: {bleu_score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval.py <model file (In models/ directory)>.pt")
    else:
        model = sys.argv[1]
        main(model)