import numpy as np, torch, torch.nn as nn, librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

np.random.seed(42); torch.manual_seed(42)

DATA_DIR = Path(r"C:\Users\SATYARANJAN\OneDrive\Desktop\tasks\task2\TESS Toronto emotional speech set data")
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 200
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

wavs = [str(p) for p in DATA_DIR.rglob("*.wav")]
labels = [Path(p).parent.name.lower().split("_")[-1] for p in wavs]
classes = sorted(sorted(set(labels)), key=lambda x: x)
cls2idx = {c:i for i,c in enumerate(classes)}

def load_mfcc(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, d1, d2]).T
    if feat.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - feat.shape[0], feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    else:
        feat = feat[:MAX_FRAMES, :]
    return feat.astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(wavs, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

class SERDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files; self.labels = labels
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        x = load_mfcc(self.files[idx])
        y = cls2idx[self.labels[idx]]
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

train_ds = SERDataset(X_train, y_train)
val_ds = SERDataset(X_val, y_val)
test_ds = SERDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class CNNBiLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (5,5), padding=(2,2)), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, (3,3), padding=(1,1)), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2))
        )
        self.lstm = nn.LSTM(input_size=32*((3*N_MFCC)//4), hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))
    def forward(self, x):
        b, c, t, f = x.shape
        x = self.cnn(x)
        b, c2, t2, f2 = x.shape
        x = x.permute(0,2,1,3).contiguous().view(b, t2, c2*f2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out)

model = CNNBiLSTM(len(classes)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()

best_val = 0.0
for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        preds = logits.argmax(1)
        total += yb.size(0); correct += (preds==yb).sum().item()
    model.eval()
    vtotal, vcorrect = 0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(1)
            vtotal += yb.size(0); vcorrect += (preds==yb).sum().item()
    vacc = vcorrect / max(1, vtotal)
    if vacc > best_val:
        best_val = vacc
        torch.save({"state_dict": model.state_dict(), "classes": classes}, "tess_ser_best.pt")
    print(f"epoch {epoch+1}/{EPOCHS} train_acc {correct/max(1,total):.3f} val_acc {vacc:.3f}")

ckpt = torch.load("tess_ser_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(1).cpu().numpy().tolist()
        y_pred += preds
        y_true += yb.numpy().tolist()
inv = {v:k for k,v in cls2idx.items()}
y_pred_labels = [inv[i] for i in y_pred]
y_true_labels = [inv[i] for i in y_true]
print(classification_report(y_true_labels, y_pred_labels, labels=classes))
print(confusion_matrix(y_true_labels, y_pred_labels, labels=classes))
