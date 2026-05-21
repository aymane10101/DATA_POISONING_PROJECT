# ============================================================
# CELL 0: ENVIRONMENT SETUP & IMPORTS
# ============================================================

import sys, os, time, random, copy, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch : {torch.__version__}")
print(f"NumPy   : {np.__version__}")
print(f"Pandas  : {pd.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device  : {device}")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
print("\n✓ All imports successful")

# ============================================================
# CELL 1: REPRODUCIBILITY — FIXED SEEDS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ All seeds fixed to {seed}")

set_seed(42)

# ============================================================
# CELL 2: LOAD UNSW-NB15 DATA
# ============================================================


DATA_DIR = "C:/Users/ayman/OneDrive/Desktop"   # ← change if needed

df_train = pd.read_csv(f"{DATA_DIR}/UNSW_NB15_training-set.csv")
df_test  = pd.read_csv(f"{DATA_DIR}/UNSW_NB15_testing-set.csv")

print(f"Training : {df_train.shape[0]:,} rows × {df_train.shape[1]} cols")
print(f"Testing  : {df_test.shape[0]:,} rows × {df_test.shape[1]} cols")

df = pd.concat([df_train, df_test], ignore_index=True)
print(f"Combined : {df.shape[0]:,} rows")

print("\nClass distribution:")
print(df['label'].value_counts())
print("\nAttack categories:")
print(df['attack_cat'].value_counts())

# ============================================================
# CELL 3: PREPROCESSING
# ============================================================

if 'id' in df.columns:
    df = df.drop('id', axis=1)

y_binary = df['label'].values
X_raw    = df.drop(['label', 'attack_cat'], axis=1)

categorical_features = ['proto', 'service', 'state']
X_encoded = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True)

X_numpy = X_encoded.values.astype(np.float32)
input_dim = X_numpy.shape[1]
print(f"Features after encoding: {input_dim}")

# 70 / 15 / 15 split  (stratified)
X_temp,  X_test,  y_temp,  y_test  = train_test_split(
    X_numpy, y_binary, test_size=0.15, random_state=42, stratify=y_binary)
X_train, X_val,   y_train, y_val   = train_test_split(
    X_temp,  y_temp,  test_size=0.1765, random_state=42, stratify=y_temp)

print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

def make_tensors(X, y, dev=device):
    return (torch.FloatTensor(X).to(dev),
            torch.LongTensor(y).to(dev))

X_train_t, y_train_t = make_tensors(X_train_scaled, y_train)
X_val_t,   y_val_t   = make_tensors(X_val_scaled,   y_val)
X_test_t,  y_test_t  = make_tensors(X_test_scaled,  y_test)

def make_loader(X_t, y_t, batch_size=1024, shuffle=True):
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

train_loader = make_loader(X_train_t, y_train_t)
val_loader   = make_loader(X_val_t,   y_val_t,   shuffle=False)
test_loader  = make_loader(X_test_t,  y_test_t,  shuffle=False)

print("✓ Data splits and loaders ready")

# ============================================================
# CELL 4: ARCHITECTURE — AUTOENCODER + MLP CLASSIFIER
# ============================================================
#
# Design rationale
# ─────────────────
# • ENCODER compresses raw features → dense latent representation
# • DECODER reconstructs the input  → regularises the encoder and
#   crucially gives us a reconstruction error we can exploit later
#   as a poisoning-detection signal
# • CLASSIFIER operates on the latent space, not raw features
#
# During training two losses are combined:
#   total_loss = CE_classification + α × MSE_reconstruction
#
# This forces the encoder to keep information useful for BOTH
# tasks, making the latent space richer and more robust.
# ============================================================

class AutoencoderMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 encoder_dims=[512, 256, 128],
                 latent_dim=64,
                 classifier_dims=[64, 32],
                 dropout_rate=0.25):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        enc = []
        prev = input_dim
        for d in encoder_dims:
            enc += [nn.Linear(prev, d), nn.BatchNorm1d(d),
                    nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = d
        enc += [nn.Linear(prev, latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        # ── Decoder ──────────────────────────────────────────
        dec = []
        prev = latent_dim
        for d in reversed(encoder_dims):
            dec += [nn.Linear(prev, d), nn.BatchNorm1d(d),
                    nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = d
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

        # ── Classifier (head) ────────────────────────────────
        clf = []
        prev = latent_dim
        for d in classifier_dims:
            clf += [nn.Linear(prev, d), nn.BatchNorm1d(d),
                    nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = d
        clf += [nn.Linear(prev, 2)]           # 2 classes: normal / attack
        self.classifier = nn.Sequential(*clf)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return logits, x_hat

    def encode(self, x):
        return self.encoder(x)

    def reconstruction_error(self, x):
        """Per-sample MSE — used by the defense detector."""
        with torch.no_grad():
            _, x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=1)


# Quick sanity check
_dummy = torch.randn(4, input_dim)
_model = AutoencoderMLP(input_dim)
_logits, _xhat = _model(_dummy)
n_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f"AutoencoderMLP — input_dim={input_dim}")
print(f"Latent dim  : 64")
print(f"Trainable params: {n_params:,}")
print(f"Logits shape: {_logits.shape}  |  Reconstruction shape: {_xhat.shape}")
del _dummy, _model
print("✓ Architecture OK")

# ============================================================
# CELL 5: SHARED TRAINING FUNCTION
# ============================================================

def train_ae_mlp(model, train_loader, val_loader,
                 epochs=400, lr=3e-4, weight_decay=1e-4,
                 alpha_rec=0.3, patience=60, device=device,
                 verbose_every=25):
    """
    Train AutoencoderMLP with combined classification + reconstruction loss.

    Parameters
    ----------
    alpha_rec : float
        Weight for reconstruction loss.  0 = pure classifier,
        1 = equal weight to both terms.
    patience  : int
        Early stopping patience (on val total loss).
    """
    model = model.to(device)
    crit_cls = nn.CrossEntropyLoss()
    crit_rec = nn.MSELoss()
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    history = defaultdict(list)
    best_val, best_state, no_improve = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        t_cls, t_rec, t_correct, t_total = 0, 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, x_hat = model(bx)
            loss_cls = crit_cls(logits, by)
            loss_rec = crit_rec(x_hat, bx)
            loss = loss_cls + alpha_rec * loss_rec
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_cls += loss_cls.item(); t_rec += loss_rec.item()
            preds = logits.argmax(1)
            t_correct += (preds == by).sum().item(); t_total += len(by)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        v_cls, v_rec = 0, 0
        v_preds_all, v_true_all = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits, x_hat = model(bx)
                v_cls += crit_cls(logits, by).item()
                v_rec += crit_rec(x_hat, bx).item()
                v_preds_all.extend(logits.argmax(1).cpu().numpy())
                v_true_all.extend(by.cpu().numpy())

        n_tr, n_vl = len(train_loader), len(val_loader)
        tr_loss = t_cls/n_tr + alpha_rec*(t_rec/n_tr)
        vl_loss = v_cls/n_vl + alpha_rec*(v_rec/n_vl)
        vl_f1   = f1_score(v_true_all, v_preds_all, average='macro', zero_division=0)
        tr_acc  = t_correct / t_total

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_f1'].append(vl_f1)
        history['train_acc'].append(tr_acc)
        history['lr'].append(opt.param_groups[0]['lr'])

        sched.step(vl_loss)

        if vl_loss < best_val:
            best_val  = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % verbose_every == 0:
            print(f"[{epoch:4d}/{epochs}]  "
                  f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
                  f"vl_F1={vl_f1:.4f}  acc={tr_acc:.4f}")

        if no_improve >= patience:
            print(f"⏹ Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, dict(history)

# ============================================================
# CELL 6: EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader, device=device, name="Model"):
    model.to(device).eval()
    probs_all, preds_all, true_all = [], [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            logits, _ = model(bx)
            pr = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            pd_ = logits.argmax(1).cpu().numpy()
            probs_all.extend(pr)
            preds_all.extend(pd_)
            true_all.extend(by.cpu().numpy())

    y_true = np.array(true_all)
    y_pred = np.array(preds_all)
    y_prob = np.array(probs_all)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    roc  = roc_auc_score(y_true, y_prob)
    pc, rc, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rc, pc)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Accuracy      {acc:.4f}    Precision   {prec:.4f}")
    print(f"  Recall        {rec:.4f}    F1-Score    {f1:.4f}")
    print(f"  ROC-AUC       {roc:.4f}    PR-AUC      {pr_auc:.4f}  ⭐")
    print(f"  FNR (missed)  {fnr:.4f}  ⚠   FPR         {fpr:.4f}")
    print(f"  Confusion: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                roc=roc, pr_auc=pr_auc, fnr=fnr, fpr=fpr,
                cm=cm, tn=tn, fp=fp, fn=fn, tp=tp,
                y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                pc=pc, rc=rc)

# ============================================================
# CELL 7: TRAIN BASELINE MODEL (clean data, AutoencoderMLP)
# ============================================================

set_seed(42)
baseline_model = AutoencoderMLP(input_dim).to(device)

print("Training baseline on CLEAN data …")
t0 = time.time()
baseline_model, baseline_history = train_ae_mlp(
    baseline_model, train_loader, val_loader,
    epochs=400, lr=3e-4, alpha_rec=0.3, patience=60)
print(f"Done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# CELL 8: EVALUATE BASELINE
# ============================================================

baseline_results = evaluate(baseline_model, test_loader, device,
                             "Baseline AutoencoderMLP (clean data)")

# Store results dict for later comparison
ALL_RESULTS = {"Baseline": baseline_results}
ALL_HISTORIES = {"Baseline": baseline_history}

# ============================================================
# CELL 9: BASELINE VISUALISATION
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ep = range(1, len(baseline_history['train_loss'])+1)

axes[0].plot(ep, baseline_history['train_loss'], label='Train')
axes[0].plot(ep, baseline_history['val_loss'],   label='Val')
axes[0].set(title='Loss Curves (Baseline)', xlabel='Epoch', ylabel='Loss')
axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(ep, baseline_history['val_f1'], color='green')
axes[1].set(title='Val Macro-F1 (Baseline)', xlabel='Epoch', ylabel='F1')
axes[1].grid(alpha=.3)

axes[2].plot(baseline_results['rc'], baseline_results['pc'], color='blue', lw=2)
axes[2].fill_between(baseline_results['rc'], baseline_results['pc'], alpha=.15)
axes[2].set(title=f'PR Curve (AUC={baseline_results["pr_auc"]:.4f})',
            xlabel='Recall', ylabel='Precision')
axes[2].grid(alpha=.3)

plt.suptitle('Baseline AutoencoderMLP — Clean Training', fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# CELL 10: ATTACK 1 — CLEAN-LABEL FEATURE POISONING 10 %
# ============================================================
# Threat model  : Attacker can modify FEATURES but NOT labels.
# Strategy      : Push attack-sample features toward the normal
#                 centroid, so the model sees "normal-like" features
#                 still labelled "attack" — confused decision boundary.
# Stealthiness  : Very high — all labels remain correct, making
#                 label-audit defenses ineffective.
# ============================================================

def clean_label_feature_poison(X, y, poison_rate=0.10, epsilon=0.3, seed=42):
    np.random.seed(seed)
    X_p = X.copy()
    atk_idx = np.where(y == 1)[0]
    n = int(len(atk_idx) * poison_rate)
    chosen = np.random.choice(atk_idx, size=n, replace=False)

    normal_centroid = X[y == 0].mean(axis=0)

    for i in chosen:
        direction = normal_centroid - X_p[i]
        X_p[i] += epsilon * direction
        X_p[i] += np.random.normal(0, 0.05, X_p[i].shape)

    X_p = np.clip(X_p, X.min(0), X.max(0))
    avg_l2 = np.linalg.norm(X_p[chosen] - X[chosen], axis=1).mean()

    print(f"Clean-Label Feature Poisoning {poison_rate:.0%}")
    print(f"  Poisoned attacks : {n:,}  |  avg L2 change : {avg_l2:.4f}")
    print(f"  Labels stay CORRECT — stealthy!")
    return X_p, y.copy(), chosen

X_train_cl10, y_train_cl10, _ = clean_label_feature_poison(
    X_train_scaled, y_train, poison_rate=0.10, epsilon=0.3, seed=42)

loader_cl10 = make_loader(
    torch.FloatTensor(X_train_cl10).to(device),
    torch.LongTensor(y_train_cl10).to(device))

set_seed(42)
model_cl10 = AutoencoderMLP(input_dim).to(device)
print("\nTraining on CLEAN-LABEL FEATURE POISON 10% data …")
t0 = time.time()
model_cl10, hist_cl10 = train_ae_mlp(
    model_cl10, loader_cl10, val_loader,
    epochs=400, lr=3e-4, alpha_rec=0.3, patience=60)
print(f"Done in {(time.time()-t0)/60:.1f} min")

res_cl10 = evaluate(model_cl10, test_loader, device,
                    "Attack 1 — Clean-Label Feature Poison 10%")
ALL_RESULTS["CleanLabel10%"]  = res_cl10
ALL_HISTORIES["CleanLabel10%"] = hist_cl10

# ============================================================
# CELL 11: ATTACK 2 — RANDOM LABEL FLIP (10 %)
# ============================================================
# Threat model  : Attacker with write-access to training labels.
# Strategy      : Randomly flip 10 % of ALL labels (both classes).
# Stealthiness  : Moderate — validation metrics may degrade visibly.
# Expected effect: General accuracy drop, elevated FNR and FPR.
# ============================================================

def random_label_flip(y, flip_rate=0.10, seed=42):
    np.random.seed(seed)
    y_p = y.copy()
    idx = np.random.choice(len(y), size=int(len(y)*flip_rate), replace=False)
    y_p[idx] = 1 - y_p[idx]
    n_flipped = (y_p != y).sum()
    print(f"Random Label Flip {flip_rate:.0%}")
    print(f"  Total samples  : {len(y):,}")
    print(f"  Labels flipped : {n_flipped:,} ({n_flipped/len(y):.2%})")
    return y_p, idx

y_train_rnd10, _ = random_label_flip(y_train, flip_rate=0.10, seed=42)

loader_rnd10 = make_loader(
    torch.FloatTensor(X_train_scaled).to(device),
    torch.LongTensor(y_train_rnd10).to(device))

set_seed(42)
model_rnd10 = AutoencoderMLP(input_dim).to(device)
print("\nTraining on RANDOM FLIP 10% poisoned data …")
t0 = time.time()
model_rnd10, hist_rnd10 = train_ae_mlp(
    model_rnd10, loader_rnd10, val_loader,
    epochs=400, lr=3e-4, alpha_rec=0.3, patience=60)
print(f"Done in {(time.time()-t0)/60:.1f} min")

res_rnd10 = evaluate(model_rnd10, test_loader, device,
                     "Attack 2 — Random Label Flip 10%")
ALL_RESULTS["RandomFlip10%"]  = res_rnd10
ALL_HISTORIES["RandomFlip10%"] = hist_rnd10

# ============================================================
# CELL 12: ATTACK 3 — TARGETED LABEL FLIP 10 % (Attack→Normal)
# ============================================================
# Threat model  : Attacker flips only ATTACK labels to Normal.
# Strategy      : Model learns "these attack features = normal"
#                 → silently raises FNR (missed attacks) at test time.
# Stealthiness  : High — normal-class accuracy stays intact, so
#                 validation metrics look healthy.
# ============================================================

def targeted_label_flip(y, flip_rate=0.10, seed=42):
    np.random.seed(seed)
    y_p = y.copy()
    atk_idx = np.where(y == 1)[0]
    n = int(len(atk_idx) * flip_rate)
    chosen = np.random.choice(atk_idx, size=n, replace=False)
    y_p[chosen] = 0
    print(f"Targeted Label Flip (Attack→Normal) {flip_rate:.0%}")
    print(f"  Attack samples total : {len(atk_idx):,}")
    print(f"  Flipped to Normal    : {n:,} ({flip_rate:.1%} of attacks)")
    print(f"  Overall poison rate  : {n/len(y):.2%}")
    return y_p, chosen

y_train_tgt10, _ = targeted_label_flip(y_train, flip_rate=0.10, seed=42)

loader_tgt10 = make_loader(
    torch.FloatTensor(X_train_scaled).to(device),
    torch.LongTensor(y_train_tgt10).to(device))

set_seed(42)
model_tgt10 = AutoencoderMLP(input_dim).to(device)
print("\nTraining on TARGETED FLIP 10% poisoned data …")
t0 = time.time()
model_tgt10, hist_tgt10 = train_ae_mlp(
    model_tgt10, loader_tgt10, val_loader,
    epochs=400, lr=3e-4, alpha_rec=0.3, patience=60)
print(f"Done in {(time.time()-t0)/60:.1f} min")

res_tgt10 = evaluate(model_tgt10, test_loader, device,
                     "Attack 3 — Targeted Flip 10% (Atk→Norm)")
ALL_RESULTS["TargetedFlip10%"]  = res_tgt10
ALL_HISTORIES["TargetedFlip10%"] = hist_tgt10

# ============================================================
# CELL 13: ATTACK SUMMARY TABLE
# ============================================================

metrics = ['acc', 'f1', 'pr_auc', 'fnr', 'fpr']
labels  = ['Accuracy', 'F1', 'PR-AUC', 'FNR ↑bad', 'FPR']

print(f"\n{'Model':<28}", end="")
for m in labels: print(f"{m:>12}", end="")
print()
print("-" * 88)
for name, r in ALL_RESULTS.items():
    print(f"{name:<28}", end="")
    for m in metrics:
        delta = ""
        if name != "Baseline":
            d = r[m] - ALL_RESULTS["Baseline"][m]
            delta = f"({d:+.3f})"
        val = r[m]
        print(f"{val:>8.4f}{delta:>4}", end="")
    print()

# ============================================================
# CELL 14: MULTI-ATTACK VISUALISATION
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = {'Baseline':'#2ecc71', 'RandomFlip10%':'#e74c3c',
          'TargetedFlip10%':'#e67e22', 'CleanLabel10%':'#9b59b6'}
labels_map = {
    'Baseline'        : 'Baseline (clean)',
    'CleanLabel10%'   : 'Atk1: Clean-Label 10%',
    'RandomFlip10%'   : 'Atk2: Random Flip 10%',
    'TargetedFlip10%' : 'Atk3: Targeted Flip 10%',
}

# PR Curves
for k, r in ALL_RESULTS.items():
    axes[0].plot(r['rc'], r['pc'], color=colors[k], lw=2,
                 label=f"{labels_map[k]} (AUC={r['pr_auc']:.3f})")
axes[0].set(title='PR Curves — All Conditions', xlabel='Recall', ylabel='Precision')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

# FNR bar chart
names = list(ALL_RESULTS.keys())
fnrs  = [ALL_RESULTS[k]['fnr'] for k in names]
bars  = axes[1].bar(names, fnrs, color=[colors[k] for k in names], alpha=.8, edgecolor='k')
axes[1].set(title='FNR — Missed Attacks (lower=better)',
            ylabel='False Negative Rate')
axes[1].set_xticklabels([labels_map[n] for n in names], rotation=15, ha='right', fontsize=8)
for b, v in zip(bars, fnrs):
    axes[1].text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9)
axes[1].grid(alpha=.3, axis='y')

# F1 bar chart
f1s = [ALL_RESULTS[k]['f1'] for k in names]
bars = axes[2].bar(names, f1s, color=[colors[k] for k in names], alpha=.8, edgecolor='k')
axes[2].set(title='F1-Score (higher=better)', ylabel='F1')
axes[2].set_ylim(max(0, min(f1s)-0.05), 1.01)
axes[2].set_xticklabels([labels_map[n] for n in names], rotation=15, ha='right', fontsize=8)
for b, v in zip(bars, f1s):
    axes[2].text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9)
axes[2].grid(alpha=.3, axis='y')

plt.suptitle('Data Poisoning — Attack Impact Overview', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# CELL 15: DEFENSE — LOSS TRAJECTORY + RECONSTRUCTION ANOMALY
# ============================================================
#
# ─────────────────────────────────────────────────────────────
# 1. LOSS TRAJECTORY: Poisoned samples (attack→normal) are
#    "easier" to fit because the model can memorize the wrong
#    mapping. Their loss drops much faster than clean samples
#    in early epochs. Clean samples need the model to actually
#    LEARN meaningful representations.
#
# 2. RECONSTRUCTION ERROR: In our AutoencoderMLP, poisoned
#    samples create a latent representation conflict —
#    attack features compressed into a space the classifier
#    head maps to "normal." This tension makes reconstruction
#    anomalous compared to genuinely normal samples.
#
# 3. NO CIRCULAR DEPENDENCY: We train a SINGLE scout model
#    on the poisoned data and analyze its per-sample BEHAVIOR
#    (loss trajectory + reconstruction), not its predictions.
#    The scout model's corruption doesn't hide these signals.
# ─────────────────────────────────────────────────────────────

def loss_trajectory_analysis(model, X_t, y_t, epochs=50, batch_size=1024, device=device):
    """
    Track per-sample loss over early training epochs.
    Poisoned samples will show suspiciously fast loss decay.
    """
    model = model.to(device)
    crit_cls = nn.CrossEntropyLoss(reduction='none')  # per-sample loss
    crit_rec = nn.MSELoss(reduction='none')
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    n_samples = len(y_t)
    loss_history = np.zeros((n_samples, epochs))

    # Create a loader that preserves sample indices
    dataset = TensorDataset(X_t, y_t, torch.arange(n_samples))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        model.train()
        epoch_losses = np.zeros(n_samples)
        for bx, by, bidx in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, x_hat = model(bx)
            loss_cls = crit_cls(logits, by)
            loss_rec = crit_rec(x_hat, bx).mean(dim=1)
            loss = loss_cls + 0.3 * loss_rec
            loss.mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Store per-sample loss (cls component dominates)
            epoch_losses[bidx.cpu().numpy()] = loss_cls.detach().cpu().numpy()

        loss_history[:, epoch] = epoch_losses

    return loss_history


def compute_reconstruction_anomaly(model, X_t, y_t, device=device):
    """
    Compute per-sample reconstruction error.
    Poisoned samples often have anomalous reconstruction because
    their latent representation is pulled in conflicting directions.
    """
    model = model.to(device).eval()
    rec_errors = np.zeros(len(y_t))
    with torch.no_grad():
        for i in range(0, len(y_t), 1024):
            bx = X_t[i:i+1024].to(device)
            _, x_hat = model(bx)
            err = ((bx - x_hat) ** 2).mean(dim=1).cpu().numpy()
            rec_errors[i:i+1024] = err
    return rec_errors


def hybrid_poison_filter(loss_history, rec_errors, y_labels,
                         loss_percentile=15, rec_percentile=15,
                         combine_method='union'):
    """
    Combine loss trajectory + reconstruction anomaly signals.

    Loss signal: Samples whose loss drops in the top X% fastest
                in the first K epochs are suspicious.

    Rec signal:  Samples with top X% highest reconstruction error
                are suspicious (latent representation conflict).

    combine_method: 'union' (more aggressive) or 'intersection' (safer)
    """
    n_samples = len(y_labels)

    # --- Signal 1: Fast loss decay (poisoned = easy to fit) ---
    # Measure how much loss dropped from epoch 5 to epoch 45
    # (skip first 5 epochs as warmup, use epoch 45 as stable early point)
    loss_drop = loss_history[:, 4] - loss_history[:, 44]
    loss_threshold = np.percentile(loss_drop, 100 - loss_percentile)
    loss_flag = loss_drop >= loss_threshold  # large drop = suspicious

    # --- Signal 2: High reconstruction error (representation conflict) ---
    rec_threshold = np.percentile(rec_errors, 100 - rec_percentile)
    rec_flag = rec_errors >= rec_threshold

    # --- Combine ---
    if combine_method == 'union':
        poison_mask = loss_flag | rec_flag
    else:
        poison_mask = loss_flag & rec_flag

    keep_mask = ~poison_mask

    print(f"\n{'='*60}")
    print("HYBRID POISON FILTER RESULTS")
    print(f"{'='*60}")
    print(f"  Loss-flagged samples    : {loss_flag.sum():,} ({loss_flag.mean():.1%})")
    print(f"  Rec-flagged samples     : {rec_flag.sum():,} ({rec_flag.mean():.1%})")
    print(f"  Combined flagged        : {poison_mask.sum():,} ({poison_mask.mean():.1%})")
    print(f"  Kept for retraining     : {keep_mask.sum():,} ({keep_mask.mean():.1%})")

    # Breakdown by true (poisoned) label
    for cls in [0, 1]:
        cls_name = "Normal" if cls == 0 else "Attack"
        in_cls = (y_labels == cls)
        flagged_in_cls = poison_mask[in_cls].sum()
        print(f"    Flagged with label={cls} ({cls_name}): {flagged_in_cls:,}")

    return keep_mask, poison_mask, loss_flag, rec_flag


# ============================================================
# EXECUTE THE DEFENSE
# ============================================================

print("=" * 60)
print("DEFENSE — Loss Trajectory + Reconstruction Anomaly")
print("Applied to TARGETED FLIP 10% attack")
print("=" * 60)

# --- Step 1: Train a scout model for 50 epochs to collect loss trajectories ---
print("\n[1/4] Training scout model for loss trajectory analysis …")
set_seed(42)
scout_model = AutoencoderMLP(input_dim).to(device)

loader_tgt10_full = make_loader(
    torch.FloatTensor(X_train_scaled).to(device),
    torch.LongTensor(y_train_tgt10).to(device),
    drop_last=False)

# We train the scout on the POISONED data — the key insight is that
# we don't trust its PREDICTIONS, we analyze its per-sample BEHAVIOR
loss_hist = loss_trajectory_analysis(
    scout_model,
    torch.FloatTensor(X_train_scaled).to(device),
    torch.LongTensor(y_train_tgt10).to(device),
    epochs=50, device=device)

# --- Step 2: Compute reconstruction anomalies on the trained scout ---
print("\n[2/4] Computing reconstruction anomaly scores …")
rec_errors = compute_reconstruction_anomaly(
    scout_model,
    torch.FloatTensor(X_train_scaled).to(device),
    torch.LongTensor(y_train_tgt10).to(device),
    device=device)

# --- Step 3: Hybrid filtering ---
print("\n[3/4] Running hybrid poison detection …")
keep_mask, poison_mask, loss_flag, rec_flag = hybrid_poison_filter(
    loss_hist, rec_errors, y_train_tgt10,
    loss_percentile=12,   # flag top 12% fastest loss decay
    rec_percentile=12,    # flag top 12% highest reconstruction error
    combine_method='union')

# --- Step 4: Retrain on cleaned data ---
X_def = X_train_scaled[keep_mask]
y_def = y_train_tgt10[keep_mask]

print(f"\n[4/4] Retraining on cleaned subset ({keep_mask.sum():,} samples) …")
loader_def = make_loader(
    torch.FloatTensor(X_def).to(device),
    torch.LongTensor(y_def).to(device),
    drop_last=True)

set_seed(42)
model_def = AutoencoderMLP(input_dim).to(device)
model_def, hist_def = train_ae_mlp(
    model_def, loader_def, val_loader,
    epochs=400, lr=3e-4, alpha_rec=0.3, patience=60, verbose_every=50)

res_def = evaluate(model_def, test_loader, device,
                   "Defense — Loss Trajectory + Reconstruction Anomaly\n(vs Targeted Flip 10%)")

ALL_RESULTS["Defense_Hybrid"] = res_def
ALL_HISTORIES["Defense_Hybrid"] = hist_def

# ============================================================
# VISUALIZATION: Defense Effectiveness
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Loss trajectories — poisoned vs clean samples
# (We know which ones were actually flipped in this synthetic attack)
true_flip_mask = np.zeros(len(y_train_tgt10), dtype=bool)
true_flip_mask[y_train == 1] = (y_train_tgt10[y_train == 1] == 0)  # attack→normal flips

clean_idx = np.where(~true_flip_mask)[0]
poison_idx = np.where(true_flip_mask)[0]

# Sample 200 from each for clarity
np.random.seed(42)
clean_sample = np.random.choice(clean_idx, min(200, len(clean_idx)), replace=False)
poison_sample = np.random.choice(poison_idx, min(200, len(poison_idx)), replace=False)

for i in clean_sample:
    axes[0].plot(loss_hist[i, :30], color='green', alpha=0.08, linewidth=0.5)
for i in poison_sample:
    axes[0].plot(loss_hist[i, :30], color='red', alpha=0.3, linewidth=0.8)

axes[0].plot([], [], color='green', alpha=0.8, linewidth=2, label='Clean samples')
axes[0].plot([], [], color='red', alpha=0.8, linewidth=2, label='Poisoned (Atk→Norm)')
axes[0].set(title='Per-Sample Loss Trajectories (First 30 Epochs)',
            xlabel='Epoch', ylabel='Cross-Entropy Loss')
axes[0].legend(); axes[0].grid(alpha=.3)

# Plot 2: Reconstruction error distribution
axes[1].hist(rec_errors[~true_flip_mask], bins=60, color='green', alpha=0.5,
             label='Clean', density=True)
axes[1].hist(rec_errors[true_flip_mask], bins=60, color='red', alpha=0.5,
             label='Poisoned', density=True)
axes[1].axvline(np.percentile(rec_errors, 88), color='black', linestyle='--',
                label='Filter threshold (88th pctile)')
axes[1].set(title='Reconstruction Error Distribution', xlabel='MSE Reconstruction Error',
            ylabel='Density')
axes[1].legend(); axes[1].grid(alpha=.3)

# Plot 3: Defense comparison bar chart
names = ['Baseline', 'TargetedFlip10%', 'Defense_Hybrid']
metrics_plot = ['f1', 'fnr', 'pr_auc']
metric_labels = ['F1-Score (↑better)', 'FNR (↓better)', 'PR-AUC (↑better)']

x = np.arange(len(names))
width = 0.25

for i, (m, lbl) in enumerate(zip(metrics_plot, metric_labels)):
    vals = [ALL_RESULTS[n][m] for n in names]
    bars = axes[2].bar(x + i*width, vals, width, label=lbl, alpha=0.8, edgecolor='k')
    for bar, v in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)

axes[2].set_xticks(x + width)
axes[2].set_xticklabels(['Baseline\n(clean)', 'Targeted Flip\n10% (no defense)', 'Hybrid Defense\n(retrained)'])
axes[2].set(title='Defense Effectiveness Comparison', ylabel='Score')
axes[2].legend(); axes[2].grid(alpha=.3, axis='y')
axes[2].set_ylim(0, 1.05)

plt.suptitle('Hybrid Defense: Loss Trajectory + Reconstruction Anomaly',
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
