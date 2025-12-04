import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset, random_split
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, transforms
import itertools
import argparse
import os
import json
from datetime import datetime

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import scipy.io.arff as arff

# ========== ARGUMENT PARSER ==========
parser = argparse.ArgumentParser(description="Welcome to Table2Image")
parser.add_argument('--data', type=str, required=True, 
                   help='Path to the dataset (csv/arff/data)')
parser.add_argument('--save_dir', type=str, required=True, 
                   help='Directory to save results (no model checkpoints will be saved)')
args = parser.parse_args()

# ========== PARAMETERS ==========
EPOCH = 50
BATCH_SIZE = 64

data_path = args.data
file_name = os.path.basename(os.path.dirname(data_path))
DATASET_ROOT = '/project/def-arashmoh/shahab33/Msc/datasets'

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print(f"\n{'='*70}")
print(f"TABLE2IMAGE - Starting Experiment (No Checkpoint Saving)")
print(f"{'='*70}")
print(f"Dataset: {file_name}")
print(f"Device: {DEVICE}")
print(f"Note: Model checkpoints will NOT be saved to disk")
print(f"{'='*70}\n")

# ========== DATA LOADING FUNCTION ==========
def load_dataset(file_path):
    """Auto-detect file format and load dataset"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        print(f"[INFO] Loading CSV file: {file_path}")
        return pd.read_csv(file_path)
    
    elif file_ext == '.arff':
        print(f"[INFO] Loading ARFF file: {file_path}")
        try:
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            print(f"[INFO] ARFF attributes: {list(meta.names())[:10]}...")
            return df
        except Exception as e:
            print(f"[WARNING] scipy.io.arff failed: {e}")
            try:
                import arff as arff_lib
                with open(file_path, 'r') as f:
                    dataset = arff_lib.load(f)
                df = pd.DataFrame(dataset['data'], 
                                columns=[attr[0] for attr in dataset['attributes']])
                return df
            except Exception as e2:
                raise Exception(f"All ARFF parsers failed. Errors: (1) {e}, (2) {e2}")
    
    elif file_ext == '.data':
        print(f"[INFO] Loading .data file: {file_path}")
        for sep in [',', ' ', '\t', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None)
                if df.shape[1] > 1:
                    print(f"[INFO] Detected delimiter: '{sep}'")
                    return df
            except:
                continue
        raise Exception("Could not determine delimiter for .data file")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


# [SAME PREPROCESSING CODE AS BEFORE - KEEPING FOR BREVITY]
print(f"[INFO] Loading dataset: {data_path}")
df = load_dataset(data_path)

if df.empty:
    raise ValueError("Dataset is empty after loading")
if df.shape[1] < 2:
    raise ValueError(f"Dataset has only {df.shape[1]} column(s), need at least 2")

print(f"[INFO] Initial dataset shape: {df.shape}")
print(f"[INFO] Columns: {df.columns.tolist()[:10]}...")

missing_markers = ['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-']
df = df.replace(missing_markers, np.nan)
initial_missing = df.isnull().sum().sum()
print(f"[INFO] Initial missing values: {initial_missing}")

target_col_candidates = [
    'target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target',
    'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology',
    'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival',
    'diagnosis', 'y'
]

target_col = next((col for col in df.columns if col in target_col_candidates), None)
if target_col is None:
    if all(isinstance(col, int) for col in df.columns):
        target_col = df.columns[-1]
        print(f"[INFO] Using last column (index {target_col}) as target.")
    else:
        target_col = df.columns[-1]
        print(f"[INFO] Using '{target_col}' as target.")
print(f"[INFO] Target column: {target_col}")

missing_threshold = 0.5
missing_pct = df.isnull().sum() / len(df)
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
if target_col in cols_to_drop:
    cols_to_drop.remove(target_col)
if cols_to_drop:
    print(f"[INFO] Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing data")
    df = df.drop(columns=cols_to_drop)
    print(f"[INFO] Shape after dropping: {df.shape}")

if df.shape[1] <= 1:
    raise ValueError("All feature columns were dropped. Dataset unusable.")

if df[target_col].dtype == 'object' or not np.issubdtype(df[target_col].dtype, np.number):
    print(f"[INFO] Converting labels to integers...")
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))
    unique_values = le_target.classes_.tolist()
else:
    y = df[target_col].astype(int).values
    unique_values = sorted(set(y))

num_classes = len(unique_values)
print(f"[INFO] Detected {num_classes} unique classes: {unique_values}")

if num_classes > 20:
    print(f"[ERROR] Dataset has {num_classes} classes (>20). Skipping...")
    exit(1)
if num_classes < 2:
    raise ValueError(f"Dataset has only {num_classes} class. Need at least 2.")

X_df = df.drop(columns=[target_col])
print(f"[INFO] Encoding categorical features...")
for col in X_df.columns:
    if X_df[col].dtype == 'object':
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
    else:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

print(f"[INFO] Imputing missing values with median...")
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X_df)
imputed_count = X_df.isnull().sum().sum()
if imputed_count > 0:
    print(f"[INFO] Imputed {imputed_count} missing values")

unique_values = sorted(set(y))
num_classes = len(unique_values)
value_map = {unique_values[i]: i for i in range(num_classes)}
y = np.array([value_map[val] for val in y])

n_cont_features = X.shape[1]
tab_latent_size = n_cont_features + 4

print(f"\n{'='*70}")
print(f"[SUMMARY] Preprocessed Data:")
print(f"  - Samples: {X.shape[0]}")
print(f"  - Features: {X.shape[1]}")
print(f"  - Classes: {num_classes}")
print(f"  - Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"  - Tab latent size: {tab_latent_size}")
print(f"{'='*70}\n")

print("[INFO] Loading FashionMNIST and MNIST datasets...")
fashionmnist_dataset = datasets.FashionMNIST(
    root=DATASET_ROOT, train=True, download=False, transform=transforms.ToTensor()
)
mnist_dataset = datasets.MNIST(
    root=DATASET_ROOT, train=True, download=False, transform=transforms.ToTensor()
)

class ModifiedLabelDataset(Dataset):
    def __init__(self, dataset, label_offset=10):
        self.dataset = dataset
        self.label_offset = label_offset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.label_offset

modified_mnist_dataset = ModifiedLabelDataset(mnist_dataset, label_offset=10)

print("[INFO] Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("[INFO] Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

train_tabular_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), 
    torch.tensor(y_train, dtype=torch.long)
)
test_tabular_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), 
    torch.tensor(y_test, dtype=torch.long)
)

print("[INFO] Calculating VIF values...")
def calculate_vif_safe(X_data):
    df_vif = pd.DataFrame(X_data)
    n_features = df_vif.shape[1]
    vif_values = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for i in range(n_features):
            try:
                vif = variance_inflation_factor(df_vif.values, i)
                if np.isnan(vif) or np.isinf(vif):
                    vif = 1.0
            except:
                vif = 1.0
            vif_values.append(vif)
    vif_values = np.array(vif_values)
    vif_values = np.clip(vif_values, 1.0, 100.0)
    return vif_values

X_sample = X_train[:min(1000, len(X_train))]
vif_values = calculate_vif_safe(X_sample)
print(f"[INFO] VIF calculated. Mean: {vif_values.mean():.2f}, Max: {vif_values.max():.2f}")

print("[INFO] Preparing synchronized image-tabular datasets...")
train_tabular_label_counts = torch.bincount(train_tabular_dataset.tensors[1], minlength=num_classes)
test_tabular_label_counts = torch.bincount(test_tabular_dataset.tensors[1], minlength=num_classes)
num_samples_needed = train_tabular_label_counts.tolist()
num_samples_needed_test = test_tabular_label_counts.tolist()
valid_labels = set(range(num_classes))

filtered_fashion = Subset(fashionmnist_dataset, 
    [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in valid_labels])
filtered_mnist = Subset(modified_mnist_dataset, 
    [i for i, (_, label) in enumerate(modified_mnist_dataset) if label in valid_labels])
combined_dataset = ConcatDataset([filtered_fashion, filtered_mnist])

indices_by_label = {label: [] for label in range(num_classes)}
for i, (_, label) in enumerate(combined_dataset):
    if label not in indices_by_label:
        print(f"[WARNING] Unexpected label {label} at index {i}")
    indices_by_label[label].append(i)

repeated_indices = {
    label: list(itertools.islice(
        itertools.cycle(indices_by_label[label]),
        num_samples_needed[label] + num_samples_needed_test[label]
    ))
    for label in indices_by_label
}

aligned_train_indices = []
aligned_test_indices = []
for label in valid_labels:
    train_tab_indices = [i for i, lbl in enumerate(y_train) if lbl == label]
    test_tab_indices = [i for i, lbl in enumerate(y_test) if lbl == label]
    train_img_indices = repeated_indices[label][:num_samples_needed[label]]
    test_img_indices = repeated_indices[label][
        num_samples_needed[label]:num_samples_needed[label] + num_samples_needed_test[label]
    ]
    if len(train_tab_indices) == len(train_img_indices) and \
       len(test_tab_indices) == len(test_img_indices):
        aligned_train_indices.extend(list(zip(train_tab_indices, train_img_indices)))
        aligned_test_indices.extend(list(zip(test_tab_indices, test_img_indices)))
    else:
        raise ValueError(f"Mismatch for label {label}")

train_filtered_tab_set = Subset(train_tabular_dataset, [idx[0] for idx in aligned_train_indices])
train_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_train_indices])
test_filtered_tab_set = Subset(test_tabular_dataset, [idx[0] for idx in aligned_test_indices])
test_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_test_indices])

class SynchronizedDataset(Dataset):
    def __init__(self, tabular_dataset, image_dataset):
        self.tabular_dataset = tabular_dataset
        self.image_dataset = image_dataset
        assert len(self.tabular_dataset) == len(self.image_dataset)
    def __len__(self):
        return len(self.tabular_dataset)
    def __getitem__(self, index):
        tab_data, tab_label = self.tabular_dataset[index]
        img_data, img_label = self.image_dataset[index]
        assert tab_label == img_label
        return tab_data, tab_label, img_data, img_label

train_synchronized_dataset = SynchronizedDataset(train_filtered_tab_set, train_filtered_img_set)
test_synchronized_dataset = SynchronizedDataset(test_filtered_tab_set, test_filtered_img_set)
train_synchronized_loader = DataLoader(train_synchronized_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_synchronized_loader = DataLoader(test_synchronized_dataset, batch_size=BATCH_SIZE)
print(f"[INFO] Synchronized datasets created. Train batches: {len(train_synchronized_loader)}")

# ========== MODEL DEFINITIONS ==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, x

class VIFInitialization(nn.Module):
    def __init__(self, input_dim, vif_values):
        super(VIFInitialization, self).__init__()
        self.input_dim = input_dim
        self.vif_values = vif_values
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)
        vif_tensor = torch.tensor(vif_values, dtype=torch.float32)
        vif_tensor = vif_tensor / (vif_tensor.mean() + 1e-6)
        inv_vif = 1.0 / torch.clamp(vif_tensor, min=1.0)
        with torch.no_grad():
            for i in range(self.fc1.weight.data.shape[0]):
                self.fc1.weight.data[i, :] = inv_vif[i % len(inv_vif)] / (self.input_dim + 4)
        print("[INFO] VIF-based weight initialization complete.")
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, input_dim, tab_latent_size, num_classes, latent_size=8, vif_values=None):
        super(CVAEWithTabEmbedding, self).__init__()
        self.mlp = SimpleMLP(input_dim, tab_latent_size, num_classes)
        if vif_values is not None:
            self.vif_model = VIFInitialization(input_dim, vif_values)
        else:
            self.vif_model = None
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size + input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size + input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        self.final_classifier = SimpleCNN(num_classes=num_classes)
    def encode(self, x, tab_embedding, vif_embedding):
        return self.encoder(torch.cat([x, tab_embedding, vif_embedding], dim=1))
    def decode(self, z, tab_embedding, vif_embedding):
        return self.decoder(torch.cat([z, tab_embedding, vif_embedding], dim=1))
    def forward(self, x, tab_data):
        if self.vif_model is not None:
            vif_embedding = self.vif_model(tab_data)
        else:
            vif_embedding = tab_data
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding, vif_embedding)
        recon_x = self.decode(z, tab_embedding, vif_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))
        return recon_x, tab_pred, img_pred

print("[INFO] Creating model...")
cvae = CVAEWithTabEmbedding(
    input_dim=n_cont_features,
    tab_latent_size=tab_latent_size,
    num_classes=num_classes,
    latent_size=8,
    vif_values=vif_values
).to(DEVICE)
optimizer = optim.AdamW(cvae.parameters(), lr=0.001)
print(f"[INFO] Model created with {sum(p.numel() for p in cvae.parameters())} parameters")

def loss_function(recon_x, x, tab_pred, tab_labels, img_pred, img_labels):
    BCE = F.mse_loss(recon_x, x)
    tab_loss = F.cross_entropy(tab_pred, tab_labels)
    img_loss = F.cross_entropy(img_pred, img_labels)
    return BCE + tab_loss + img_loss

def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(DEVICE)
        tab_data = tab_data.to(DEVICE)
        img_label = img_label.to(DEVICE).long()
        tab_label = tab_label.to(DEVICE).long()
        optimizer.zero_grad()
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(DEVICE)
        recon_x, tab_pred, img_pred = model(x_rand, tab_data)
        loss = loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_data_loader)

# ========== TESTING FUNCTION (NO CHECKPOINT SAVING) ==========
def test(model, test_data_loader, epoch, best_accuracy, best_auc, best_epoch):
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_img_total = 0
    total = 0
    all_tab_labels, all_tab_preds = [], []
    all_img_labels, all_img_preds = [], []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            img_label = img_label.to(DEVICE).long()
            tab_label = tab_label.to(DEVICE).long()
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            tab_probs = F.softmax(tab_pred, dim=1)
            img_probs = F.softmax(img_pred, dim=1)
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_probs.cpu().numpy())
            all_img_labels.extend(img_label.cpu().numpy())
            all_img_preds.extend(img_probs.cpu().numpy())
            tab_predicted = torch.argmax(tab_pred, dim=1)
            img_predicted = torch.argmax(img_pred, dim=1)
            correct_tab_total += (tab_predicted == tab_label).sum().item()
            correct_img_total += (img_predicted == img_label).sum().item()
            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy_total = 100 * correct_tab_total / total
    img_accuracy_total = 100 * correct_img_total / total
    
    all_tab_preds_arr = np.array(all_tab_preds)
    all_img_preds_arr = np.array(all_img_preds)
    all_tab_labels_arr = np.array(all_tab_labels)
    all_img_labels_arr = np.array(all_img_labels)

    tab_auc, img_auc = 0.0, 0.0
    if not (np.isnan(all_tab_preds_arr).any() or np.isinf(all_tab_preds_arr).any()):
        try:
            if num_classes == 2:
                tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr[:, 1])
            else:
                tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[WARNING] Tab AUC calculation failed: {e}")
    
    if not (np.isnan(all_img_preds_arr).any() or np.isinf(all_img_preds_arr).any()):
        try:
            if num_classes == 2:
                img_auc = roc_auc_score(all_img_labels_arr, all_img_preds_arr[:, 1])
            else:
                img_auc = roc_auc_score(all_img_labels_arr, all_img_preds_arr, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[WARNING] Img AUC calculation failed: {e}")

    # Update best metrics (NO CHECKPOINT SAVING)
    if img_accuracy_total > best_accuracy:
        best_accuracy = img_accuracy_total
        best_epoch = epoch
        print(f"[INFO] New best accuracy: {best_accuracy:.2f}% at epoch {epoch} (not saved to disk)")
    if img_auc > best_auc:
        best_auc = img_auc

    return best_accuracy, best_auc, best_epoch, test_loss, tab_accuracy_total, img_accuracy_total

# ========== TRAINING LOOP ==========
print(f"\n{'='*70}")
print(f"Starting Training for {EPOCH} epochs")
print(f"{'='*70}\n")

best_accuracy = 0
best_auc = 0
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train_loss = train(cvae, train_synchronized_loader, optimizer, epoch)
    best_accuracy, best_auc, best_epoch, test_loss, tab_acc, img_acc = test(
        cvae, test_synchronized_loader, epoch, best_accuracy, best_auc, best_epoch
    )
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCH} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Tab Acc: {tab_acc:.2f}% | Img Acc: {img_acc:.2f}%")

# ========== FINAL RESULTS ==========
print(f"\n{'='*70}")
print(f"Training Complete!")
print(f"{'='*70}")
print(f"Best Image Classification Accuracy: {best_accuracy:.4f}% at epoch {best_epoch}")
print(f"Best AUC: {best_auc:.4f}")
print(f"Note: Model checkpoints were NOT saved to conserve disk space")
print(f"{'='*70}\n")

# ========== SAVE RESULTS TO JSON (ONLY LOGS, NO MODEL) ==========
class_dist = dict(zip(*np.unique(y, return_counts=True)))
class_dist_json = {int(k): int(v) for k, v in class_dist.items()}

results = {
    'dataset': file_name,
    'timestamp': datetime.now().isoformat(),
    'num_samples': int(len(y)),
    'num_features': int(n_cont_features),
    'num_classes': int(num_classes),
    'class_distribution': class_dist_json,
    'best_accuracy': float(best_accuracy),
    'best_auc': float(best_auc),
    'best_epoch': int(best_epoch),
    'total_epochs': int(EPOCH),
    'device': str(DEVICE)
}

# Save to logs directory
model_dir = os.path.dirname(args.save_dir)
run_dir = os.path.dirname(model_dir)
logs_dir = os.path.join(run_dir, 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

results_file = os.path.join(logs_dir, 'results.jsonl')
with open(results_file, 'a') as f:
    f.write(json.dumps(results) + '\n')

print(f"[INFO] Results appended to: {results_file}")
print(f"[INFO] No model checkpoints were saved to disk")
