import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
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
import warnings
import scipy.io.arff as arff
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score

# Argument parser
parser = argparse.ArgumentParser(description="Welcome to Table2Image")
parser.add_argument('--data', type=str, required=True, 
                   help='Path to the dataset (csv/arff/data)')
parser.add_argument('--save_dir', type=str, required=True, 
                   help='Path to save the final model')
args = parser.parse_args()

# Parameters
EPOCH = 50
BATCH_SIZE = 64

data_path = args.data
file_name = os.path.basename(data_path).rsplit('.', 1)[0]  # Remove any extension
saving_path = args.save_dir + '.pt'

print(f"\n{'='*70}")
print(f"TABLE2IMAGE - Starting Experiment")
print(f"{'='*70}")
print(f"Dataset: {file_name}")
print(f"{'='*70}\n")

# ========== DATA LOADING FUNCTION ==========
def load_dataset(file_path):
    """Auto-detect file format and load dataset"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        print(f"[INFO] Loading CSV file: {file_path}")
        return pd.read_csv(file_path), None
    
    elif file_ext == '.arff':
        print(f"[INFO] Loading ARFF file: {file_path}")
        try:
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            # Decode byte strings to regular strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            print(f"[INFO] ARFF attributes: {list(meta.names())[:10]}...")
            return df, meta  # Return metadata too
        except Exception as e:
            print(f"[WARNING] scipy.io.arff failed: {e}")
            # Fallback to alternative ARFF parser
            try:
                import arff as arff_lib
                with open(file_path, 'r') as f:
                    dataset = arff_lib.load(f)
                df = pd.DataFrame(dataset['data'], 
                                columns=[attr[0] for attr in dataset['attributes']])
                return df, None
            except Exception as e2:
                raise Exception(f"All ARFF parsers failed. Errors: (1) {e}, (2) {e2}")
    
    elif file_ext == '.data':
        print(f"[INFO] Loading .data file: {file_path}")
        # Try different delimiters
        for sep in [',', ' ', '\t', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None)
                if df.shape[1] > 1:
                    print(f"[INFO] Detected delimiter: '{sep}'")
                    return df, None
            except:
                continue
        raise Exception("Could not determine delimiter for .data file")
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported: .csv, .arff, .data")

# ========== LOAD DATASET ==========
print(f"[INFO] Loading dataset: {data_path}")

# Load dataset (with metadata if ARFF)
file_ext = os.path.splitext(data_path)[1].lower()
if file_ext == '.arff':
    df, arff_meta = load_dataset(data_path)
else:
    df, arff_meta = load_dataset(data_path)
    arff_meta = None

if df.empty:
    raise ValueError("Dataset is empty after loading")
if df.shape[1] < 2:
    raise ValueError(f"Dataset has only {df.shape[1]} column(s), need at least 2")

print(f"[INFO] Initial dataset shape: {df.shape}")
print(f"[INFO] Columns: {df.columns.tolist()[:10]}...")

# ========== HANDLE MISSING VALUES ==========
print(f"\n[INFO] Handling missing values...")

# Replace common missing value markers with NaN
missing_markers = ['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-']
df = df.replace(missing_markers, np.nan)

initial_missing = df.isnull().sum().sum()
print(f"[INFO] Initial missing values: {initial_missing}")

# Drop columns with too many missing values (>50% threshold)
missing_threshold = 0.5
missing_pct = df.isnull().sum() / len(df)
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

# ========== IMPROVED TARGET COLUMN DETECTION ==========
print(f"\n[INFO] Detecting target column...")

target_col = None

# Strategy 1: For ARFF files, use metadata to identify target
if arff_meta is not None:
    print("[INFO] Detecting target column from ARFF metadata...")
    try:
        attr_names = list(arff_meta.names())
        # ARFF convention: last attribute is typically the class/target
        target_col = attr_names[-1]
        print(f"[INFO] ARFF metadata indicates target: '{target_col}'")
        
        # Verify this column exists in dataframe
        if target_col not in df.columns:
            print(f"[WARNING] Metadata target '{target_col}' not found in dataframe. Falling back...")
            target_col = None
    except Exception as e:
        print(f"[WARNING] Could not read ARFF metadata: {e}")
        target_col = None

# Strategy 2: Search for known target column names
if target_col is None:
    target_col_candidates = [
        'target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target',
        'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology',
        'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival',
        'diagnosis', 'y', 'Author', 'Utility'
    ]
    target_col = next((col for col in df.columns if col in target_col_candidates), None)
    if target_col:
        print(f"[INFO] Found target column by name: '{target_col}'")

# Strategy 3: Use last column as fallback
if target_col is None:
    target_col = df.columns[-1]
    if all(isinstance(col, int) for col in df.columns):
        print(f"[INFO] Using last column (index {target_col}) as target.")
    else:
        print(f"[INFO] Using last column '{target_col}' as target.")

print(f"[INFO] Final target column: {target_col}")

# Don't drop the target column even if it has missing values
if target_col in cols_to_drop:
    cols_to_drop.remove(target_col)

if cols_to_drop:
    print(f"[INFO] Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing data")
    df = df.drop(columns=cols_to_drop)
    print(f"[INFO] Shape after dropping: {df.shape}")

if df.shape[1] <= 1:
    raise ValueError("All feature columns were dropped. Dataset unusable.")

# ========== EARLY CLASS DISTRIBUTION CHECK ==========
print(f"\n[INFO] Checking class distribution before preprocessing...")

if target_col in df.columns:
    # Show raw distribution
    target_value_counts = df[target_col].value_counts()
    print(f"[INFO] Raw class distribution:")
    for val, count in target_value_counts.items():
        print(f"  Class '{val}': {count} samples")
    
    # Check for classes with too few samples
    min_samples_per_class = 10
    rare_classes = target_value_counts[target_value_counts < min_samples_per_class]
    
    if len(rare_classes) > 0:
        print(f"\n[WARNING] Found {len(rare_classes)} class(es) with <{min_samples_per_class} samples:")
        for cls, count in rare_classes.items():
            print(f"  Class '{cls}': {count} samples")
        
        # Filter out rare classes
        valid_classes = target_value_counts[target_value_counts >= min_samples_per_class].index.tolist()
        
        if len(valid_classes) < 2:
            print(f"[ERROR] Only {len(valid_classes)} valid class(es) remain after filtering. Need at least 2.")
            print(f"[ERROR] Skipping dataset: insufficient samples per class.")
            exit(0)
        
        print(f"[INFO] Filtering dataset to keep only classes with >={min_samples_per_class} samples...")
        original_size = len(df)
        df = df[df[target_col].isin(valid_classes)]
        filtered_size = len(df)
        pct_removed = ((original_size - filtered_size) / original_size) * 100
        print(f"   ⚠️  WARNING: Removed {original_size - filtered_size} samples ({pct_removed:.1f}% of original data)")
        print(f"[INFO] New dataset shape: {df.shape}")
        
        # Show new distribution
        new_distribution = df[target_col].value_counts()
        print(f"[INFO] Filtered class distribution:")
        for val, count in new_distribution.items():
            print(f"  Class '{val}': {count} samples")
else:
    print(f"[ERROR] Target column '{target_col}' not found in dataframe!")
    exit(1)

# ========== ENCODE TARGET LABELS ==========
print(f"\n[INFO] Encoding target labels...")

# Convert target to integer labels
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

# ========== ENCODE FEATURES ==========
print(f"\n[INFO] Preparing features...")

X_df = df.drop(columns=[target_col])
print(f"[INFO] Encoding categorical features...")

# Encode categorical columns
for col in X_df.columns:
    if X_df[col].dtype == 'object':
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
    else:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

# ========== IMPUTE REMAINING MISSING VALUES ==========
print(f"[INFO] Imputing missing values with median...")
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X_df)

imputed_count = X_df.isnull().sum().sum()
if imputed_count > 0:
    print(f"[INFO] Imputed {imputed_count} missing values")

# ========== REMAP LABELS ==========
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

# ========== DEVICE SETUP ==========
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}")

# ========== LOAD IMAGE DATASETS ==========
print(f"\n[INFO] Loading FashionMNIST and MNIST datasets...")

# Load FashionMNIST
fashionmnist_dataset = datasets.FashionMNIST(
    root='.',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Load MNIST
mnist_dataset = datasets.MNIST(
    root='.',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Target + 10 (MNIST)
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

# ========== NORMALIZE AND SPLIT DATA ==========
print(f"\n[INFO] Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("[INFO] Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Create TensorDatasets
train_tabular_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), 
    torch.tensor(y_train, dtype=torch.long)
)
test_tabular_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), 
    torch.tensor(y_test, dtype=torch.long)
)

# Calculate number of samples needed for each label
train_tabular_label_counts = torch.bincount(train_tabular_dataset.tensors[1], minlength=num_classes)
test_tabular_label_counts = torch.bincount(test_tabular_dataset.tensors[1], minlength=num_classes)

num_samples_needed = train_tabular_label_counts.tolist()
num_samples_needed_test = test_tabular_label_counts.tolist()

valid_labels = {i for i in range(num_classes)}

# Filter FashionMNIST dataset
filtered_fashion = Subset(fashionmnist_dataset, 
    [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in valid_labels])

# Filter MNIST dataset and remap labels
filtered_mnist = Subset(modified_mnist_dataset, 
    [i for i, (_, label) in enumerate(modified_mnist_dataset) if label in valid_labels])

# Combine FashionMNIST and MNIST
combined_dataset = ConcatDataset([filtered_fashion, filtered_mnist])

# Integrity check
indices_by_label = {label: [] for label in range(num_classes)}

for i, (_, label) in enumerate(combined_dataset):
    if label not in indices_by_label:
        print(f"Unexpected label {label} at index {i}")
    indices_by_label[label].append(i)

# Generate repeated indices for balanced dataset
repeated_indices = {
    label: list(itertools.islice(itertools.cycle(indices_by_label[label]),
                                 num_samples_needed[label] + num_samples_needed_test[label]))
    for label in indices_by_label
}

# Align the train and test indices for both tabular and image datasets by label
aligned_train_indices = []
aligned_test_indices = []

for label in valid_labels:
    train_tab_indices = [i for i, lbl in enumerate(y_train) if lbl == label]
    test_tab_indices = [i for i, lbl in enumerate(y_test) if lbl == label]

    train_img_indices = repeated_indices[label][:num_samples_needed[label]]
    test_img_indices = repeated_indices[label][num_samples_needed[label]:num_samples_needed[label] + num_samples_needed_test[label]]

    if len(train_tab_indices) == len(train_img_indices) and len(test_tab_indices) == len(test_img_indices):
        aligned_train_indices.extend(list(zip(train_tab_indices, train_img_indices)))
        aligned_test_indices.extend(list(zip(test_tab_indices, test_img_indices)))
    else:
        raise ValueError(f"Mismatch in train/test counts for label {label}")

# Create final filtered subsets with aligned indices
train_filtered_tab_set = Subset(train_tabular_dataset, [idx[0] for idx in aligned_train_indices])
train_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_train_indices])

test_filtered_tab_set = Subset(test_tabular_dataset, [idx[0] for idx in aligned_test_indices])
test_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_test_indices])

# Define synchronized dataset class with consistency check
class SynchronizedDataset(Dataset):
    def __init__(self, tabular_dataset, image_dataset):
        self.tabular_dataset = tabular_dataset
        self.image_dataset = image_dataset
        assert len(self.tabular_dataset) == len(self.image_dataset), "Datasets must have the same length."

    def __len__(self):
        return len(self.tabular_dataset)

    def __getitem__(self, index):
        tab_data, tab_label = self.tabular_dataset[index]
        img_data, img_label = self.image_dataset[index]
        assert tab_label == img_label, f"Label mismatch: tab_label={tab_label}, img_label={img_label}"
        return tab_data, tab_label, img_data, img_label

# Create synchronized datasets
train_synchronized_dataset = SynchronizedDataset(train_filtered_tab_set, train_filtered_img_set)
test_synchronized_dataset = SynchronizedDataset(test_filtered_tab_set, test_filtered_img_set)

# Create data loaders
train_synchronized_loader = DataLoader(dataset=train_synchronized_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_synchronized_loader = DataLoader(dataset=test_synchronized_dataset, batch_size=BATCH_SIZE)

print(f"[INFO] Synchronized datasets created. Train batches: {len(train_synchronized_loader)}")

# ========== MODEL DEFINITIONS ==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
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
    def __init__(self, tab_latent_size=tab_latent_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_cont_features, tab_latent_size)
        self.fc2 = nn.Linear(tab_latent_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, torch.sigmoid(x)


model_with_embeddings = SimpleMLP(tab_latent_size)


# VIF Embedding
def calculate_vif(df):
    df = pd.DataFrame(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data


class VIFInitialization(nn.Module):
    def __init__(self, input_dim, vif_values):
        super(VIFInitialization, self).__init__()
        
        # VIF-based init
        self.input_dim = input_dim
        self.vif_values = vif_values
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # fc1 weight init
        with torch.no_grad():
            vif_tensor = torch.tensor(self.vif_values, dtype=torch.float32)
            inv_vif = 1 / vif_tensor  # reciprocal of vif values
            for i in range(self.input_dim):
                self.fc1.weight.data[i, :] = inv_vif[i] / (self.input_dim + 4)
            
            # fc2 weight init (default xavier init)
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, tab_latent_size=8, latent_size=8):
        super(CVAEWithTabEmbedding, self).__init__()
        
        self.mlp = model_with_embeddings
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size + n_cont_features, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size + n_cont_features, 128),
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
        vif_df = calculate_vif(tab_data.detach().cpu().numpy())
        vif_values = vif_df['VIF'].values
        input_dim = tab_data.shape[1]
        vif_model = VIFInitialization(input_dim, vif_values).to(DEVICE)
        vif_embedding = vif_model(tab_data)
        
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding, vif_embedding)
        recon_x = self.decode(z, tab_embedding, vif_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))
        return recon_x, tab_pred, img_pred


print("[INFO] Creating model...")
cvae = CVAEWithTabEmbedding(tab_latent_size).to(DEVICE)
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
        img_label = img_label.to(DEVICE)
        tab_label = tab_label.to(DEVICE)
        
        optimizer.zero_grad()
        
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(DEVICE)
        
        recon_x, tab_pred, img_pred = model(x_rand, tab_data)
        tab_pred = tab_pred.squeeze(-1)
        img_pred = img_pred.squeeze(-1)

        loss = loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


def test(model, test_data_loader, epoch, best_accuracy, best_auc, best_epoch, best_model_path='best_model.pth'):
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_img_total = 0
    total = 0
    correct_tab = {i: 0 for i in range(num_classes)}
    total_tab = {i: 0 for i in range(num_classes)}
    correct_img = {i: 0 for i in range(num_classes)}
    total_img = {i: 0 for i in range(num_classes)}
    
    all_tab_labels = []
    all_tab_preds = []
    all_img_labels = []
    all_img_preds = []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            img_label = img_label.to(DEVICE)
            tab_label = tab_label.to(DEVICE)
            
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)
            
            tab_pred = tab_pred.squeeze(-1)
            img_pred = img_pred.squeeze(-1)
            
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_pred.cpu().numpy())
            all_img_labels.extend(img_label.cpu().numpy())
            all_img_preds.extend(img_pred.cpu().numpy())

            if tab_pred.dim() == 1:
                tab_predicted = (tab_pred > 0.5).long()
            else:
                tab_predicted = torch.argmax(tab_pred, dim=1)
            
            for i in range(len(tab_label)):
                label = tab_label[i].item()
                correct_tab[label] += (tab_predicted[i] == label).item()
                total_tab[label] += 1

            if img_pred.dim() == 1:
                img_predicted = (img_pred > 0.5).long()
            else:
                img_predicted = torch.argmax(img_pred, dim=1)
            
            for i in range(len(img_label)):
                label = img_label[i].item()
                correct_img[label] += (img_predicted[i] == label).item()
                total_img[label] += 1
                
            correct_tab_total += (tab_predicted == tab_label).sum().item()
            correct_img_total += (img_predicted == img_label).sum().item()

            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy_total = 100 * correct_tab_total / total
    img_accuracy_total = 100 * correct_img_total / total
    tab_accuracy = {cls: (correct_tab[cls] / total_tab[cls]) * 100 if total_tab[cls] > 0 else 0 for cls in range(num_classes)}
    img_accuracy = {cls: (correct_img[cls] / total_img[cls]) * 100 if total_img[cls] > 0 else 0 for cls in range(num_classes)}

    # Calculate AUC
    try:
        tab_auc = roc_auc_score(all_tab_labels, all_tab_preds, multi_class="ovr", average="macro")
        img_auc = roc_auc_score(all_img_labels, all_img_preds, multi_class="ovr", average="macro")
    except:
        tab_auc = 0.0
        img_auc = 0.0

    if img_accuracy_total > best_accuracy:
        best_accuracy = img_accuracy_total
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"[INFO] New best accuracy: {best_accuracy:.2f}% at epoch {epoch}")
        
    if img_auc > best_auc:
        best_auc = img_auc

    return best_accuracy, best_auc, best_epoch


# ========== TRAINING LOOP ==========
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

best_accuracy = 0
best_auc = 0
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train(cvae, train_synchronized_loader, optimizer, epoch)
    best_accuracy, best_auc, best_epoch = test(cvae, test_synchronized_loader, epoch, best_accuracy, best_auc, best_epoch, best_model_path=saving_path)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"[Epoch {epoch:3d}] Best Acc: {best_accuracy:.2f}% | Best AUC: {best_auc:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print(f"Best model image classification accuracy: {best_accuracy:.4f} at epoch: {best_epoch}, Best AUC: {best_auc:.4f}")
print("="*70 + "\n")