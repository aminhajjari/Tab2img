import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import os
import json
import sys
from datetime import datetime
import scipy.io.arff as arff

# Machine Learning Models
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ========== ARGUMENT PARSER ==========
parser = argparse.ArgumentParser(description="Baseline Models Comparison for Table2Image")
parser.add_argument('--data', type=str, 
                   help='Path to the dataset (csv/arff/data)')
parser.add_argument('--data_dir', type=str,
                   help='Directory containing multiple datasets to process')
parser.add_argument('--output_dir', type=str, default='baseline_results',
                   help='Directory to save comparison results')
parser.add_argument('--skip_tuning', action='store_true',
                   help='Skip hyperparameter tuning (use defaults)')
parser.add_argument('--random_state', type=int, default=42,
                   help='Random state for reproducibility')
args = parser.parse_args()

# Check that at least one data source is provided
if not args.data and not args.data_dir:
    parser.error("Either --data or --data_dir must be provided")
    sys.exit(1)

# ========== PARAMETERS ==========
RANDOM_STATE = args.random_state
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print(f"\n{'='*70}")
print(f"BASELINE MODELS COMPARISON FOR TABLE2IMAGE")
print(f"{'='*70}")
print(f"Device: {DEVICE}")
print(f"Random State: {RANDOM_STATE}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"{'='*70}\n")

# ========== DATA LOADING ==========
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
            return df, meta
        except Exception as e:
            print(f"[WARNING] scipy.io.arff failed: {e}")
            import arff as arff_lib
            with open(file_path, 'r') as f:
                dataset = arff_lib.load(f)
            df = pd.DataFrame(dataset['data'], 
                            columns=[attr[0] for attr in dataset['attributes']])
            return df, None
    
    elif file_ext == '.data':
        print(f"[INFO] Loading .data file: {file_path}")
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
        raise ValueError(f"Unsupported file format: {file_ext}")

# ========== PREPROCESSING ==========
def preprocess_data(data_path):
    """Preprocess dataset similar to Table2Image preprocessing"""
    
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    file_ext = os.path.splitext(data_path)[1].lower()
    
    print(f"\n[INFO] Processing dataset: {file_name}")
    
    # Load dataset
    if file_ext == '.arff':
        df, arff_meta = load_dataset(data_path)
    else:
        df = load_dataset(data_path)
        arff_meta = None
    
    print(f"[INFO] Initial dataset shape: {df.shape}")
    
    # Handle missing values
    missing_markers = ['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-']
    df = df.replace(missing_markers, np.nan)
    
    # Detect target column
    target_col = None
    if arff_meta is not None:
        try:
            attr_names = list(arff_meta.names())
            target_col = attr_names[-1]
        except:
            pass
    
    if target_col is None:
        target_col_candidates = [
            'target', 'class', 'outcome', 'Class', 'binaryClass', 
            'status', 'Target', 'label', 'Label', 'y'
        ]
        target_col = next((col for col in df.columns if col in target_col_candidates), None)
    
    if target_col is None:
        target_col = df.columns[-1]
    
    print(f"[INFO] Target column: {target_col}")
    
    # Check class distribution and filter rare classes
    min_samples_per_class = 10
    target_value_counts = df[target_col].value_counts()
    rare_classes = target_value_counts[target_value_counts < min_samples_per_class]
    
    if len(rare_classes) > 0:
        print(f"[INFO] Filtering out {len(rare_classes)} rare classes...")
        valid_classes = target_value_counts[target_value_counts >= min_samples_per_class].index.tolist()
        df = df[df[target_col].isin(valid_classes)]
    
    # Drop columns with >50% missing data
    missing_threshold = 0.5
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Encode target
    if df[target_col].dtype == 'object' or not np.issubdtype(df[target_col].dtype, np.number):
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target_col].astype(str))
    else:
        y = df[target_col].astype(int).values
        unique_values = sorted(set(y))
        value_map = {unique_values[i]: i for i in range(len(unique_values))}
        y = np.array([value_map[val] for val in y])
    
    num_classes = len(np.unique(y))
    print(f"[INFO] Number of classes: {num_classes}")
    
    if num_classes > 20:
        raise ValueError(f"Dataset has {num_classes} classes (>20). Skipping...")
    if num_classes < 2:
        raise ValueError(f"Dataset has only {num_classes} class. Need at least 2.")
    
    # Prepare features
    X_df = df.drop(columns=[target_col])
    
    # Encode categorical features
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_df)
    
    print(f"\n[INFO] Preprocessed Data:")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, num_classes, file_name

# ========== PYTORCH MLP MODEL ==========
class SimpleMLP(nn.Module):
    """Simple MLP for fair comparison with Table2Image"""
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64]):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_pytorch_mlp(X_train, y_train, X_test, y_test, num_classes, 
                     epochs=100, batch_size=64, lr=0.001):
    """Train PyTorch MLP"""
    print("\n[INFO] Training PyTorch MLP...")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SimpleMLP(X_train.shape[1], num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            
            acc = 100 * correct / total
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {train_loss/len(train_loader):.4f} | "
                      f"Acc: {acc:.2f}%")
    
    # Final evaluation
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except:
        auc = 0.0
    
    print(f"  Best Accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"  Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"  Final Test AUC: {auc:.4f}")
    
    return {
        'model': 'PyTorch_MLP',
        'accuracy': accuracy * 100,
        'auc': auc,
        'best_epoch': best_epoch,
        'predictions': all_preds,
        'probabilities': all_probs
    }

# ========== XGBOOST ==========
def train_xgboost(X_train, y_train, X_test, y_test, num_classes, skip_tuning=False):
    """Train XGBoost with optional hyperparameter tuning"""
    print("\n[INFO] Training XGBoost...")
    
    if skip_tuning:
        print("  Using default parameters...")
        params = {
            'max_depth': 6,
            'eta': 0.3,
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
            'seed': RANDOM_STATE
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(params, dtrain, num_boost_round=100)
    else:
        print("  Performing hyperparameter tuning...")
        param_grid = {
            'max_depth': [6, 10, 20],
            'eta': [0.1, 0.3, 0.5],
            'n_estimators': [100, 500]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss' if num_classes > 2 else 'logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        print(f"  Best parameters: {grid_search.best_params_}")
    
    # Predictions
    if skip_tuning:
        y_pred_proba = model.predict(dtest)
        if num_classes == 2:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
            y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate AUC
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
    except:
        auc = 0.0
    
    print(f"  Test Accuracy: {accuracy*100:.2f}%")
    print(f"  Test AUC: {auc:.4f}")
    
    return {
        'model': 'XGBoost',
        'accuracy': accuracy * 100,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# ========== LIGHTGBM ==========
def train_lightgbm(X_train, y_train, X_test, y_test, num_classes, skip_tuning=False):
    """Train LightGBM with optional hyperparameter tuning"""
    print("\n[INFO] Training LightGBM...")
    
    if skip_tuning:
        print("  Using default parameters...")
        params = {
            'num_leaves': 31,
            'max_depth': 6,
            'n_estimators': 100,
            'objective': 'multiclass' if num_classes > 2 else 'binary',
            'num_class': num_classes if num_classes > 2 else None,
            'random_state': RANDOM_STATE,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
    else:
        print("  Performing hyperparameter tuning...")
        param_grid = {
            'num_leaves': [31, 60, 100],
            'max_depth': [6, 10, 20],
            'n_estimators': [100, 500]
        }
        
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass' if num_classes > 2 else 'binary',
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        print(f"  Best parameters: {grid_search.best_params_}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate AUC
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
    except:
        auc = 0.0
    
    print(f"  Test Accuracy: {accuracy*100:.2f}%")
    print(f"  Test AUC: {auc:.4f}")
    
    return {
        'model': 'LightGBM',
        'accuracy': accuracy * 100,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# ========== SINGLE DATASET PROCESSING ==========
def process_single_dataset(data_path):
    """Process a single dataset"""
    try:
        # Load and preprocess data
        X, y, num_classes, dataset_name = preprocess_data(data_path)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n{'='*70}")
        print("TRAINING BASELINE MODELS")
        print(f"{'='*70}")
        
        # Train all models
        results = {}
        
        # 1. XGBoost
        results['xgboost'] = train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test, 
            num_classes, args.skip_tuning
        )
        
        # 2. LightGBM
        results['lightgbm'] = train_lightgbm(
            X_train_scaled, y_train, X_test_scaled, y_test, 
            num_classes, args.skip_tuning
        )
        
        # 3. PyTorch MLP
        results['pytorch_mlp'] = train_pytorch_mlp(
            X_train_scaled, y_train, X_test_scaled, y_test, 
            num_classes
        )
        
        # ========== CREATE COMPARISON TABLE ==========
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        
        comparison_df = pd.DataFrame({
            'Model': [r['model'] for r in results.values()],
            'Accuracy (%)': [r['accuracy'] for r in results.values()],
            'AUC': [r['auc'] for r in results.values()]
        })
        
        print(comparison_df.to_string(index=False))
        
        # Save results
        dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(dataset_dir, 'baseline_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Results saved to: {csv_path}")
        
        # Save detailed JSON
        detailed_results = {
            'dataset': dataset_name,
            'num_samples': len(X),
            'num_features': X.shape[1],
            'num_classes': num_classes,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'models': {k: {
                'model': v['model'],
                'accuracy': float(v['accuracy']),
                'auc': float(v['auc'])
            } for k, v in results.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = os.path.join(dataset_dir, 'baseline_results.json')
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # ========== CREATE VISUALIZATION ==========
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        axes[0].bar(comparison_df['Model'], comparison_df['Accuracy (%)'], 
                    color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 105])
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(comparison_df['Accuracy (%)']):
            axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # AUC comparison
        axes[1].bar(comparison_df['Model'], comparison_df['AUC'], 
                    color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[1].set_ylabel('AUC', fontsize=12, fontweight='bold')
        axes[1].set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(comparison_df['AUC']):
            axes[1].text(i, v + 0.03, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.suptitle(f'Baseline Models Comparison - {dataset_name}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(dataset_dir, 'baseline_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Visualization saved to: {plot_path}")
        
        # Print JSON for batch processing
        print("\n" + "="*70)
        print("RESULTS_JSON_START")
        print(json.dumps(detailed_results))
        print("RESULTS_JSON_END")
        print("="*70 + "\n")
        
        print(f"✅ Baseline comparison completed successfully for {dataset_name}!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ========== MAIN ==========
def main():
    """Main entry point"""
    success_count = 0
    fail_count = 0
    
    if args.data:
        # Process single dataset
        if process_single_dataset(args.data):
            success_count += 1
        else:
            fail_count += 1
    
    elif args.data_dir:
        # Process multiple datasets
        print(f"\n[INFO] Processing all datasets in: {args.data_dir}")
        
        # Find all dataset files
        dataset_files = []
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith(('.csv', '.arff', '.data')):
                    dataset_files.append(os.path.join(root, file))
        
        print(f"[INFO] Found {len(dataset_files)} datasets to process")
        
        for i, data_path in enumerate(dataset_files, 1):
            print(f"\n{'#'*70}")
            print(f"PROCESSING DATASET {i}/{len(dataset_files)}: {os.path.basename(data_path)}")
            print(f"{'#'*70}")
            
            if process_single_dataset(data_path):
                success_count += 1
            else:
                fail_count += 1
    
    # Final summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"Total: {success_count + fail_count}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()