import os
# Force CPU usage - must be set before importing torch/synthcity
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
from pathlib import Path
import warnings
import glob
import re
import sys
import argparse
from joblib import Parallel, delayed
import time
from datetime import datetime
import psutil
import threading

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import get_config, get_dataset_config, list_available_datasets

# Load configuration
config = get_config()

BASE_PATH = Path(__file__).parent.parent  # Project root (parent of src/)
SYNTHETIC_PATH_BASE = BASE_PATH / "data" / "synthetic"
MODELS_PATH = BASE_PATH / "models"

# These will be set dynamically based on dataset
PROCESSED_PATH = None
RESULTS_PATH = None
TABLES_PATH = None
FIGURES_PATH = None
SYNTHETIC_PATH = None
TARGET_FEATURE = None

CLASS_BENIGN = 0
CLASS_MALIGNANT = 1

RANDOM_STATE = config.experiment.random_state
GENERATORS_TO_TEST = config.models.generators

SYNTHETIC_PATH_BASE.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

monitoring_active = False

def format_bytes(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


def monitor_resources(interval=None):
    global monitoring_active
    if interval is None:
        interval = config.execution.resource_monitor.interval

    while monitoring_active:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        print(f"\r CPU: {cpu_percent:5.1f}% | RAM: {format_bytes(memory.used)}/{format_bytes(memory.total)} ({memory.percent:.1f}%)", end='', flush=True)
        time.sleep(interval)


def extract_dataset_info(filename: str):
    parts = filename.replace('.csv', '').split('_')
    
    info = {
        'dataset_type': None,
        'imbalance_ratio': None,
        'repetition_id': 1  
    }
    
    if 'imbalanced' in parts:
        info['dataset_type'] = 'imbalanced'
    elif 'control' in parts:
        info['dataset_type'] = 'control'
    
    if 'ir' in parts:
        ir_idx = parts.index('ir')
        if ir_idx + 1 < len(parts):
            ir_value = re.search(r'\d+', parts[ir_idx + 1])
            if ir_value:
                info['imbalance_ratio'] = int(ir_value.group())
    
    for part in parts:
        if part.startswith('rep'):
            rep_match = re.search(r'rep(\d+)', part)
            if rep_match:
                info['repetition_id'] = int(rep_match.group(1))
    
    return info


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def train_and_evaluate_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.drop(columns=[TARGET_FEATURE])
    y_train = train_df[TARGET_FEATURE]
    X_test = test_df.drop(columns=[TARGET_FEATURE])
    y_test = test_df[TARGET_FEATURE]

    model = RandomForestClassifier(
        random_state=config.models.random_forest.random_state,
        n_jobs=config.models.random_forest.n_jobs
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    pos_label_idx = 1 if 1 in model.classes_ else 0

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    minority_label_str = "1"

    metrics = {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_minority": report[minority_label_str]["f1-score"],
        "precision_minority": report[minority_label_str]["precision"],
        "recall_minority": report[minority_label_str]["recall"],
        "roc_auc": roc_auc_score(y_test, y_prob[:, pos_label_idx])
    }
    return metrics


def run_single_experiment(train_path: Path, test_path: Path, generator_name: str, strategy: str):
    train_df, test_df = load_data(train_path, test_path)
    dataset_info = extract_dataset_info(train_path.name)

    if generator_name.lower() == 'baseline':
        results = train_and_evaluate_classifier(train_df, test_df)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            loader = GenericDataLoader(train_df, target_column=TARGET_FEATURE)
            generator = Plugins().get(generator_name)
            generator.fit(loader, cond=train_df[TARGET_FEATURE])

            value_counts = train_df[TARGET_FEATURE].value_counts()
            n_benign = value_counts.get(CLASS_BENIGN, 0)
            n_malignant = value_counts.get(CLASS_MALIGNANT, 0)

            if n_benign > n_malignant:
                n_to_generate = n_benign - n_malignant
                minority_class_label = CLASS_MALIGNANT
                conditions = [minority_class_label] * n_to_generate
                
                synth_data = generator.generate(count=n_to_generate, cond=conditions).dataframe()
                balanced_train_df = pd.concat([train_df, synth_data], ignore_index=True)
                
                synth_filename = f"{train_path.stem}_balanced_by_{generator_name}.csv"
                balanced_train_df.to_csv(SYNTHETIC_PATH / synth_filename, index=False)
            else:
                balanced_train_df = train_df

            results = train_and_evaluate_classifier(balanced_train_df, test_df)

    result_log = {
        "repetition_id": dataset_info['repetition_id'],
        "dataset_type": dataset_info['dataset_type'],
        "imbalance_ratio": f"{dataset_info['imbalance_ratio']}:1",
        "model": generator_name,
        "strategy": strategy,
        "train_set_size": len(train_df),
        **results 
    }
    return result_log


def process_single_dataset(train_path: Path, test_path: Path, dataset_idx: int, total_datasets: int):
    results = []
    dataset_info = extract_dataset_info(train_path.name)
    start_time = time.time()
    
    # Baseline
    try:
        baseline_result = run_single_experiment(train_path, test_path, "baseline", "N/A")
        results.append(baseline_result)
    except Exception as e:
        print(f"\n[ERROR] Baseline failed on {train_path.name}: {e}")

    # Generators
    for generator in GENERATORS_TO_TEST:
        try:
            synthetic_result = run_single_experiment(
                train_path, test_path, generator, "Naive Oversampling"
            )
            results.append(synthetic_result)
        except Exception as e:
            print(f"\n[ERROR] Generator '{generator}' failed on {train_path.name}: {e}")
    
    return results

class ProgressTracker:
    def __init__(self, total_datasets):
        self.total_datasets = total_datasets
        self.completed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment(self):
        with self.lock:
            self.completed += 1
            self._display_progress()
    
    def _display_progress(self):
        elapsed = time.time() - self.start_time
        if self.completed > 0:
            avg_time = elapsed / self.completed
            remaining = avg_time * (self.total_datasets - self.completed)
            eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            eta = "calculating..."
        
        percent = (self.completed / self.total_datasets) * 100
        bar_length = 50
        filled = int(bar_length * self.completed / self.total_datasets)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\rProgress: [{bar}] {percent:5.1f}% ({self.completed}/{self.total_datasets}) | ETA: {eta}   ", end='', flush=True)


def setup_dataset_paths(dataset_name: str, timestamp: str):
    global PROCESSED_PATH, RESULTS_PATH, TABLES_PATH, FIGURES_PATH, SYNTHETIC_PATH, TARGET_FEATURE

    # Load dataset configuration
    dataset_config = get_dataset_config(dataset_name)

    # Set paths based on dataset configuration
    PROCESSED_PATH = BASE_PATH / dataset_config['processed_path']
    RESULTS_PATH = BASE_PATH / dataset_config['results_path']
    TABLES_PATH = RESULTS_PATH / "tables"
    FIGURES_PATH = RESULTS_PATH / "figures"
    SYNTHETIC_PATH = SYNTHETIC_PATH_BASE / f"run_{timestamp}" / dataset_name
    TARGET_FEATURE = dataset_config['target_column']

    # Create directories
    SYNTHETIC_PATH.mkdir(parents=True, exist_ok=True)
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def main(dataset_name: str = "mammographic_mass"):
    """Run experiments for a specific dataset.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup paths for this dataset
    setup_dataset_paths(dataset_name, timestamp)

    print(f"\n\n")
    print(f"Experimental Pipeline - Dataset: {dataset_name}")
    print(f"\n")

    print(f"\nDataset Configuration:")
    print(f"  • Target column: {TARGET_FEATURE}")
    print(f"  • Processed data: {PROCESSED_PATH}")
    print(f"  • Results path: {RESULTS_PATH}")
    print(f"  • Synthetic data: {SYNTHETIC_PATH}")

    start_time = time.time()
    test_path = PROCESSED_PATH / "test.csv"

    train_paths = sorted(glob.glob(str(PROCESSED_PATH / "train_*.csv")))
    
    if not train_paths:
        print(f"ERROR: No training datasets found in {PROCESSED_PATH}")
        return
    
    n_datasets = len(train_paths)
    n_methods = len(GENERATORS_TO_TEST) + 1
    total_experiments = n_datasets * n_methods
    
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    
    print(f"\n System Information:")
    print(f"   • CPU Cores: {cpu_count}")
    print(f"   • Total RAM: {format_bytes(memory.total)}")
    print(f"   • Available RAM: {format_bytes(memory.available)}")
    
    print(f"\n Experiment Configuration:")
    print(f"   • Datasets: {n_datasets}")
    print(f"   • Methods: {n_methods} (1 baseline + {len(GENERATORS_TO_TEST)} generators)")
    print(f"   • Total experiments: {total_experiments}")
    print(f"   • Parallel workers: {cpu_count}")
    
    global monitoring_active
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    train_paths = [Path(p) for p in train_paths]
    
    try:
        all_results_nested = Parallel(
            n_jobs=config.execution.parallel.n_jobs,
            backend=config.execution.parallel.backend,
            verbose=0
        )(
            delayed(process_single_dataset)(train_path, test_path, idx+1, n_datasets)
            for idx, train_path in enumerate(train_paths)
        )
    finally:
        monitoring_active = False
        time.sleep(0.5) 
        print("\r" + " "*120)  
    
    all_results = [result for dataset_results in all_results_nested for result in dataset_results]
    
    elapsed_time = time.time() - start_time
    
    print(" All experiments complete!")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"Speed: {total_experiments/elapsed_time:.2f} experiments/second")
    
    results_df = pd.DataFrame(all_results)

    summary_df = results_df.groupby(['dataset_type', 'imbalance_ratio', 'model', 'strategy']).agg(
        {
            'f1_minority': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'recall_minority': ['mean', 'std'],
            'precision_minority': ['mean', 'std']
        }
    ).reset_index()

    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.rename(columns={
        'dataset_type_': 'dataset_type',
        'imbalance_ratio_': 'imbalance_ratio',
        'model_': 'model',
        'strategy_': 'strategy'
    })

    run_folder = TABLES_PATH / f"run_{timestamp}"
    run_folder.mkdir(exist_ok=True)

    results_df.to_csv(run_folder / "detailed_experiment_results.csv", index=False)
    summary_df.to_csv(run_folder / "summary_experiment_results.csv", index=False)

    print(f"\nResults saved to: {run_folder}")
    print(f"Synthetic data saved to: {SYNTHETIC_PATH}")
    
    print("Performance by Model (F1-Minority / ROC-AUC):")
    
    summary_table = results_df.groupby('model').agg({
        'f1_minority': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }).round(4)
    
    for model in summary_table.index:
        f1_mean = summary_table.loc[model, ('f1_minority', 'mean')]
        f1_std = summary_table.loc[model, ('f1_minority', 'std')]
        auc_mean = summary_table.loc[model, ('roc_auc', 'mean')]
        auc_std = summary_table.loc[model, ('roc_auc', 'std')]
        print(f"   {model:<15} | F1: {f1_mean:.4f} (±{f1_std:.4f}) | AUC: {auc_mean:.4f} (±{auc_std:.4f})")
            
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run synthetic data generation experiments on tabular datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments on a single dataset
  python run_experiment.py --dataset mammographic_mass

  # Run experiments on all registered datasets
  python run_experiment.py --dataset all

  # List available datasets
  python run_experiment.py --list-datasets
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mammographic_mass",
        help="Name of the dataset to run experiments on, or 'all' to run on all datasets"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit"
    )

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    args = parse_arguments()

    # List datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        for dataset in list_available_datasets():
            dataset_config = get_dataset_config(dataset)
            print(f"  • {dataset}")
            print(f"    - Target: {dataset_config['target_column']}")
            print(f"    - Path: {dataset_config['processed_path']}")
            if 'description' in dataset_config:
                print(f"    - Description: {dataset_config['description']}")
        sys.exit(0)

    # Run experiments on specified dataset(s)
    if args.dataset.lower() == "all":
        print("Running experiments on all datasets...")
        available_datasets = list_available_datasets()
        print(f"Found {len(available_datasets)} datasets: {', '.join(available_datasets)}\n")

        for dataset in available_datasets:
            try:
                print(f"\n\n")
                print(f"Starting experiments for dataset: {dataset}")
                print(f"\n\n")
                main(dataset)
            except Exception as e:
                print(f"\nERROR running experiments for {dataset}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n\n")
        print("All dataset experiments completed!")
        print(f"\n\n")
    else:
        # Run on single dataset
        try:
            main(args.dataset)
        except ValueError as e:
            print(f"ERROR: {e}")
            print("\nUse --list-datasets to see available datasets")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)