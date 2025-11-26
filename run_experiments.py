#!/usr/bin/env python
"""
Run experiments to reproduce Table 1 in the paper.

Evaluates IPW, TARNet, CFRNet WASS, Dragonnet, EB, and Ebalnet 
on IHDP (1000 realizations) and JOBS (100 bootstrap samples) datasets.

Usage:
    python run_experiments.py --ihdp_realizations 1000 --jobs_bootstrap 100
    python run_experiments.py --dataset ihdp --ihdp_realizations 100
    python run_experiments.py --dataset jobs --jobs_bootstrap 50
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

# Import from package
from src.ebal_util import NNEbal, ebal_bin
from src.baseline_methods import IPW, TARNet, CFRNet_WASS, Dragonnet


def naive_ebal_predict(X, treatment, y, effect='ATT', use_pca=True, max_iter=10):
    """Entropy balancing on original covariates"""
    constraint_tolerance = 0.0001
    ebal_output = None
    
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    
    for _ in range(max_iter):
        try:
            ebal_model = ebal_bin(
                effect=effect,
                PCA=use_pca,
                print_level=-1,
                constraint_tolerance=constraint_tolerance,
            )
            ebal_output = ebal_model.ebalance(treatment, X_df, y)
            break
        except:
            constraint_tolerance = constraint_tolerance * 10
    
    if ebal_output is None:
        return None
    
    treatment_index = (treatment == 1)
    control_index = (treatment == 0)
    ebal_weight = ebal_output['w']
    
    mu1_hat = np.sum(y[treatment_index] * ebal_weight[treatment_index])
    mu0_hat = np.sum(y[control_index] * ebal_weight[control_index])
    
    return mu1_hat - mu0_hat


def load_ihdp_data(data_dir='data'):
    """Load IHDP dataset"""
    train_path = os.path.join(data_dir, 'ihdp_npci_1-1000.train.npz')
    test_path = os.path.join(data_dir, 'ihdp_npci_1-1000.test.npz')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"IHDP data files not found in {data_dir}. "
            "Please ensure ihdp_npci_1-1000.train.npz and ihdp_npci_1-1000.test.npz exist."
        )
    
    train = np.load(train_path)
    test = np.load(test_path)
    
    return {
        'X_tr': train['x'], 'T_tr': train['t'], 'YF_tr': train['yf'],
        'mu_0_tr': train['mu0'], 'mu_1_tr': train['mu1'],
        'X_te': test['x'], 'T_te': test['t'], 'YF_te': test['yf'],
        'mu_0_te': test['mu0'], 'mu_1_te': test['mu1'],
    }


def load_jobs_data(data_dir='data'):
    """Load JOBS dataset"""
    train_path = os.path.join(data_dir, 'jobs_DW_bin.new.10.train.npz')
    test_path = os.path.join(data_dir, 'jobs_DW_bin.new.10.test.npz')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"JOBS data files not found in {data_dir}. "
            "Please ensure jobs_DW_bin.new.10.train.npz and jobs_DW_bin.new.10.test.npz exist."
        )
    
    train = np.load(train_path)
    test = np.load(test_path)
    
    return {
        'X_tr': train['x'], 'T_tr': train['t'], 'Y_tr': train['yf'], 'E_tr': train['e'],
        'X_te': test['x'], 'T_te': test['t'], 'Y_te': test['yf'], 'E_te': test['e'],
    }


def evaluate_ihdp_realization(idx, data, nn_epochs=300, verbose=False):
    """Evaluate all methods on one IHDP realization"""
    results = {'idx': idx}
    
    try:
        # Extract data for this realization
        t_tr = data['T_tr'][:, idx]
        y_tr = data['YF_tr'][:, idx]
        x_tr = data['X_tr'][:, :, idx]
        mu0_tr = data['mu_0_tr'][:, idx]
        mu1_tr = data['mu_1_tr'][:, idx]
        
        t_te = data['T_te'][:, idx]
        y_te = data['YF_te'][:, idx]
        x_te = data['X_te'][:, :, idx]
        mu0_te = data['mu_0_te'][:, idx]
        mu1_te = data['mu_1_te'][:, idx]
        
        # Combine train and test
        t_al = np.r_[t_tr, t_te]
        y_al = np.r_[y_tr, y_te]
        x_al = np.r_[x_tr, x_te]
        mu0_al = np.r_[mu0_tr, mu0_te]
        mu1_al = np.r_[mu1_tr, mu1_te]
        
        # True ATT
        ATT_truth = (mu1_al[t_al == 1] - mu0_al[t_al == 1]).mean()
        results['ATT_truth'] = ATT_truth
        
        seed = 123400 + idx * 11
        
        # IPW
        try:
            ipw = IPW(random_seed=seed)
            ipw.fit(x_al, t_al, y_al)
            results['IPW'] = np.abs(ATT_truth - ipw.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['IPW'] = None
            if verbose: print(f"  IPW failed: {e}")
        
        # TARNet
        try:
            tarnet = TARNet(
                input_dim=x_al.shape[1], hidden_layers=3, hidden_units=200,
                repr_dim=100, epochs=nn_epochs, verbose=False, random_seed=seed
            )
            tarnet.fit(x_al, t_al, y_al)
            results['TARNet'] = np.abs(ATT_truth - tarnet.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['TARNet'] = None
            if verbose: print(f"  TARNet failed: {e}")
        
        # CFRNet WASS
        try:
            cfrnet = CFRNet_WASS(
                input_dim=x_al.shape[1], hidden_layers=3, hidden_units=200,
                repr_dim=100, alpha=1.0, epochs=nn_epochs, verbose=False, random_seed=seed
            )
            cfrnet.fit(x_al, t_al, y_al)
            results['CFRNet_WASS'] = np.abs(ATT_truth - cfrnet.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['CFRNet_WASS'] = None
            if verbose: print(f"  CFRNet failed: {e}")
        
        # Dragonnet
        try:
            dragonnet = Dragonnet(
                input_dim=x_al.shape[1], hidden_layers=3, hidden_units=200,
                repr_dim=100, epochs=nn_epochs, verbose=False, random_seed=seed
            )
            dragonnet.fit(x_al, t_al, y_al)
            results['Dragonnet'] = np.abs(ATT_truth - dragonnet.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['Dragonnet'] = None
            if verbose: print(f"  Dragonnet failed: {e}")
        
        # EB (naive entropy balancing)
        try:
            naive_att = naive_ebal_predict(x_al, t_al, y_al, effect='ATT', use_pca=True)
            if naive_att is not None:
                results['EB'] = np.abs(ATT_truth - naive_att)
            else:
                results['EB'] = None
        except Exception as e:
            results['EB'] = None
            if verbose: print(f"  EB failed: {e}")
        
        # Ebalnet
        try:
            params = {
                "input_dim": x_al.shape[1],
                "use_adam": True,
                'verbose': False,
                "act_fn": "gelu",
                "epochs": 500,
                "num_layers": 5,
                "embedding_dim": int(3 * np.sqrt(x_al.shape[0])),
                "neurons_per_layer": 100,
                "dropout_rate": 0,
                "reg_l2": 5,
                "learning_rate": 1e-3,
                "weighted_loss": False,
                'random_seed': seed,
            }
            nn_ebal = NNEbal(params)
            nn_ebal.fit(x_al, t_al, y_al)
            results['Ebalnet'] = np.abs(ATT_truth - nn_ebal.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['Ebalnet'] = None
            if verbose: print(f"  Ebalnet failed: {e}")
            
    except Exception as e:
        results['error'] = str(e)
    
    return results


def evaluate_jobs_bootstrap(bootstrap_idx, data, nn_epochs=300, verbose=False):
    """Evaluate all methods on one JOBS bootstrap sample"""
    realization_idx = bootstrap_idx % 10  # JOBS has 10 realizations
    results = {'bootstrap_idx': bootstrap_idx, 'realization_idx': realization_idx}
    
    try:
        # Combine train and test for this realization
        x_base = np.r_[data['X_tr'][:, :, realization_idx], data['X_te'][:, :, realization_idx]]
        t_base = np.r_[data['T_tr'][:, realization_idx], data['T_te'][:, realization_idx]]
        y_base = np.r_[data['Y_tr'][:, realization_idx], data['Y_te'][:, realization_idx]]
        e_base = np.r_[data['E_tr'][:, realization_idx], data['E_te'][:, realization_idx]]
        
        # Bootstrap sampling
        seed = 1234 + bootstrap_idx * 11
        np.random.seed(seed)
        n = len(y_base)
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        
        x_al = x_base[bootstrap_indices]
        t_al = t_base[bootstrap_indices]
        y_al = y_base[bootstrap_indices]
        e_al = e_base[bootstrap_indices]
        
        # True ATT from experimental sample
        exp_mask = e_al == 1
        exp_treated = (t_al == 1) & exp_mask
        exp_control = (t_al == 0) & exp_mask
        
        if np.sum(exp_treated) == 0 or np.sum(exp_control) == 0:
            results['error'] = 'No experimental samples in bootstrap'
            return results
        
        ATT_truth = np.mean(y_al[exp_treated]) - np.mean(y_al[exp_control])
        results['ATT_truth'] = ATT_truth
    
        
        # IPW
        try:
            ipw = IPW(random_seed=seed)
            ipw.fit(x_al, t_al, y_al)
            results['IPW'] = np.abs(ATT_truth - ipw.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['IPW'] = None
            if verbose: print(f"  IPW failed: {e}")
        
        # TARNet
        try:
            tarnet = TARNet(
                input_dim=x_al.shape[1], hidden_layers=3, hidden_units=100,
                repr_dim=50, epochs=nn_epochs, verbose=False, random_seed=seed
            )
            tarnet.fit(x_al, t_al, y_al)
            results['TARNet'] = np.abs(ATT_truth - tarnet.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['TARNet'] = None
            if verbose: print(f"  TARNet failed: {e}")
        
        # CFRNet WASS
        try:
            cfrnet = CFRNet_WASS(
                input_dim=x_al.shape[1], hidden_layers=3, hidden_units=100,
                repr_dim=50, alpha=1.0, epochs=nn_epochs, verbose=False, random_seed=seed
            )
            cfrnet.fit(x_al, t_al, y_al)
            results['CFRNet_WASS'] = np.abs(ATT_truth - cfrnet.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['CFRNet_WASS'] = None
            if verbose: print(f"  CFRNet failed: {e}")
        
        # Dragonnet
        try:
            dragonnet = Dragonnet(
                input_dim=x_al.shape[1], hidden_layers=3, hidden_units=100,
                repr_dim=50, epochs=nn_epochs, verbose=False, random_seed=seed
            )
            dragonnet.fit(x_al, t_al, y_al)
            results['Dragonnet'] = np.abs(ATT_truth - dragonnet.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['Dragonnet'] = None
            if verbose: print(f"  Dragonnet failed: {e}")
        
        # EB
        try:
            naive_att = naive_ebal_predict(x_al, t_al, y_al, effect='ATT', use_pca=True)
            if naive_att is not None:
                results['EB'] = np.abs(ATT_truth - naive_att)
            else:
                results['EB'] = None
        except Exception as e:
            results['EB'] = None
            if verbose: print(f"  EB failed: {e}")
        
        # Ebalnet
        try:
            params = {
                "input_dim": x_al.shape[1],
                "use_adam": True,
                'verbose': False,
                "act_fn": "gelu",
                "epochs": 500,
                "num_layers": 3,
                "embedding_dim": int(np.sqrt(x_al.shape[0])),
                "neurons_per_layer": 10,
                "dropout_rate": 0,
                "reg_l2": 0.1,
                "learning_rate": 1e-3,
                "weighted_loss": False,
                'task': 'reg',
                'random_seed': seed,
            }
            nn_ebal = NNEbal(params)
            nn_ebal.fit(x_al, t_al, y_al)
            results['Ebalnet'] = np.abs(ATT_truth - nn_ebal.predict_att(x_al, t_al, y_al))
        except Exception as e:
            results['Ebalnet'] = None
            if verbose: print(f"  Ebalnet failed: {e}")
            
    except Exception as e:
        results['error'] = str(e)
    
    return results


def summarize_results(results_list, methods):
    """Compute mean and standard error for each method"""
    df = pd.DataFrame(results_list)
    summary = {}
    
    for method in methods:
        if method in df.columns:
            errors = df[method].dropna().values
            if len(errors) > 0:
                mean_error = np.mean(errors)
                std_error = np.std(errors) / np.sqrt(len(errors))
                summary[method] = (mean_error, std_error, len(errors))
            else:
                summary[method] = (np.nan, np.nan, 0)
        else:
            summary[method] = (np.nan, np.nan, 0)
    
    return summary


def print_results(summary, dataset_name, methods):
    """Print results table"""
    print("\n" + "=" * 60)
    print(f"Results for {dataset_name} (ATT Absolute Error)")
    print("=" * 60)
    print(f"{'Method':<15} {'Mean':<12} {'SE':<12} {'N':<8}")
    print("-" * 50)
    
    for method in methods:
        mean, se, n = summary.get(method, (np.nan, np.nan, 0))
        if np.isnan(mean):
            print(f"{method:<15} {'N/A':<12} {'N/A':<12} {n:<8}")
        else:
            print(f"{method:<15} {mean:.4f}       {se:.4f}       {n:<8}")


def print_latex_table(ihdp_summary, jobs_summary, methods):
    """Print LaTeX formatted table"""
    method_names = {
        'IPW': 'IPW',
        'TARNet': 'TARNet',
        'CFRNet_WASS': 'CFRNet WASS',
        'Dragonnet': 'Dragonnet',
        'EB': 'EB',
        'Ebalnet': 'Ebalnet'
    }
    
    # Find best (lowest error) for bolding
    ihdp_vals = [ihdp_summary[m][0] for m in methods if not np.isnan(ihdp_summary.get(m, (np.nan,))[0])]
    jobs_vals = [jobs_summary[m][0] for m in methods if not np.isnan(jobs_summary.get(m, (np.nan,))[0])]
    
    ihdp_best = min(ihdp_vals) if ihdp_vals else np.nan
    jobs_best = min(jobs_vals) if jobs_vals else np.nan
    
    print("\n" + "=" * 70)
    print("LaTeX Table for Paper")
    print("=" * 70)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Comparison of ATT estimation error across methods on IHDP and JOBS datasets.")
    print(r"Entries are mean absolute error (and standard error) across samples. Lower values indicate better performance.}")
    print(r"\label{tab:att_error}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{IHDP} & \textbf{JOBS} \\")
    print(r"\midrule")
    
    for method in methods:
        ihdp_mean, ihdp_se, _ = ihdp_summary.get(method, (np.nan, np.nan, 0))
        jobs_mean, jobs_se, _ = jobs_summary.get(method, (np.nan, np.nan, 0))
        
        # Format IHDP
        if np.isnan(ihdp_mean):
            ihdp_str = "N/A"
        elif not np.isnan(ihdp_best) and np.isclose(ihdp_mean, ihdp_best, rtol=0.02):
            ihdp_str = f"\\textbf{{{ihdp_mean:.3f}}} $\\pm$ {ihdp_se:.3f}"
        else:
            ihdp_str = f"{ihdp_mean:.3f} $\\pm$ {ihdp_se:.3f}"
        
        # Format JOBS
        if np.isnan(jobs_mean):
            jobs_str = "N/A"
        elif not np.isnan(jobs_best) and np.isclose(jobs_mean, jobs_best, rtol=0.02):
            jobs_str = f"\\textbf{{{jobs_mean:.3f}}} $\\pm$ {jobs_se:.3f}"
        else:
            jobs_str = f"{jobs_mean:.3f} $\\pm$ {jobs_se:.3f}"
        
        print(f"{method_names[method]:<15} & {ihdp_str} & {jobs_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def run_ihdp_experiments(num_realizations, data_dir, nn_epochs, verbose):
    """Run IHDP experiments"""
    print("\n" + "=" * 60)
    print(f"Running IHDP Experiments ({num_realizations} realizations)")
    print("=" * 60)
    
    data = load_ihdp_data(data_dir)
    max_realizations = data['X_tr'].shape[2]
    
    if num_realizations > max_realizations:
        print(f"Warning: Requested {num_realizations} realizations but only {max_realizations} available")
        num_realizations = max_realizations
    
    results_list = []
    start_time = time.time()
    
    for idx in range(num_realizations):
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (idx + 1)) * (num_realizations - idx - 1) if idx > 0 else 0
            print(f"Processing realization {idx+1}/{num_realizations} "
                  f"(Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min)")
        
        result = evaluate_ihdp_realization(idx, data, nn_epochs, verbose)
        results_list.append(result)
        
        if 'error' in result and verbose:
            print(f"  Error on realization {idx}: {result['error']}")
    
    total_time = time.time() - start_time
    print(f"\nCompleted {num_realizations} IHDP realizations in {total_time/60:.1f} minutes")
    
    return results_list


def run_jobs_experiments(num_bootstrap, data_dir, nn_epochs, verbose):
    """Run JOBS experiments"""
    print("\n" + "=" * 60)
    print(f"Running JOBS Experiments ({num_bootstrap} bootstrap samples)")
    print("=" * 60)
    
    data = load_jobs_data(data_dir)
    
    results_list = []
    start_time = time.time()
    
    for idx in range(num_bootstrap):
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (idx + 1)) * (num_bootstrap - idx - 1) if idx > 0 else 0
            print(f"Processing bootstrap sample {idx+1}/{num_bootstrap} "
                  f"(Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min)")
        
        result = evaluate_jobs_bootstrap(idx, data, nn_epochs, verbose)
        results_list.append(result)
        
        if 'error' in result and verbose:
            print(f"  Error on bootstrap {idx}: {result['error']}")
    
    total_time = time.time() - start_time
    print(f"\nCompleted {num_bootstrap} JOBS bootstrap samples in {total_time/60:.1f} minutes")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(
        description='Run causal inference experiments for Ebalnet paper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='both', choices=['ihdp', 'jobs', 'both'],
                        help='Which dataset to run experiments on')
    parser.add_argument('--ihdp_realizations', type=int, default=1000,
                        help='Number of IHDP realizations (max 1000)')
    parser.add_argument('--jobs_bootstrap', type=int, default=100,
                        help='Number of JOBS bootstrap samples')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data files')
    parser.add_argument('--nn_epochs', type=int, default=300,
                        help='Number of epochs for neural network training')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress and errors')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    methods = ['IPW', 'TARNet', 'CFRNet_WASS', 'Dragonnet', 'EB', 'Ebalnet']
    
    print("=" * 60)
    print("Ebalnet Experiment Runner")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset(s): {args.dataset}")
    if args.dataset in ['ihdp', 'both']:
        print(f"IHDP realizations: {args.ihdp_realizations}")
    if args.dataset in ['jobs', 'both']:
        print(f"JOBS bootstrap samples: {args.jobs_bootstrap}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    ihdp_summary = {}
    jobs_summary = {}
    ihdp_results = []
    jobs_results = []
    
    # Run IHDP experiments
    if args.dataset in ['ihdp', 'both']:
        try:
            ihdp_results = run_ihdp_experiments(
                args.ihdp_realizations, args.data_dir, args.nn_epochs, args.verbose
            )
            ihdp_summary = summarize_results(ihdp_results, methods)
            print_results(ihdp_summary, "IHDP", methods)
            
            # Save detailed results
            ihdp_df = pd.DataFrame(ihdp_results)
            ihdp_df.to_csv(os.path.join(args.output_dir, 'ihdp_detailed_results.csv'), index=False)
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Skipping IHDP experiments.")
    
    # Run JOBS experiments
    if args.dataset in ['jobs', 'both']:
        try:
            jobs_results = run_jobs_experiments(
                args.jobs_bootstrap, args.data_dir, args.nn_epochs, args.verbose
            )
            jobs_summary = summarize_results(jobs_results, methods)
            print_results(jobs_summary, "JOBS", methods)
            
            # Save detailed results
            jobs_df = pd.DataFrame(jobs_results)
            jobs_df.to_csv(os.path.join(args.output_dir, 'jobs_detailed_results.csv'), index=False)
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Skipping JOBS experiments.")
    
    # Print LaTeX table if both datasets were run
    if ihdp_summary and jobs_summary:
        print_latex_table(ihdp_summary, jobs_summary, methods)
    
    # Save summary
    if ihdp_summary or jobs_summary:
        summary_data = {'Method': methods}
        if ihdp_summary:
            summary_data['IHDP_Mean'] = [ihdp_summary.get(m, (np.nan,))[0] for m in methods]
            summary_data['IHDP_SE'] = [ihdp_summary.get(m, (np.nan, np.nan))[1] for m in methods]
            summary_data['IHDP_N'] = [ihdp_summary.get(m, (np.nan, np.nan, 0))[2] for m in methods]
        if jobs_summary:
            summary_data['JOBS_Mean'] = [jobs_summary.get(m, (np.nan,))[0] for m in methods]
            summary_data['JOBS_SE'] = [jobs_summary.get(m, (np.nan, np.nan))[1] for m in methods]
            summary_data['JOBS_N'] = [jobs_summary.get(m, (np.nan, np.nan, 0))[2] for m in methods]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(args.output_dir, 'experiment_summary.csv'), index=False)
        
        print(f"\nResults saved to {args.output_dir}/")
        print("  - experiment_summary.csv")
        if ihdp_results:
            print("  - ihdp_detailed_results.csv")
        if jobs_results:
            print("  - jobs_detailed_results.csv")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

