import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
#evrificar os thresholds nos excels do ablation, nao é suposto ter.

def plot_confusion_matrix(cm, axis_name, save_path, title_suffix="", normalize=False):
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm_normalized
        fmt = '.2%'
        cmap = 'Blues'
    else:
        cm_to_plot = cm
        fmt = 'd'
        cmap = 'Blues'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=['Pred_0', 'Pred_1'], 
                yticklabels=['True_0', 'True_1'],
                cbar_kws={'label': 'Percentage' if normalize else 'Count'})
    
    title = f'Confusion Matrix - {axis_name}'
    if title_suffix:
        title += f' ({title_suffix})'
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def cross_validation_summary(is_ablation, all_folds_test_metrics, axes, base_results_dir, threshold_tuning=False):
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}\n")

    avg_test_loss = np.mean([f['test_loss'] for f in all_folds_test_metrics])
    avg_test_acc = np.mean([f['test_avg_acc'] for f in all_folds_test_metrics])
    avg_test_f1 = np.mean([f['test_avg_f1'] for f in all_folds_test_metrics])

    std_test_loss = np.std([f['test_loss'] for f in all_folds_test_metrics])
    std_test_acc = np.std([f['test_avg_acc'] for f in all_folds_test_metrics])
    std_test_f1 = np.std([f['test_avg_f1'] for f in all_folds_test_metrics])

    avg_test_accs_per_axis = np.mean([f['test_accs'] for f in all_folds_test_metrics], axis=0)
    avg_test_f1s_per_axis = np.mean([f['test_f1s'] for f in all_folds_test_metrics], axis=0)
    std_test_accs_per_axis = np.std([f['test_accs'] for f in all_folds_test_metrics], axis=0)
    std_test_f1s_per_axis = np.std([f['test_f1s'] for f in all_folds_test_metrics], axis=0)

    if threshold_tuning:
        avg_thresholds_per_axis = np.mean([f['best_thresholds'] for f in all_folds_test_metrics], axis=0)
        std_thresholds_per_axis = np.std([f['best_thresholds'] for f in all_folds_test_metrics], axis=0)

    print(f"Average Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"Average Test Acc: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Average Test F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}\n")

    if threshold_tuning:
        print("Average Thresholds across folds:")
        for i, axis in enumerate(axes):
            print(f"{axis}: {avg_thresholds_per_axis[i]:.3f} ± {std_thresholds_per_axis[i]:.3f}")
        print()

    for i, axis in enumerate(axes):
        threshold_str = f" | Avg Thr: {avg_thresholds_per_axis[i]:.3f} ± {std_thresholds_per_axis[i]:.3f}" if threshold_tuning else ""
        print(f"{axis} | Avg Test Acc: {avg_test_accs_per_axis[i]:.4f} ± {std_test_accs_per_axis[i]:.4f} | "
            f"Avg Test F1: {avg_test_f1s_per_axis[i]:.4f} ± {std_test_f1s_per_axis[i]:.4f}{threshold_str}")

    cv_summary_path = os.path.join(base_results_dir, f'cv_summary.xlsx')
    with pd.ExcelWriter(cv_summary_path, engine='openpyxl') as summary_writer:
        if is_ablation:
            overall_df = pd.DataFrame({
                'Metric': ['Avg_Test_Loss', 'Avg_Test_Acc', 'Avg_Test_F1'],
                'Mean': [avg_test_loss, avg_test_acc, avg_test_f1],
            })
        else:
            overall_df = pd.DataFrame({
                'Metric': ['Avg_Test_Loss', 'Avg_Test_Acc', 'Avg_Test_F1'],
                'Mean': [avg_test_loss, avg_test_acc, avg_test_f1],
                'Std': [std_test_loss, std_test_acc, std_test_f1]
            })
    
        overall_df.to_excel(summary_writer, sheet_name='Overall_Averages', index=False)
        
        if is_ablation:
            overall_df_per_axis = pd.DataFrame({
                'Trait': axes,
                'Avg_Test_Acc': avg_test_accs_per_axis,
                'Avg_Test_F1': avg_test_f1s_per_axis
            })
        else:
            per_axis_dict = {
                'Trait': axes,
                'Avg_Test_Acc': avg_test_accs_per_axis,
                'Std_Test_Acc': std_test_accs_per_axis,
                'Avg_Test_F1': avg_test_f1s_per_axis,
                'Std_Test_F1': std_test_f1s_per_axis
            }
        
        if threshold_tuning:
            per_axis_dict['Avg_Threshold'] = avg_thresholds_per_axis
            per_axis_dict['Std_Threshold'] = std_thresholds_per_axis
        
        per_axis_df = pd.DataFrame(per_axis_dict)
        per_axis_df.to_excel(summary_writer, sheet_name='Per_Axis_Averages', index=False)
        
        if not is_ablation:
            folds_overall_dict = {
                'Fold': [f['fold'] for f in all_folds_test_metrics],
                'Best_Epoch': [f['best_epoch'] for f in all_folds_test_metrics],
                'Test_Loss': [f['test_loss'] for f in all_folds_test_metrics],
                'Avg_Test_Acc': [f['test_avg_acc'] for f in all_folds_test_metrics],
                'Avg_Test_F1': [f['test_avg_f1'] for f in all_folds_test_metrics]
            }
            
            if threshold_tuning:
                for i, axis in enumerate(axes):
                    folds_overall_dict[f'Threshold_{axis}'] = [f['best_thresholds'][i] for f in all_folds_test_metrics]
            
            folds_overall_df = pd.DataFrame(folds_overall_dict)
            folds_overall_df.to_excel(summary_writer, sheet_name='Folds_Overall', index=False)
            
            fold_trait_rows = []
            for f in all_folds_test_metrics:
                for i, axis in enumerate(axes):
                    row_dict = {
                        'Fold': f['fold'],
                        'Trait': axis,
                        'Test_Acc': f['test_accs'][i],
                        'Test_F1': f['test_f1s'][i]
                    }
                    if threshold_tuning:
                        row_dict['Threshold'] = f['best_thresholds'][i]
                    fold_trait_rows.append(row_dict)
            
            folds_per_trait_df = pd.DataFrame(fold_trait_rows)
            folds_per_trait_df.to_excel(summary_writer, sheet_name='Folds_Per_Trait', index=False)
        
        n_folds = len(all_folds_test_metrics)

        for i, axis in enumerate(axes):
            fig, axes_plot = plt.subplots(1, n_folds, figsize=(6*n_folds, 5))
            
            if n_folds == 1:
                axes_plot = [axes_plot]
            
            for fold_num, f in enumerate(all_folds_test_metrics):
                threshold = f['best_thresholds'][i] if threshold_tuning else 0.5
                cm = confusion_matrix(
                    np.array(f['test_labels'][i]),
                    (np.array(f['test_preds'][i]) > threshold).astype(int)
                )
                
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                        ax=axes_plot[fold_num],
                        xticklabels=['Pred_0', 'Pred_1'],
                        yticklabels=['True_0', 'True_1'],
                        cbar_kws={'label': 'Percentage'},
                        vmin=0, vmax=1)
                
                title = f'Fold {fold_num + 1}'
                if threshold_tuning:
                    title += f'\nThr={threshold:.3f}'
                axes_plot[fold_num].set_title(title)
                axes_plot[fold_num].set_ylabel('True Label')
                axes_plot[fold_num].set_xlabel('Predicted Label')
            
            main_title = f'Confusion Matrices Comparison - {axis} (Normalized)'
            if threshold_tuning:
                main_title += ' - with Threshold Tuning'
            plt.suptitle(main_title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            comparison_path = os.path.join(base_results_dir, f'CM_comparison_{axis}.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved confusion matrix comparison for {axis}")
        
        aggregated_cm_dir = os.path.join(base_results_dir, 'aggregated_confusion_matrices')
        os.makedirs(aggregated_cm_dir, exist_ok=True)
        
        for i, axis in enumerate(axes):
            cms = []
            for f in all_folds_test_metrics:
                threshold = f['best_thresholds'][i] if threshold_tuning else 0.5
                cm = confusion_matrix(
                    np.array(f['test_labels'][i]),
                    (np.array(f['test_preds'][i]) > threshold).astype(int)
                )
                cms.append(cm)
            
            aggregated_cm = np.sum(cms, axis=0)
            
            title_suffix = 'Aggregated across all folds'
            if threshold_tuning:
                title_suffix += f' (Avg Thr={avg_thresholds_per_axis[i]:.3f})'
            
            png_path = os.path.join(aggregated_cm_dir, f'Aggregated_CM_{axis}_counts.png')
            plot_confusion_matrix(aggregated_cm, axis, png_path, 
                                title_suffix=title_suffix,
                                normalize=False)
            
            png_path_norm = os.path.join(aggregated_cm_dir, f'Aggregated_CM_{axis}_normalized.png')
            plot_confusion_matrix(aggregated_cm, axis, png_path_norm, 
                                title_suffix=title_suffix,
                                normalize=True)

    print(f"\nCross-validation summary saved to {cv_summary_path}")
    print(f"All results saved to {base_results_dir}")



def fold_plots(dataset_name, actual_epochs, train_losses_per_epoch, test_losses_per_epoch, avg_test_f1s_per_epoch, fold_idx, fold_dir, threshold_tuning=False):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    ax1.plot(range(1, actual_epochs+1), train_losses_per_epoch, label='Train Loss', marker='o', color=color1, linestyle='-')
    ax1.plot(range(1, actual_epochs+1), test_losses_per_epoch, label='Val Loss', marker='s', color='tab:cyan', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    if dataset_name == "Kaggle":
        ax2.set_ylabel('Avg Macro F1', color=color2)
        ax2.plot(range(1, actual_epochs+1), avg_test_f1s_per_epoch, label='Avg Val Macro-F1', marker='x', color=color2, linestyle='--')
    else:
        ax2.set_ylabel('Avg Acc', color=color2)
        ax2.plot(range(1, actual_epochs+1), avg_test_f1s_per_epoch, label='Avg Val Acc', marker='x', color=color2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.legend(loc='upper right')
    
    if dataset_name == "Kaggle":
        title = f'Training/Validation Loss & Avg Val Macro-F1 per Epoch'
    else:
        title = f'Training/Validation Loss & Avg Val Acc per Epoch'
    if threshold_tuning:
        title += ' (with Threshold Tuning)'
    plt.title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(fold_dir, f'train_val_metrics_plot.png'), dpi=150)
    plt.close()