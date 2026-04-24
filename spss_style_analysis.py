"""
SPSS-STYLE COMPREHENSIVE ANALYSIS
Generates statistical outputs and visualizations for all load balancing algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, levene
import warnings

warnings.filterwarnings('ignore')

# Set style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PROGRAM 1: COMPARE LOAD BALANCERS
# ============================================================================
print("\n" + "="*80)
print("PROGRAM 1: COMPARE_LOAD_BALANCERS.PY")
print("="*80)

from compare_load_balancers import run_comparison_evaluation

fold_results = run_comparison_evaluation()

# Extract accuracy data for Program 1
program1_data = {
    'DRL-LB': [r.drl_lb for r in fold_results],
    'Round Robin': [r.round_robin for r in fold_results],
    'Least Connection': [r.least_connection for r in fold_results],
    'Static Heuristic': [r.static_heuristic for r in fold_results]
}

df_prog1 = pd.DataFrame(program1_data)

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS - PROGRAM 1")
print("="*80)
print(df_prog1.describe())

print("\n" + "-"*80)
print("GROUP STATISTICS")
print("-"*80)
for col in df_prog1.columns:
    print(f"\n{col}:")
    print(f"  N: {len(df_prog1[col])}")
    print(f"  Mean: {df_prog1[col].mean():.4f}")
    print(f"  Std. Deviation: {df_prog1[col].std():.4f}")
    print(f"  Std. Error Mean: {df_prog1[col].std()/np.sqrt(len(df_prog1[col])):.4f}")

# ANOVA Test
print("\n" + "-"*80)
print("ANOVA - Test for Equality of Means")
print("-"*80)
f_stat, p_value = f_oneway(*[df_prog1[col].values for col in df_prog1.columns])
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value (Sig.): {p_value:.4f}")

# Levene's Test
print("\n" + "-"*80)
print("LEVENE'S TEST - Test for Equality of Variances")
print("-"*80)
levene_stat, levene_p = levene(*[df_prog1[col].values for col in df_prog1.columns])
print(f"F-statistic: {levene_stat:.4f}")
print(f"P-value (Sig.): {levene_p:.4f}")

# T-Tests for top 2 performers
print("\n" + "-"*80)
print("INDEPENDENT SAMPLES T-TEST (Program 1)")
print("-"*80)
best_two = sorted(df_prog1.columns, key=lambda x: df_prog1[x].mean(), reverse=True)[:2]
t_stat, t_pval = ttest_ind(df_prog1[best_two[0]], df_prog1[best_two[1]])
print(f"Comparing: {best_two[0]} vs {best_two[1]}")
print(f"Mean Difference: {df_prog1[best_two[0]].mean() - df_prog1[best_two[1]].mean():.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"P-value (2-tailed): {t_pval:.4f}")
print(f"95% CI Lower: {(df_prog1[best_two[0]].mean() - df_prog1[best_two[1]].mean()) - 1.96*(df_prog1[best_two[0]].std()/np.sqrt(len(df_prog1[best_two[0]]))):.4f}")
print(f"95% CI Upper: {(df_prog1[best_two[0]].mean() - df_prog1[best_two[1]].mean()) + 1.96*(df_prog1[best_two[0]].std()/np.sqrt(len(df_prog1[best_two[0]]))):.4f}")

# Create SPSS-style graphs for Program 1
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PROGRAM 1: COMPARE_LOAD_BALANCERS - Statistical Analysis', fontsize=14, fontweight='bold')

# Graph 1: Bar chart with error bars
ax1 = axes[0, 0]
means = df_prog1.mean()
stds = df_prog1.std()
ax1.bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_xticks(range(len(means)))
ax1.set_xticklabels(means.index, rotation=45, ha='right')
ax1.set_ylabel('Mean ACCURACY')
ax1.set_xlabel('GROUP')
ax1.set_title('Simple Bar Mean of ACCURACY by GROUP\nError Bars: 95% CI')
ax1.grid(axis='y', alpha=0.3)

# Graph 2: Box plot
ax2 = axes[0, 1]
df_prog1.boxplot(ax=ax2)
ax2.set_ylabel('ACCURACY')
ax2.set_title('Box Plot - Accuracy Distribution')
ax2.grid(axis='y', alpha=0.3)

# Graph 3: Line plot
ax3 = axes[1, 0]
for col in df_prog1.columns:
    ax3.plot(range(1, 11), df_prog1[col].values, marker='o', label=col, linewidth=2)
ax3.set_xlabel('Fold')
ax3.set_ylabel('Accuracy')
ax3.set_title('Accuracy by Fold')
ax3.legend()
ax3.grid(alpha=0.3)

# Graph 4: Violin plot
ax4 = axes[1, 1]
df_prog1.plot.box(ax=ax4, grid=True)
ax4.set_ylabel('ACCURACY')
ax4.set_title('Distribution by Algorithm')

plt.tight_layout()
plt.savefig('c:/Users/student/Desktop/project/Program1_SPSS_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Program 1 graph saved: Program1_SPSS_Analysis.png")
plt.close()

# ============================================================================
# PROGRAM 2: BFL-LB COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PROGRAM 2: BFL_LB_COMPARISON.PY")
print("="*80)

from bfl_lb_comparison import run_bfl_evaluation

bfl_results = run_bfl_evaluation()

# Extract accuracy data for Program 2
program2_data = {
    'BFL-LB': [r['bfl'] for r in bfl_results],
    'Centralized': [r['centralized'] for r in bfl_results],
    'Non-Secure Distributed': [r['non_secure'] for r in bfl_results],
    'Heuristic Security': [r['heuristic'] for r in bfl_results]
}

df_prog2 = pd.DataFrame(program2_data)

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS - PROGRAM 2")
print("="*80)
print(df_prog2.describe())

print("\n" + "-"*80)
print("GROUP STATISTICS")
print("-"*80)
for col in df_prog2.columns:
    print(f"\n{col}:")
    print(f"  N: {len(df_prog2[col])}")
    print(f"  Mean: {df_prog2[col].mean():.4f}")
    print(f"  Std. Deviation: {df_prog2[col].std():.4f}")
    print(f"  Std. Error Mean: {df_prog2[col].std()/np.sqrt(len(df_prog2[col])):.4f}")

# T-Test for Primary vs Secondary
print("\n" + "-"*80)
print("INDEPENDENT SAMPLES T-TEST (BFL-LB vs Centralized)")
print("-"*80)
t_stat, t_pval = ttest_ind(df_prog2['BFL-LB'], df_prog2['Centralized'])
mean_diff = df_prog2['BFL-LB'].mean() - df_prog2['Centralized'].mean()
se_diff = np.sqrt((df_prog2['BFL-LB'].std()**2 / len(df_prog2['BFL-LB'])) + (df_prog2['Centralized'].std()**2 / len(df_prog2['Centralized'])))
ci_lower = mean_diff - 1.96 * se_diff
ci_upper = mean_diff + 1.96 * se_diff

print(f"Mean Difference: {mean_diff:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"P-value (2-tailed): {t_pval:.4f}")
print(f"95% CI Lower: {ci_lower:.4f}")
print(f"95% CI Upper: {ci_upper:.4f}")

# Create SPSS-style graphs for Program 2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PROGRAM 2: BFL_LB_COMPARISON - Statistical Analysis', fontsize=14, fontweight='bold')

# Graph 1: Bar chart
ax1 = axes[0, 0]
means = df_prog2.mean()
stds = df_prog2.std()
ax1.bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_xticks(range(len(means)))
ax1.set_xticklabels(means.index, rotation=45, ha='right')
ax1.set_ylabel('Mean ACCURACY')
ax1.set_xlabel('GROUP')
ax1.set_title('Simple Bar Mean of ACCURACY by GROUP\nError Bars: 95% CI')
ax1.grid(axis='y', alpha=0.3)

# Graph 2: Box plot
ax2 = axes[0, 1]
df_prog2.boxplot(ax=ax2)
ax2.set_ylabel('ACCURACY')
ax2.set_title('Box Plot - Accuracy Distribution')
ax2.grid(axis='y', alpha=0.3)

# Graph 3: Line plot
ax3 = axes[1, 0]
for col in df_prog2.columns:
    ax3.plot(range(1, 11), df_prog2[col].values, marker='o', label=col, linewidth=2)
ax3.set_xlabel('Fold')
ax3.set_ylabel('Accuracy')
ax3.set_title('Accuracy by Fold')
ax3.legend()
ax3.grid(alpha=0.3)

# Graph 4: Comparison
ax4 = axes[1, 1]
comparison_data = [df_prog2['BFL-LB'], df_prog2['Centralized']]
ax4.violinplot(comparison_data, positions=[1, 2], showmeans=True)
ax4.set_xticks([1, 2])
ax4.set_xticklabels(['BFL-LB', 'Centralized'])
ax4.set_ylabel('ACCURACY')
ax4.set_title('Distribution Comparison: BFL-LB vs Centralized')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('c:/Users/student/Desktop/project/Program2_SPSS_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Program 2 graph saved: Program2_SPSS_Analysis.png")
plt.close()

# ============================================================================
# PROGRAM 3: GNN-LB COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PROGRAM 3: GNN_LB_COMPARISON.PY")
print("="*80)

from gnn_lb_comparison import run_gnn_evaluation

gnn_results = run_gnn_evaluation()

# Extract accuracy data for Program 3
program3_data = {
    'GNN-LB': [r['gnn'] for r in gnn_results],
    'Centralized': [r['centralized'] for r in gnn_results],
    'Non-Secure Distributed': [r['non_secure'] for r in gnn_results],
    'Heuristic Security': [r['heuristic'] for r in gnn_results]
}

df_prog3 = pd.DataFrame(program3_data)

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS - PROGRAM 3")
print("="*80)
print(df_prog3.describe())

print("\n" + "-"*80)
print("GROUP STATISTICS")
print("-"*80)
for col in df_prog3.columns:
    print(f"\n{col}:")
    print(f"  N: {len(df_prog3[col])}")
    print(f"  Mean: {df_prog3[col].mean():.4f}")
    print(f"  Std. Deviation: {df_prog3[col].std():.4f}")
    print(f"  Std. Error Mean: {df_prog3[col].std()/np.sqrt(len(df_prog3[col])):.4f}")

# Create SPSS-style graphs for Program 3
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PROGRAM 3: GNN_LB_COMPARISON - Statistical Analysis', fontsize=14, fontweight='bold')

# Graph 1: Bar chart
ax1 = axes[0, 0]
means = df_prog3.mean()
stds = df_prog3.std()
ax1.bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_xticks(range(len(means)))
ax1.set_xticklabels(means.index, rotation=45, ha='right')
ax1.set_ylabel('Mean ACCURACY')
ax1.set_xlabel('GROUP')
ax1.set_title('Simple Bar Mean of ACCURACY by GROUP\nError Bars: 95% CI')
ax1.grid(axis='y', alpha=0.3)

# Graph 2: Box plot
ax2 = axes[0, 1]
df_prog3.boxplot(ax=ax2)
ax2.set_ylabel('ACCURACY')
ax2.set_title('Box Plot - Accuracy Distribution')
ax2.grid(axis='y', alpha=0.3)

# Graph 3: Line plot
ax3 = axes[1, 0]
for col in df_prog3.columns:
    ax3.plot(range(1, 11), df_prog3[col].values, marker='o', label=col, linewidth=2)
ax3.set_xlabel('Fold')
ax3.set_ylabel('Accuracy')
ax3.set_title('Accuracy by Fold')
ax3.legend()
ax3.grid(alpha=0.3)

# Graph 4: Comparison
ax4 = axes[1, 1]
comparison_data = [df_prog3['GNN-LB'], df_prog3['Centralized']]
ax4.violinplot(comparison_data, positions=[1, 2], showmeans=True)
ax4.set_xticks([1, 2])
ax4.set_xticklabels(['GNN-LB', 'Centralized'])
ax4.set_ylabel('ACCURACY')
ax4.set_title('Distribution Comparison: GNN-LB vs Centralized')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('c:/Users/student/Desktop/project/Program3_SPSS_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Program 3 graph saved: Program3_SPSS_Analysis.png")
plt.close()

# ============================================================================
# PROGRAM 4: MOEO-LB COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PROGRAM 4: MOEO_LB_COMPARISON.PY")
print("="*80)

from moeo_lb_comparison import run_moeo_evaluation

moeo_results = run_moeo_evaluation()

# Extract accuracy data for Program 4
program4_data = {
    'MOEO-LB': [r['moeo'] for r in moeo_results],
    'Topology-Agnostic': [r['topo'] for r in moeo_results],
    'Centralized': [r['centralized'] for r in moeo_results],
    'Traditional Distributed': [r['traditional'] for r in moeo_results]
}

df_prog4 = pd.DataFrame(program4_data)

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS - PROGRAM 4")
print("="*80)
print(df_prog4.describe())

print("\n" + "-"*80)
print("GROUP STATISTICS")
print("-"*80)
for col in df_prog4.columns:
    print(f"\n{col}:")
    print(f"  N: {len(df_prog4[col])}")
    print(f"  Mean: {df_prog4[col].mean():.4f}")
    print(f"  Std. Deviation: {df_prog4[col].std():.4f}")
    print(f"  Std. Error Mean: {df_prog4[col].std()/np.sqrt(len(df_prog4[col])):.4f}")

# Create SPSS-style graphs for Program 4
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PROGRAM 4: MOEO_LB_COMPARISON - Statistical Analysis', fontsize=14, fontweight='bold')

# Graph 1: Bar chart
ax1 = axes[0, 0]
means = df_prog4.mean()
stds = df_prog4.std()
ax1.bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_xticks(range(len(means)))
ax1.set_xticklabels(means.index, rotation=45, ha='right')
ax1.set_ylabel('Mean ACCURACY')
ax1.set_xlabel('GROUP')
ax1.set_title('Simple Bar Mean of ACCURACY by GROUP\nError Bars: 95% CI')
ax1.grid(axis='y', alpha=0.3)

# Graph 2: Box plot
ax2 = axes[0, 1]
df_prog4.boxplot(ax=ax2)
ax2.set_ylabel('ACCURACY')
ax2.set_title('Box Plot - Accuracy Distribution')
ax2.grid(axis='y', alpha=0.3)

# Graph 3: Line plot
ax3 = axes[1, 0]
for col in df_prog4.columns:
    ax3.plot(range(1, 11), df_prog4[col].values, marker='o', label=col, linewidth=2)
ax3.set_xlabel('Fold')
ax3.set_ylabel('Accuracy')
ax3.set_title('Accuracy by Fold')
ax3.legend()
ax3.grid(alpha=0.3)

# Graph 4: Comparison
ax4 = axes[1, 1]
comparison_data = [df_prog4['MOEO-LB'], df_prog4['Topology-Agnostic']]
ax4.violinplot(comparison_data, positions=[1, 2], showmeans=True)
ax4.set_xticks([1, 2])
ax4.set_xticklabels(['MOEO-LB', 'Topology-Agnostic'])
ax4.set_ylabel('ACCURACY')
ax4.set_title('Distribution Comparison: MOEO-LB vs Topology-Agnostic')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('c:/Users/student/Desktop/project/Program4_SPSS_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Program 4 graph saved: Program4_SPSS_Analysis.png")
plt.close()

# ============================================================================
# SUMMARY COMPARISON - ALL PROGRAMS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: ALL PROGRAMS COMPARISON")
print("="*80)

summary_stats = pd.DataFrame({
    'Program 1 Best': [df_prog1.mean().max()],
    'Program 2 Best': [df_prog2.mean().max()],
    'Program 3 Best': [df_prog3.mean().max()],
    'Program 4 Best': [df_prog4.mean().max()]
})

print("\nBest Mean Accuracy by Program:")
print(summary_stats)

# Overall comparison graph
fig, ax = plt.subplots(figsize=(12, 7))

all_means = {
    'Program 1': df_prog1.mean().max(),
    'Program 2': df_prog2.mean().max(),
    'Program 3': df_prog3.mean().max(),
    'Program 4': df_prog4.mean().max()
}

programs = list(all_means.keys())
values = list(all_means.values())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = ax.bar(programs, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
ax.set_xlabel('Program', fontsize=12, fontweight='bold')
ax.set_title('Overall Best Algorithm Performance by Program', fontsize=14, fontweight='bold')
ax.set_ylim([0, max(values) * 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('c:/Users/student/Desktop/project/All_Programs_Summary.png', dpi=300, bbox_inches='tight')
print("\n✓ Summary graph saved: All_Programs_Summary.png")
plt.close()

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated Files:")
print("  1. Program1_SPSS_Analysis.png")
print("  2. Program2_SPSS_Analysis.png")
print("  3. Program3_SPSS_Analysis.png")
print("  4. Program4_SPSS_Analysis.png")
print("  5. All_Programs_Summary.png")
print("\n" + "="*80)
