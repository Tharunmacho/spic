# Load Balancing SPSS-Style Statistical Analysis

## Project Overview
This project contains comprehensive SPSS-style statistical analysis of 4 different load balancing algorithms with 10-fold cross-validation and professional visualizations.

## Files Included

### Python Programs
1. **compare_load_balancers.py** - Compares DRL-LB, Round Robin, Least Connection, and Static Heuristic
2. **bfl_lb_comparison.py** - BFL-LB vs Centralized Scheduler with security variants
3. **gnn_lb_comparison.py** - GNN-LB based comparison
4. **moeo_lb_comparison.py** - Multi-Objective Evolutionary Optimization analysis

### Analysis Scripts
- **spss_simple_analysis.py** - Main SPSS-style analysis generator
- **spss_analysis_final.py** - Enhanced analysis version
- **spss_comprehensive_analysis.py** - Comprehensive analysis suite

### Generated Reports & Visualizations
- **Program1_SPSS_Analysis.png** - Compare Load Balancers statistical graphs
- **Program2_SPSS_Analysis.png** - BFL-LB comparison analysis
- **Program3_SPSS_Analysis.png** - GNN-LB comparison analysis
- **Program4_SPSS_Analysis.png** - MOEO-LB comparison analysis
- **All_Programs_Summary.png** - Overall performance comparison
- **SPSS_Analysis_Results_Summary.txt** - Detailed text report

### Datasets
- **Dynamic_Workflow_Scheduling_Dataset.csv** - Main scheduling dataset
- **Load Balancing Improved.csv** - Optimized load balancing dataset
- **multi_cloud_service_dataset.csv** - Multi-cloud environment data

## Key Results

### Best Performing Algorithms
1. **BFL-LB (Blockchain Federated Learning)**: 95.80% mean accuracy
2. **GNN-LB (Graph Neural Network)**: 95.80% mean accuracy
3. Both significantly outperform traditional methods (p < 0.001)

### Statistical Metrics
- **10-Fold Cross-Validation**: Stratified K-fold
- **Confidence Intervals**: 95% CI on all means
- **Statistical Tests**: ANOVA, T-tests, Levene's test
- **Consistency Metrics**: Standard deviation, coefficient of variation

## How to Use

### Run Analysis
```bash
python spss_simple_analysis.py
```

### View Results
1. Open PNG files in image viewer to see SPSS-style graphs
2. Read SPSS_Analysis_Results_Summary.txt for detailed statistics
3. Check individual program outputs for fold-by-fold performance

## Visualization Features
- **Bar Charts with 95% CI Error Bars**: Mean accuracy comparisons
- **Box Plots**: Distribution analysis and outlier detection
- **Violin Plots**: Probability density visualization
- **Line Plots**: Fold-by-fold performance trends

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

## Installation
```bash
pip install pandas numpy matplotlib scipy scikit-learn seaborn
```

## Statistical Analysis Performed
- [x] Descriptive Statistics (Mean, Std Dev, Min, Max, Quartiles)
- [x] Group Statistics with Standard Error
- [x] ANOVA F-test for equality of means
- [x] Independent Samples T-Test (95% Confidence Intervals)
- [x] Levene's Test for Equality of Variances
- [x] Box Plots and Violin Plots
- [x] Error Bar Charts with 95% CI
- [x] Line plots for fold-by-fold performance

## Key Findings

### Program 1: Compare Load Balancers
- **Best Algorithm**: Static Heuristic
- **Mean Accuracy**: 0.20 (20%)
- **Status**: All algorithms have similar performance

### Program 2: BFL-LB Comparison
- **Best Algorithm**: BFL-LB (Blockchain Federated Learning-Based Load Balancer)
- **Mean Accuracy**: 95.80%
- **Improvement over Centralized**: 12.20% (p < 0.001)
- **Ranking**:
  1. BFL-LB: 95.80% (SD: 0.60)
  2. Non-Secure Distributed: 93.60% (SD: 0.60)
  3. Centralized Scheduler: 83.60% (SD: 1.20)
  4. Heuristic Security: 62.20% (SD: 1.10)

### Program 3: GNN-LB Comparison
- **Best Algorithm**: GNN-LB (Graph Neural Network-Based Load Balancer)
- **Mean Accuracy**: 95.80%
- **Improvement over Centralized**: 12.20% (p < 0.001)
- **Ranking**: Same as Program 2

### Program 4: MOEO-LB Comparison
- **Best Algorithm**: GNN-LB (Multi-Objective Evolutionary Optimization)
- **Mean Accuracy**: 95.80%
- **Improvement over Topology-Agnostic**: 24.40% (p < 0.001)
- **Ranking**:
  1. GNN-LB: 95.80% (SD: 0.60)
  2. Traditional Distributed: 89.90% (SD: 0.80)
  3. Centralized Scheduler: 83.60% (SD: 1.20)
  4. Topology-Agnostic: 71.40% (SD: 1.50)

## Overall Conclusions

1. **Best Overall Algorithms**: BFL-LB and GNN-LB (both at 95.80%)
2. **Most Consistent**: GNN-LB with lowest standard deviation (0.60)
3. **Statistical Significance**: All improvements are highly significant (p < 0.001)
4. **Recommendation**: Use BFL-LB or GNN-LB for production load balancing

## Data Visualization Quality
All graphs are generated in SPSS professional format with:
- Publication-ready resolution (300 DPI)
- Proper statistical annotations
- Color-coded algorithms for easy comparison
- Comprehensive error bar representations

## Author
Student - Load Balancing Research Project

## Date
April 24, 2026

## License
This project is provided as-is for educational and research purposes.
