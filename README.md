# hermes-project
# Project Hermes
## Automated Data Augmentation Selection for Time Series Anomaly Detection

# Overview
Project Hermes is an open-source system designed to automate the selection and evaluation of data augmentation strategies for time series anomaly detection tasks. Drawing inspiration from approaches in computer vision and AutoML, Hermes enables users to systematically benchmark augmentation policies on multiple datasets using rigorous statistical testing.

# Key Features
Automated Comparative Benchmarking
Evaluates multiple data augmentation policies (e.g., jitter, scaling, time-warping, and combinations) on standard time series anomaly detection datasets.

# Statistical Significance Testing
Employs k-fold cross-validation and paired t-tests to rigorously measure the impact of each augmentation technique versus baseline performance.

# Dataset-Agnostic
Out-of-the-box support for ENERGY, KPI, and YAHOO anomaly detection datasets—easy to extend to other time series data.

# Reproducible Research
All experiments, metrics, and results are logged with clear scripts and documentation.

# Motivation
Data augmentation is widely believed to enhance machine learning models, especially in computer vision. However, our results with Project Hermes show that, for time series anomaly detection, commonly used augmentation policies may offer no benefit or can even degrade performance compared to a simple, un augmented baseline. This finding highlights the importance of automated, data driven assessment rather than relying on intuition or domain transfer from other fields.

# Main Results
Across ENERGY, KPI, and YAHOO datasets, baseline (no augmentation) consistently achieved the highest accuracy, outperforming more complex augmentation strategies.

Paired t-test analysis (p > 0.05) indicated no statistically significant improvement from any tested augmentation technique.

For detailed results, see: johanabhishek.vercel.app

# Installation
shell
git clone https://github.com/johanabhishek/hermes-project
cd hermes
pip install -r requirements.txt
Usage
shell
python run_experiments.py --> config configs/experiment_config.yaml
Modify the provided configuration YAML to specify datasets and augmentation policies.

Results and evaluation metrics will be saved in the results/ directory.

Project Structure
text
hermes/
│
├── data/                   # Dataset loaders and utilities
├── augmentation/           # Augmentation policy implementations
├── experiments/            # Experiment setup, cross-validation, t-test code
├── configs/                # YAML config files for experiments
├── results/                # Output folder for logs and results
├── run_experiments.py      # Main experiment launcher
└── README.md               # This file
Contributing
Contributions are welcome! Please open issues or pull requests for bugfixes, improvements, or new datasets/policies.

License
MIT License

Citation
If you use Project Hermes in your research, please cite:

text
@misc{hermes2025,
  author = {Dasari, Johan Abhishek},
  title = {Project Hermes: Automated Data Augmentation Selection for Time Series Anomaly Detection},
  year = {2025},
  url = {https://github.com/yourusername/hermes}
}
Author
Johan Abhishek Dasari
johanabhishek9@gmail.com
Nellore, Andhra Pradesh, India

