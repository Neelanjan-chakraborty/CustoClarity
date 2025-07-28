# CUSTO CLARITY 🛍️📊
## Customer Segmentation Analysis for Retail Strategy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### 📖 Project Overview

**CUSTO CLARITY** is a comprehensive data science project focused on customer segmentation analysis using machine learning clustering techniques. This project analyzes retail customer data to identify distinct customer segments that can guide marketing strategies and product development decisions.

### 🎯 Objectives

- **Customer Segmentation**: Identify distinct customer groups based on purchasing behavior
- **Pattern Discovery**: Uncover hidden patterns in customer data through EDA
- **Marketing Insights**: Provide actionable insights for targeted marketing campaigns
- **Strategic Planning**: Guide product strategy and customer retention efforts

### 🔬 Methodology

1. **Exploratory Data Analysis (EDA)**
   - Statistical analysis of customer demographics and spending patterns
   - Data quality assessment and outlier detection
   - Correlation analysis and feature relationships

2. **Data Preprocessing**
   - Missing value treatment
   - Feature scaling and normalization
   - Feature engineering and selection

3. **Dimensionality Reduction**
   - Principal Component Analysis (PCA)
   - t-Distributed Stochastic Neighbor Embedding (t-SNE)

4. **Clustering Analysis**
   - K-Means clustering with optimal cluster determination
   - DBSCAN for density-based clustering
   - Cluster validation and interpretation

5. **Visualization & Insights**
   - Interactive cluster visualizations
   - Customer segment profiling
   - Business recommendations

### 📊 Dataset

The project utilizes the **Mall Customer Segmentation Dataset**, which includes:
- Customer demographics (Age, Gender)
- Annual Income information
- Spending Score metrics
- 200 customer records for analysis

### 🛠️ Technologies Used

- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib**: Statistical plotting
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive visualizations
- **Jupyter Notebook**: Interactive development environment

### 📁 Project Structure

```
CUSTO CLARITY/
├── data/                          # Raw and processed datasets
│   ├── raw/                       # Original dataset files
│   └── processed/                 # Cleaned and transformed data
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Initial data exploration
│   ├── 02_preprocessing.ipynb     # Data cleaning and preparation
│   ├── 03_clustering_analysis.ipynb # Clustering implementation
│   └── 04_insights_visualization.ipynb # Results and insights
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessor.py           # Data preprocessing functions
│   ├── clustering.py             # Clustering algorithms
│   └── visualizer.py             # Visualization functions
├── outputs/                       # Generated outputs
│   ├── figures/                   # Plots and visualizations
│   ├── models/                    # Trained model files
│   └── reports/                   # Analysis reports
├── docs/                          # Documentation
│   └── methodology.md             # Detailed methodology
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # Project documentation
```

### 🚀 Getting Started

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Custo Clarity"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv custo_clarity_env
   custo_clarity_env\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - The Mall Customer Segmentation dataset will be automatically downloaded
   - Or manually place the dataset in `data/raw/` directory

#### Usage

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Run the analysis notebooks in sequence:**
   - `01_data_exploration.ipynb`: Initial data exploration
   - `02_preprocessing.ipynb`: Data cleaning and preparation
   - `03_clustering_analysis.ipynb`: Clustering implementation
   - `04_insights_visualization.ipynb`: Results and insights

3. **Generate reports**
   ```bash
   python src/generate_report.py
   ```

### 📈 Key Findings

*(Results will be populated after analysis)*

- **Customer Segments Identified**: X distinct customer groups
- **Primary Segmentation Factors**: Income level, spending behavior, age demographics
- **Marketing Recommendations**: Targeted strategies for each segment
- **Business Impact**: Potential revenue optimization opportunities

### 📊 Visualizations

The project generates various visualization types:
- **Cluster Scatter Plots**: 2D and 3D cluster visualizations
- **Demographic Analysis**: Age and income distribution by segment
- **Spending Patterns**: Purchase behavior analysis
- **Business Dashboards**: Interactive insights for stakeholders

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👨‍💻 Author

**Neelanjan Chakraborty**
- 🌐 Website: [neelanjanchakraborty.in](https://neelanjanchakraborty.in/)
- 💼 LinkedIn: [linkedin.com/in/neelanjanchakraborty](https://linkedin.com/in/neelanjanchakraborty)
- 🐙 GitHub: [github.com/Neelanjan-chakraborty](https://github.com/Neelanjan-chakraborty)
- 📧 Location: Kharagpur, West Bengal
- 📱 Phone: +91 8617352997

**Professional Background:**
- Junior Full Stack Developer specializing in ML & Data Science
- Experience in Generative AI and 3D Character Animation
- Certified in Oracle Cloud Infrastructure AI, Databricks, and multiple ML platforms
- Expertise in Python, TensorFlow, Scikit-learn, and data visualization tools

### 🙏 Acknowledgments

- Mall Customer Segmentation Dataset contributors
- Scikit-learn community for excellent ML tools
- Seaborn and Matplotlib for visualization capabilities
- Jupyter Project for interactive development environment

### 📞 Support

For questions, suggestions, or collaboration opportunities:
- 📧 Contact via website: [neelanjanchakraborty.in](https://neelanjanchakraborty.in/)
- 💬 Open an issue in this repository
- 🔗 Connect on LinkedIn for professional discussions

---

*"Transforming customer data into actionable business insights through advanced analytics and machine learning."*

**Created with ❤️ by Neelanjan Chakraborty | 2025**
