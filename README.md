
---

# **10 Academy - Week 3 Challenge: Insurance Analytics**

## **Project Overview**
This repository contains the work for Week 3 of the 10 Academy Artificial Intelligence Mastery program. The project focuses on analyzing historical insurance claim data for AlphaCare Insurance Solutions (ACIS) to optimize marketing strategies and identify low-risk clients. 

Through this challenge, the goal is to enhance skills in **Data Engineering**, **Predictive Analytics**, and **Machine Learning Engineering**, simulating real-world analytics tasks.

---

## **Business Objective**
Analyze historical insurance claim data to:
- Optimize marketing strategies for car insurance in South Africa.
- Identify low-risk clients to offer reduced premiums, attracting new customers.
- Build predictive models to inform future business decisions.

---

## **Key Deliverables**
1. **Task 1: GitHub and EDA**
   - Git repository setup with version control and CI/CD workflows.
   - Exploratory Data Analysis (EDA) for data understanding and statistical insights.
2. **Task 2: Data Version Control (DVC)**
   - Implement DVC to version and manage datasets.
   - Track changes and maintain reproducibility.
3. **Task 3: A/B Hypothesis Testing**
   - Perform hypothesis testing to analyze risk and margin differences across demographics and geography.
   - Provide statistically supported business recommendations.
4. **Task 4: Statistical Modeling**
   - Build predictive models using statistical and machine learning techniques.
   - Evaluate model performance and interpret feature importance.

---

## **How to Use This Repository**
### 1. **Clone the Repository**
```bash
git clone <https://github.com/chapi1420/AlphaCare_Insurance_solutions>
cd <AlphaCare_Insurance_solutions>
```

### 2. **Branch Structure**
- `main`: Consolidated final project.
- `task-1`: EDA and initial setup.
- `task-2`: Data versioning with DVC.
- `task-3`: A/B hypothesis testing.
- `task-4`: Statistical modeling.

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Run the Analysis**
Navigate to the specific task folder and run the scripts/notebooks provided:
```bash
# For example, to run the EDA notebook:
jupyter notebook notebooks/task1_eda.ipynb
```

---

## **Folder Structure**
```
.
├── data/                 # Raw and processed datasets (excluded via .gitignore)
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Python scripts for each task
├── models/               # Saved models and evaluation reports
├── .github/              # GitHub Actions CI/CD workflows
├── .dvc/                 # DVC configuration files for dataset versioning
├── .gitignore            # Ignored files and folders
├── README.md             # Project documentation
└── requirements.txt      # Required Python libraries
```

---

## **Project Workflow**
### **Task 1: GitHub and EDA**
- Set up a GitHub repository with proper version control.
- Perform EDA to:
  - Summarize data.
  - Assess data quality.
  - Analyze statistical relationships and visualize insights.

### **Task 2: Data Version Control (DVC)**
- Install and configure DVC.
- Track datasets using DVC and manage dataset versions.

### **Task 3: A/B Hypothesis Testing**
- Test hypotheses on risk and margin differences across demographics.
- Use statistical methods (t-tests, chi-squared) to validate results.

### **Task 4: Statistical Modeling**
- Prepare data (e.g., handling missing values, feature engineering).
- Build models:
  - Linear Regression.
  - Random Forests.
  - XGBoost.
- Evaluate models and interpret feature importance using SHAP or LIME.

---

## **Key Tools and Libraries**
- **Programming Language:** Python
- **Data Manipulation and Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Statistical Analysis:** `scipy`, `statsmodels`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Version Control:** Git, GitHub
- **Data Versioning:** DVC
- **CI/CD:** GitHub Actions

---

## **Key Results**
- **EDA Insights:** Discovered patterns in premium and claims data.
- **Hypothesis Testing:** Validated or rejected hypotheses on risks and margins.
- **Modeling Outcomes:** Developed predictive models with actionable insights for marketing strategies.

---

## **Next Steps**
- Further refine models with additional data or features.
- Explore deployment pipelines for productionizing the models.
- Optimize workflows for better scalability.

---

## **Contributors**
- Nahom Temesgen Nadew (Mohan)

---

## **License**
This project is licensed under the MIT License.

---

