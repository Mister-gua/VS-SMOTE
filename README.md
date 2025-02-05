# Space-SMOTE: Synthetic Sample Generation in High-Value Spaces

 To address the issue of low-quality samples introduced by SMOTE and its improvements, we propose a new method called Space-SMOTE. Space-SMOTE treats the region containing the synthesized samples as multiple elongated subspaces and uses the middle samples to determine their spatial quality levels. Subsequently, depending on the spatial quality level, it first selects safe spaces and then selects high-value spaces among the safe spaces. By generating high-value spaces, Space-SMOTE can balance noise control and data-mining tasks.![fig1](C:\Users\JC\Documents\WeChat Files\wxid_a32zwhpc0q7522\FileStorage\File\2025-02\论文用图\论文用图\PNG\fig1.png)

*Figure 1: Comparison of space views between Borderline-SMOTE and Space-SMOTE.*

## Algorithm Introduction

**Space-SMOTE** is a novel oversampling method designed to address class imbalance in machine learning. Unlike traditional SMOTE and its variants, Space-SMOTE focuses on generating synthetic samples in **high-value spaces** while avoiding noise. Key innovations include:

1. **Subspace Evaluation**:  
   The synthetic sample space is divided into multiple independent subspaces (long strips), each evaluated for quality using **middle samples** (Fig. 2). ![fig2](C:\Users\JC\Documents\WeChat Files\wxid_a32zwhpc0q7522\FileStorage\File\2025-02\论文用图\论文用图\PNG\fig2.png)*Figure 2: Middle samples assess spatial quality in majority-dominated regions.*

   

2. **Two-Stage Selection**:  

   - **Safe Space Selection**: Filters subspaces with minimal majority-class interference.  

   - **High-Value Space Selection**: Prioritizes subspaces near decision boundaries or intra-cluster regions for data mining.  

     

3. **Noise Control**:  
   Avoids generating samples in low-quality spaces dominated by majority classes or noise (Fig. 3).  
   ![fig3](C:\Users\JC\Documents\WeChat Files\wxid_a32zwhpc0q7522\FileStorage\File\2025-02\论文用图\论文用图\PNG\fig3.png)

   *Figure 3: Three types of generating spaces (noise, high-value, low-value).*

   

**Algorithm Workflow**:  

```python
# Pseudocode (Algorithm 1)
Input: Dataset D, imbalance ratio ir, parameters k, kmin, kmax
Output: Balanced dataset D*

1. For each minority sample, generate k connecting lines to its neighbors.
2. Create middle samples on each line and evaluate their neighborhoods.
3. Select lines where middle samples have ≥kmin and ≤kmax minority neighbors.
4. Generate synthetic samples uniformly on selected lines.
```



## Experimental Datasets

Experiments were conducted on **30 real datasets** from KEEL and **2 synthetic datasets** (Moon and Circle). Key details:

### Real Datasets (30)

| ID   | Dataset | Attributes | Samples | Majority | Minority | IR    |
| :--- | :------ | :--------- | :------ | :------- | :------- | :---- |
| D1   | ecoli1  | 7          | 336     | 259      | 77       | 3.36  |
| D2   | ecoli2  | 7          | 336     | 284      | 52       | 5.46  |
| ...  | ...     | ...        | ...     | ...      | ...      | ...   |
| D30  | yeast5  | 8          | 1484    | 1440     | 44       | 32.73 |

### Synthetic Datasets (2)

| Dataset | Parameters                              |
| :------ | :-------------------------------------- |
| Moon    | `n_samples=1000, noise=0.3`             |
| Circle  | `n_samples=1000, noise=0.2, factor=0.5` |



## Experimental Results

### Key Metrics

- **AUC** (Area Under ROC Curve) and **G-mean** (Geometric Mean of Precision/Recall) were used for evaluation.

### Performance Comparison

Space-SMOTE outperformed 9 state-of-the-art methods (e.g., SMOTE, Borderline-SMOTE, ADASYN) across 5 classifiers:

| Classifier  | AUC Improvement (%) | G-mean Improvement (%) |
| :---------- | :------------------ | :--------------------- |
| BYS         | 9.58                | 17.18                  |
| KNN         | 2.54                | 10.66                  |
| SVM         | 3.76                | 12.26                  |
| **Average** | **5.02**            | **12.24**              |



## Experimental Results

### Key Metrics

- **AUC** (Area Under ROC Curve) and **G-mean** (Geometric Mean of Precision/Recall) were used for evaluation.

### Performance Comparison

Space-SMOTE outperformed 9 state-of-the-art methods (e.g., SMOTE, Borderline-SMOTE, ADASYN) across 5 classifiers:

| Classifier  | AUC Improvement (%) | G-mean Improvement (%) |
| :---------- | :------------------ | :--------------------- |
| BYS         | 9.58                | 17.18                  |
| KNN         | 2.54                | 10.66                  |
| SVM         | 3.76                | 12.26                  |
| **Average** | **5.02**            | **12.24**              |

![fig5](C:\Users\JC\Documents\WeChat Files\wxid_a32zwhpc0q7522\FileStorage\File\2025-02\论文用图\论文用图\PNG\fig5.png)
*Figure 4: Average ranking of methods across classifiers.*



## Usage

### Installation

```bash
git clone https://github.com/xxxxxx/Space-SMOTE
pip install -r requirements.txt			##暂时还不存在这个文件
```

### Example

```python
from space_smote import SpaceSMOTE

# Initialize with parameters
smoter = SpaceSMOTE(k=5, kmin=3, kmax=5)
X_resampled, y_resampled = smoter.fit_resample(X, y)
```

### Parameters

- `k`: Number of nearest neighbors (default=5).
- `kmin`: Minimum minority neighbors for safe space (default=3).
- `kmax`: Maximum minority neighbors for high-value space (default=5).

## 





