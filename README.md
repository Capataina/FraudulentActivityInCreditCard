# Credit Card Fraud Detection

## Overview
This project focuses on the detection of fraudulent credit card transactions using machine learning. The dataset is heavily imbalanced, with fraudulent transactions making up only a very small fraction of the total. The aim was to build a model that could identify fraud cases effectively while keeping false positives at a reasonable level.

## Dataset
- **Source**: Kaggle Credit Card Fraud Detection dataset (~285,000 transactions)  
- **Features**: 30 anonymised PCA-transformed features (`V1`–`V28`), along with `Time` and `Amount`  
- **Target**: Binary classification → Fraud (`1`) vs Normal (`0`)  
- **Imbalance**: ~0.17% fraudulent transactions  

## Technologies Used
- **Python**  
- **XGBoost** for model training  
- **Scikit-learn** for preprocessing, evaluation and baselines  
- **UMAP, PCA, t-SNE** for dimensionality reduction and visualisation  
- **Plotly** for interactive graphs  
- **Pandas / NumPy** for data handling  

## Approach
1. **Data Preprocessing**  
   - Split into training and testing sets  
   - Addressed imbalance using `scale_pos_weight` within XGBoost  

2. **Modelling**  
   - Trained an **XGBoost classifier** and tuned key hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, etc.)  
   - Evaluated using accuracy, precision, recall, F1, ROC AUC and PR AUC  

3. **Visualisation**  
   - Used UMAP, PCA and t-SNE to project high-dimensional data into 2D  
   - Displayed how fraud cases cluster separately from normal activity  
   - Ran **error analysis plots** to highlight false positives and false negatives  

4. **Evaluation Metrics**  
   - Accuracy: ~99% (not a useful metric here due to class imbalance)  
   - Fraud detection:  
     - Precision: ~94%  
     - Recall: ~84%  
     - F1 Score: ~89%  
   - ROC-AUC: ~0.97  
   - PR-AUC: ~0.86  

## Visual Results
- Fraud cases formed **small, distinct clusters** in reduced 2D space  
- XGBoost proved effective at separating these clusters  
- Error analysis showed that **missed fraud cases (false negatives)** were generally located in areas that overlap heavily with normal cases  

## Conclusion

For this project I decided to go with XGBoost as it is particularly good for tabular datasets like this one where each row and each column represents one thing with no inconsistencies. Unlike deep learning or neuroevolution methods, which can be better for images, text or inconsistent datasets with varying data types, ranges and points; XGBoost can handle smaller feature sets efficiently and gives excellent performance on row/column based imbalanced data. It also trains much faster and is easier to tune compared to more complex approaches.

The results were strong: the model reached over 99 percent overall accuracy, though this is easy to reach as the dataset is heavily skewed towards non-fraudulent activity, but with fraud detection performance at around 94 percent precision and 84 percent recall with a 89 percent F1 score which is the main target. This means it was able to correctly identify most fraudulent transactions while keeping false positives relatively low. 

Through dimensionality reduction with UMAP, PCA and t-SNE, we also saw that fraud cases cluster in small, distinct regions rather than being dispersed equally throughout the dataset, which helps explain why XGBoost was effective. 

Finally, the error analysis showed that the few missed frauds tend to be cases that are hard to separate from normal transactions, suggesting that these would be challenging for any model.

Overall, the project shows that XGBoost is both a practical and reliable choice for credit card fraud detection, balancing accuracy, interpretability and training efficiency.
