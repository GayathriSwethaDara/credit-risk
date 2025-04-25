# credit-risk
This project addresses the critical challenge of credit risk assessment using machine learning. By analyzing a dataset of 999 loan applicants, we developed a predictive model to classify applicants as either "Low Risk" (likely to repay) or "High Risk" (likely to default). The goal was to automate and enhance the accuracy of credit evaluations while ensuring transparency and fairness.
## Methodology
The dataset included features such as age, employment type, credit amount, loan duration, savings, and loan purpose. To create the target variable (Risk), we defined logical thresholds:
**Low Risk:** Applicants with loans ≤ €5,000, repayment durations ≤ 24 months, and substantial savings ("quite rich" or "rich").
**High Risk:** Applicants with loans > €8,000, durations > 48 months, or minimal savings ("little" or "none").

Missing values in categorical fields (e.g., "Saving accounts") were replaced with "unknown" to preserve data integrity. Two engineered features improved model performance:
**Monthly Payment:** Credit amount divided by loan duration.
**Age Groups:** Binned into 18-30, 30-45, 45-60, and 60+ to capture life-stage financial behaviors.
![Screenshot 2025-04-25 221039](https://github.com/user-attachments/assets/bbd9264d-89ec-4d56-87ee-ea4b08731cf9)
![Screenshot 2025-04-25 220853](https://github.com/user-attachments/assets/576360ba-0487-4587-a5e1-d648cb37a659)
## Model Selection and Training
A **Random Forest classifier** was chosen for its ability to handle mixed data types (numeric and categorical), robustness to outliers, and native support for feature importance analysis. To address class imbalance (85% Low Risk vs. 15% High Risk), we applied SMOTE (Synthetic Minority Oversampling Technique), which generated synthetic samples for the minority class.
The model achieved 89% accuracy on the test set, with strong performance across metrics:
**High Risk (Class 0):** 78% precision, 65% recall.
**Low Risk (Class 1):** 91% precision, 95% recall.
The disparity in recall for High Risk cases highlights the challenge of identifying defaulters, a common issue in imbalanced datasets.
![Screenshot 2025-04-25 220927](https://github.com/user-attachments/assets/e1d52b1f-9ca2-4f30-bd48-144ac7720554)
## Conclusion
This project demonstrates the viability of machine learning in credit risk assessment, achieving 89% accuracy while identifying key risk factors. The Random Forest model’s interpretability and robustness make it suitable for initial deployments, though future work should focus on fairness and dynamic risk modeling. By integrating real-time economic data and explainability tools, the system can evolve into a scalable, ethical solution for financial institutions.

