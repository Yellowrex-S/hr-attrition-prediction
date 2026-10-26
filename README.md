# HR Analytics: Job Satisfaction and Employee Attrition

A two-stage machine learning project from the Yandex Practicum Data Science program. The notebook is written in Russian; this README summarizes it in English.

## Business task

HR analysts at the company «Работа с заботой» help businesses avoid financial losses and staff turnover. Two models were needed:

1. **Regression:** predict an employee's job satisfaction level from company data.
2. **Classification:** predict whether an employee will quit, using the predicted satisfaction as one of the features.

## What I did

1. **Data preparation and EDA** for both tasks: missing values, anomalies, distributions, portraits of employees who quit.
2. **Correlation analysis** with φK.
3. **Pipelines** with `ColumnTransformer` (imputation, scaling, encoding) and hyperparameter search over several models for each task.
4. **Stage two uses stage one:** the satisfaction predicted by the first model became a feature for the attrition model.
5. **Interpretation** of feature importance with SHAP.

## Results

| Task | Best model | Metric | Cross-validation | Test |
|---|---|---|---|---|
| Job satisfaction | Decision Tree | SMAPE, % | 15.69 | **14.76** |
| Attrition | Gradient Boosting | ROC-AUC | 0.942 | **0.929** |

- The attrition model reaches accuracy 0.91 on the test set (2,000 employees).
- An interesting finding for future research: job satisfaction is strongly linked to the performance rating given by the manager. It is worth checking whether employees know their ratings and what the ratings give them.

## Tech stack

Python · pandas · NumPy · Matplotlib · Seaborn · SciPy · phik · scikit-learn · category_encoders · SHAP

## How to run

```bash
pip install -r requirements.txt
jupyter notebook hr_attrition_prediction.ipynb
```

The notebook downloads the data from public links, so no local files are needed.

---

**Кратко по-русски.** Две связанные модели для HR-аналитики (учебный проект Яндекс Практикума): прогноз удовлетворённости сотрудника (дерево решений, SMAPE 14,76 на тесте) и прогноз увольнения, где предсказанная удовлетворённость стала признаком (градиентный бустинг, ROC-AUC 0,929 на тесте). Важность признаков объяснена через SHAP.
