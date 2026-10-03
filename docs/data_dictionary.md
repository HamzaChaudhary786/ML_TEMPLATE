# Data Dictionary Template

A data dictionary is a central document that describes every column in your raw and processed datasets. It prevents guesswork and ensures consistent definitions across teams.

| Column Name | Data Type | Description | Allowed Values / Range | Missing Value Strategy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | Integer | Unique identifier for each entity | 1 to ∞ | Drop before modeling | Non-predictive identifier |
| `age` | Integer | Customer age in years | 18 – 100 | Impute with median | Outliers > 100 clipped |
| `income` | Float | Annual gross income in USD | 0.0 – 5,000,000 | Impute with median | Log-transformed in pipeline |
| `credit_score` | Integer | Credit bureau score | 300 – 850 | Impute with median | Standard scaled |
| `education` | Categorical | Highest level of education | High School, Bachelor, Master, PhD | Impute with mode ("Bachelor") | One-hot encoded |
| `employment_type` | Categorical | Employment status | Full-time, Part-time, Self-employed, Unemployed | Impute with mode | One-hot encoded |
| `target` | Binary (0/1) | Whether the customer defaulted | 0 = No Default, 1 = Default | Must not contain nulls | Classification target |

---

### Audit Checklist For New Data
1. [ ] Check for missing values (`df.isnull().sum()`)
2. [ ] Check data types match this dictionary (`df.dtypes`)
3. [ ] Verify numerical column bounds (min, max, quantiles)
4. [ ] Verify categorical allowed categories (`df['col'].unique()`)
5. [ ] Check for duplicate rows (`df.duplicated().sum()`)
