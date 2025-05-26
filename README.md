# Parkinson-Risk-Assessment (Group 5)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-Interactive%20Computing-F37626?style=for-the-badge&logo=jupyter)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-%23ff69b4?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Data%20Visualization-3776AB?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge)

> Qing Ji, Allan Ortiz

## Project Overview

This project's end goal is to assess Parkinson's disease risk using a simple machine learning mode, training on vocal biomarkers. We will analyze the dataset (`parkinsons.data`) to identify and train patterns that distinguish characteristics in individuals with Parkinson's disease from healthy individuals. 

## What is Parkinsons disease?

Parkinson's disease was described by James Parkinson in 1817 as a "shaking palsy". Parkinson's disease is a neurodegenerative disorder of the brain that results in a loss of dopamine-producing neurons in the region called substantia nigra. Loss of this neurotransmitter dopamine causes neurons to fire randomly, leading to the symptoms of Parkinson's disease (listed below). Moreover, people with Parkinson's disease also lose norepinephrine, a chemical messenger that controls many of the body's functions.

The exact cause of Parkinson's disease is not fully understood, however, there exist several factors that seem to influence the risk, including:
- <b>Specific Genes</b>: Specific genetic changes are linked to Parkinson's disease, however, these are rare unless many family members have had Parkinson's disease.
- <b>Environmental Factors</b>: Exposure to toxins and other environmental factors may increase the risk of Parkinson's disease. For example, <a href="https://en.wikipedia.org/wiki/MPTP">MPTP</a> a substance found in many illegal drugs is a neurotoxin that can cause Parkinson's disease-like symptoms. Other examples include pesticides and well water for drinking. It is important to note that no environmental factor has proven to be the cause.

Research on Parkinson's disease has shown that the brain undergoes significant changes. These changes include:
- <b>Lewy bodies</b>: Clumps of proteins in the brain, namely Lewy bodies are associated with Parkinson's disease. Researchers believe these proteins hold an important clue to the cause.
- <b>Alpha-synuclein within the Lewy bodies</b>: Alpha-synuclein is a protein found in all Lewy bodies. Moreover, interestingly, Alpha-synuclein proteins have been found in the spinal fluid of people who later got Parkinson's disease.
- <b>Altered mitochondria</b>: Mitochondria are powerhouse factories inside cells that create a significant portion of the body's energy.  Changes in the mitochondria have been found in the brains of those with Parkinson's disease.

## Symptoms

Parkinson's symptoms may include the following:
- <b>Tremor</b>: A tremble shaking usually begins in the hands or fingers. Additionally, the tremor can begin in the foot or jaw.
- <b>Slowed movement (Bradykinesia)</b>: Parkinson's disease can slow your movement, making simple tasks significantly more difficult.
- <b>Rigid Muscles</b>: Muscles may feel tense and painful, and arm movements may be short and jerky.
- <b>Poor Posture and Balance</b>: Posture may begin to sink, and may have balance problems.
- <b>Loss of automatic movements</b>: Difficulty performing movements that are traditionally done automatically, such as blinking, smiling, etc...
- <b>Writing Changes</b>: Difficulty writing, and the writing may appear cramped and small.
- <b>Nonmotor symptoms</b>: These symptoms could include depression, anxiety, sleep problems, difficulty smelling, and problems with thinking and memory.
- <b>Voice changes</b>: Parkinson's disease can affect one voice, including a quieter voice, hoarseness, and slurred speech (What we will be looking at).

## Resources

<a href="https://www.mayoclinic.org/diseases-conditions/parkinsons-disease/symptoms-causes/syc-20376055">Mayo Clinic</a>, <a href="https://med.stanford.edu/parkinsons/introduction-PD/history.html#:~:text=First%20described%20in%201817%20by,of%20cells%20that%20produce%20dopamine.">Stanford Medicine</a>

## Starting the project

### 1. Understanding the Data

For this project, you will use the `parkinsons.data` dataset (Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig). It is important to understand the variables and the dataset.

#### Matrix column entries (attributes):
- name - ASCII subject name and recording number
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
- MDVP:Flo(Hz) - Minimum vocal fundamental frequency
- MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP - Several measures of variation in fundamental frequency
- MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
- NHR, HNR - Two measures of the ratio of noise to tonal components in the voice
- status - The health status of the subject (one) - Parkinson's, (zero) - healthy
- RPDE, D2 - Two nonlinear dynamical complexity measures
- DFA - Signal fractal scaling exponent
- spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation

### 2. Libraries / Dependencies

This project will use `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` libraries.

### 3. Exploratory Data Analysis (EDA)

<!--
The first look at your data. During this EDA process, you are trying to understand the data's characteristics, identify patterns, and uncover potential insights by examining its structure, relationships, and anomalies.

1. The first step, will be to import all the libraries previously mentioned.
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
3. We want to read the dataset using the `read_csv` function from the `pandas` library. The dataframe variable is generally `df`, which is what you will be using to call the functions below.
4. Next we want to use the following functions for EDA:
    * `.info()` provides a quick summary of the info about the data frame.
    * `.head()` shows the first few rows of the dataset.
    * `.describe()` provides high-level summaries of all the columns in the dataset. Including count, mean, std, min, max, and 25%, 50%, and 75% percentiles of the values.
    * `.hist()` creates histograms for each column of the dataset. You can play around with the figure size, by passing `figsize=(x,y)` as a parameter.
    * `.isnull().sum()` provides a summary of the number of "missing values" for each of the columns. However, there is a caveat, you must look at the columns, and verify that a null value is NOT logical, thus it IS a missing value. There are multiple ways to address null values, which will be explored later.
    * `.duplicated().sum()` provides a summary of, if any duplicate rows. This is important because duplicates can skew data and affect model accuracy.
    * `.corr()` provides a data frame summary of the computed pairwise correlation of columns in the dataset. However, this can be intimidating and also hard to understand. Thus we can also visualize in a heatmap using the `seaborn` library.
5. Using the seaborn library create a heatmap showing the pairwise correlation of all the columns in the dataset. **Hint:** This should be 3-4 lines of code using `sns.heatmap()` function.

> [!IMPORTANT]
> When completed, insert your findings, graphs, and observations here, then comment out the instructions above.

-->

#### Histogram 

![Histogram](Histogram.png)

Key features for histogram analysis in the Parkinson’s dataset include MDVP:Fo(Hz), MDVP:Jitter(%), MDVP:Shimmer, HNR, and PPE. These variables relate to vocal frequency, amplitude variation, voice clarity, and pitch irregularity—all of which are commonly affected in Parkinson’s patients.

We find that most values are lower on those variable which indicates majority of these paitants are healthy have or milder symptoms. Higher values on the X axis may indicated more severe Parkinson’s traits.

#### Correlation Heatmap

Below is the heatmap visualization of the correlation matrix across all numerical features in the dataset.

![Correlation Heatmap](correlation_heatmap.png)

#### Findings from the Heatmap:

- **Highly Correlated Features**:
  - `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:PPQ`, and `Jitter:DDP` are strongly correlated (r > 0.9), suggesting they measure similar vocal jitter properties.
  - `Shimmer` variables (`MDVP:Shimmer`, `Shimmer:APQ3`, `Shimmer:DDA`, etc.) also show strong mutual correlation, which aligns with their shared focus on amplitude variations.

- **Strong Negative Relationships**:
  - `HNR` is strongly negatively correlated with `MDVP:Jitter(%)` (~ -0.73). This supports known voice science, where more jitter reduces harmonic-to-noise ratio.
  - `spread1` negatively correlates with several frequency-based features like `MDVP:Fo(Hz)` and `MDVP:Fhi(Hz)`.

- **Key Predictive Signals (Target Variable: status)**:
  - `status` shows moderate positive correlations (around 0.30–0.36) with jitter and shimmer features like `MDVP:Jitter(%)`, `MDVP:Shimmer`, and `Shimmer:APQ3`, suggesting a link between vocal instability and Parkinson’s though these are not the strongest signals.
  - The highest correlations with `status` are found in:
   - `spread1` (0.53)
   - `spread2` (0.52)
   - `PPE` (0.45)
  - These features are likely capturing more abstract or high-level voice signal variations, making them strong candidates for classification tasks.
  -`D2` shows only a modest correlation (~0.34) with `status`, so it may still hold value, but not as strong a predictive signal as `spread1`, `spread2`, or `PPE`.

- **Why This Makes Sense**:
  - Many of the high correlations are expected, as jitter and shimmer are sub-measures of frequency and amplitude instability in voice; a key indicator in Parkinson’s patients.
  - These insights will be useful for feature selection or dimensionality reduction in our ML model later on.

### 4. Support Vector Machine (SVM)

<!---
A Support Vector Machine (SVM) is a supervised machine learning algorithm that classifies data by identifying the optimal hyperplane that maximizes the margin between different classes in an N-dimensional space. For the Machine Learning model, use an SVM. To explore more about SVM and its applications, check out this <a href="https://www.ibm.com/think/topics/support-vector-machine#:~:text=A%20support%20vector%20machine%20(SVM,the%201990s%20by%20Vladimir%20N.">resource</a>.

Before getting started, import the necessary libraries for splitting data, normalizing features, and training a Support Vector Machine (SVM) model."
```
from sklearn.model_selection import train_test_split # Divides the dataset into training and test sets
from sklearn.preprocessing import StandardScaler # Standardizes numerical features to have a mean of 0 and an SD of 1.
from sklearn.svm import SVC  # Imports Support # Init a SVM for classification
```
1. The first step is to drop any rows with missing values. To do this we can use the `dropna()` function calling on the `df` variable that was created in the EDA. However, as we previously saw in the EDA, there are no missing values (It is good practice to drop null values though).

> [!IMPORTANT]
> When completed, Insert a code block that combines the 1st and 2nd step code. Above the code block add a header "Dropping rows with missing values & Extract features and target variables", then comment out the instructions above.

3. The next step is to extract features and target variables.
   * Create a new data frame containing only the features relevant to predicting Parkinson’s. How can you ensure it excludes unnecessary columns?
      * We want to drop specific columns that are either irrelevant or can cause data leakage. Think about which features are useful for prediction. Are there columns that provide no relevant information or could unintentionally give away the outcome? Discuss as a group and justify your choices.
         * **Hint 1:** One column may not contribute to meaningful predictions.
         * **Hint 2:** Another column could unintentionally reveal the correct answer, making the model less reliable (data leakage).
      * Look up the <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html">drop()</a> function in pandas documentation. How can you use it to remove multiple columns at once?
   * In regards to the target variable (`y`, or a name of your choice), what is our end goal in the model? What are we trying to predict? And what column reflects that in the dataset?

Once you've identified the necessary features, write the code to drop irrelevant columns and define `X` and `y`. Before moving forward, check with your PD to confirm your choices.

> [!IMPORTANT]
> When completed, Insert a code block that combines the 1st and 2nd step code. Above the code block add a header "Dropping rows with missing values & Extract features and target variables", then comment out the instructions above.

-->
#### Dropping rows with missing values & Extract features and target variables

```python
# Dropping rows with missing values & Extract features and target variables

# Step 1: Drop null values (none found, but good practice)
df = df.dropna()

# Step 2: Drop non-predictive columns and define X (features) and y (label)
X = df.drop(['name', 'status'], axis=1)
y = df['status']
```

<!---

3. To evaluate how well our model generalizes to unseen data, we need to split the dataset into training and testing sets, following the 80/20 rule. Meaning: 80% of the data will be used for training, and 20% for testing.
   * Use the train_test_split() function from sklearn.model_selection to randomly divide `X` (features) and `y` (labels).
   * The `train_test_split()` function from `sklearn.model_selection` randomly splits `X` and `y` into training and testing sets. The parameters are in the following order:
       * `X` (features): Input for data prediction.
       * `y` (labels): target variable.
       * `test_size =`: the proportion of the dataset that should be allocated to the test set (e.g. `0.3` for 30%).
       * `random_state =`: effectively behaves as a seed to ensure reproducibility.
   * During this process features (`X`) and labels (`y`) must remain properly aligned after splitting. In other words, for every row in `X_train`, there is a corresponding label in `y_train` that indicates whether that person has Parkinson's or not. How can you use the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">train_test_split()</a> function to do this?

4. SVMs are sensitive to feature dominance because they rely on finding optimal lines/hyperplanes. Without scaling, features with larger values could disproportionately influence the model. Standardizing ensures all features contribute equally. How can you use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScalar()</a> to scale the training and test data separately? In this step, simply initialize the StandardScalar.

5. After initializing the scaler, how do we ensure that both training and test sets follow the same transformation? Look at `fit_transform()` and `transform()` in following <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">documentation</a>. Why should we apply `fit_transform()` to the training set but only `transform()` to the test set?
    * Store the transformed training and test features in new variables (e.g. `X_scaled_train` and `X_scaled_test`). Feel free to choose other meaningful names.

> [!IMPORTANT]
> When completed, Insert a code block that combines the 3rd, 4th, and 5th step code. Above the code block add a header "Splitting Data into train and test sets & scaling and fitting the data", then comment out the instructions above.

--->

#### Splitting Data into train and test sets & scaling and fitting the data

```python
# Splitting Data into train and test sets & scaling and fitting the data

# Step 3: Split data into training and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Step 4: Initialize StandardScaler to normalize the feature data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Step 5: Fit on training data and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

<!---
6. A linear kernel is useful when the data is linearly separable, meaning a straight-line decision boundary can effectively classify the data. You may also want to experiment with other kernels, such as 'rbf' or 'poly' to see how they impact performance.

```
m = SVC(kernel = 'linear')
```

7. After defining our SVM model, the next step is to train it on the standardized training data. Training allows the model to learn patterns and relationships between the features `X_scaled_train` and the labels `y_train`. How could you use the `.fit()` function to do this? In this case, the `.fit()` function is allowing the model to learn the optimal decision boundary.

> [!IMPORTANT]
> Now, put everything together! Create a code block that initializes, trains, and tests the model. Above the code block, add the header 'Initializing, Training, and Testing the SVM', then comment out the instructions above.
--->
<!---
Initializing, Training, and Testing the SVM 

8. After training, the model can be used to classify new data. The `.predict()` function allows us to generate predictions on our test dataset `X_scaled_test`. How could we ust the `.predict()` function and save it to a new variable named `y_pred`? Additionally, The `accuracy_score()` from the submodule `sklearn.metrics` function compares the model's predictions `y_pred` to the actual labels `y_test` to determine how well our model performs.

```
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

### 5. Confusion Matrix

1. Import the necessary submodule from `sklearn.metrics`:
```
from sklearn.metrics import confusion_matrix
```
2. Compute the confusion matrix: Use `confusion_matrix(y_pred, y_test)` function from `sklearn.metrics`.
3. Visualize with Seaborn using `sns.heatmap()` to create a heatmap with labels, colors, and annotations. Two parameters you might need are:
      * `annot=` is a boolean value for displayed values in the heatmap
      * `cmap=` takes in a string of a color map to adjust the color map of the heatmap (e.g. `"Blues"`, `"coolwarm"`, `"BrBG"`, etc...).


> [!IMPORTANT]
> Now, the last step! Create a code block showing how you completed setp 8, with the header "SVM Accuracy". Additionally, include a photo of the Confusion Matrix, and a paragraph analysis of the Confusion Matrix. Lastly, comment out the instructions above.
--->

#### SVM Accuracy
```python
# Step 6: Import model and accuracy evaluation tools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 7: Initialize SVM with linear kernel
model = SVC(kernel='linear')

# Step 8: Train the model
model.fit(x_train_scaled, y_train)

# Step 9: Predict on test set
y_pred = model.predict(x_test_scaled)

# Step 10: Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

Model Accuracy: 0.87
```
### 5. Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

#### Analysis

The confusion matrix reveals that the model is generally effective at identifying Parkinson's disease cases, correctly classifying 30 individuals with the condition and missing only 2 (false negatives). This suggests high sensitivity, meaning the model is adept at detecting true positive cases. On the other hand, among healthy individuals, only 4 were correctly identified, while 3 were incorrectly classified as having Parkinson’s (false positives). This indicates a slight tendency to over-predict the presence of the disease in healthy individuals, which could lead to unnecessary concern or further testing. While the model shows strong overall performance, refining its ability to distinguish healthy individuals more accurately would enhance its clinical reliability.
