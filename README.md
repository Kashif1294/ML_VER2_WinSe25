## VER: Human vs Object – End‑to‑End ML Pipeline

This project develops a complete machine‑learning pipeline to distinguish **humans** from **objects** using measurements from a vertically oriented ultrasonic sensor.  
All stages of the workflow – from raw data ingestion through model training and comparative evaluation – are implemented in the notebook `VER_pipeline.ipynb`.

The pipeline:

- Ingests two large raw datasets (human and object recordings).
- Applies a project‑specific header schema to the leading metadata columns.
- Constructs a binary classification target (0 = object, 1 = human).
- Derives a high‑dimensional feature representation of the ultrasonic signal.
- Trains and evaluates a set of classical, convolutional, and transformer‑based models.
- Summarises performance using accuracy, macro F1‑score, and visual comparisons.

The following sections describe the main components in an academic and structured way.

---

## 1. Data Sources and Header Structure

### 1.1 Raw data

Two primary CSV files are used:

- `Dataset/human_5000.csv` – ultrasonic returns corresponding to human targets.
- `Dataset/object_5000.csv` – ultrasonic returns corresponding to object targets.

Each file is **wide**: one row corresponds to a single acquisition, and the large number of columns represent discrete samples of the ultrasonic signal (and some metadata). Conceptually, a 1D time series is stored as many adjacent feature columns.

### 1.2 Header schema

The file `Project_Docs/Header structure Winter 2021 (1).xlsx` contains the intended header structure for the **first 17 columns** of the combined data. The pipeline:

1. Loads the Excel file and reads the first row as a list of column names.
2. Renames only the first *N* columns of the combined DataFrame, where *N* is the smaller of:
   - The number of columns in the header file.
   - The number of columns present in the DataFrame.
3. Leaves all remaining columns unchanged to preserve the full signal information.

After this step, the combined dataset (`combined_df`) has:

- Human‑readable names for the leading metadata columns.
- A large set of unnamed numeric columns corresponding to the ultrasonic signal samples.

---

## 2. Binary Label Construction (0 vs 1)

The study is formulated as a binary classification problem. The aim is to decide whether a given measurement corresponds to a **human** or an **object**.

To this end:

- The human and object datasets are concatenated row‑wise.
- **Column 3 (zero‑based index 2)** is reserved as the **class label**.
- Labels are assigned positionally:
  - Rows originating from the human dataset receive label `1`.
  - Rows originating from the object dataset receive label `0`.

From this point onward, column 3 is treated as the ground‑truth target for all models.

---

## 3. Feature Selection and Construction

### 3.1 Dropping non‑informative columns

The raw CSVs include a small set of leading columns that act as indices or metadata and are not useful as input features. The pipeline removes these by **positional index**:

- Columns with indices `0`, `1`, and `3–17` (inclusive) are dropped.
- The label column at index `2` is retained.
- All remaining columns, which predominantly encode the ultrasonic signal, are kept.

This yields a streamlined DataFrame `df_model` in which:

- Column index `2` still stores the binary class label.
- All other columns represent candidate input features.

### 3.2 Definition of features and labels

Based on `df_model`, the following notation is used:

- `y` – target vector:
  - Extracted from column index `2`.
  - Converted to integer type, with values `0` (object) and `1` (human).
- `X` – feature matrix:
  - All columns of `df_model` except the label column.
  - Cast to `float32` to reduce memory footprint and to standardise the input type for downstream models.

The resulting feature space is high‑dimensional, reflecting many sampled points of the ultrasonic return signal.

---

## 4. Train/Test Split and Normalisation

The dataset is partitioned into training and test subsets to enable unbiased evaluation of all models.

- A single split is created using `train_test_split` with:
  - `test_size = 0.2` (80% training, 20% testing).
  - `random_state = 42` to ensure reproducibility.
  - Stratification when feasible:
    - If the minority class has fewer than two examples, stratification is disabled to avoid errors and a simple random split is used instead.

Feature scaling is then applied:

- A `StandardScaler` is fit on `X_train` and applied to both `X_train` and `X_test`.
- The scaled matrices `X_train_scaled` and `X_test_scaled` are passed to all models.

For later comparison, two dictionaries are maintained:

- `model_accuracies` – test accuracy for each model.
- `model_macro_f1` – macro F1‑score for each model (i.e., unweighted mean of per‑class F1‑scores).

---

## 5. Model Architectures

All models operate on the same train/test split and the same scaled feature representations. Deep learning models treat each sample as a one‑dimensional sequence of length equal to the number of features.

### 5.1 Linear SVM (SGDClassifier)

- Implemented via `SGDClassifier` with hinge loss, corresponding to a linear support vector machine.
- Scales to large feature spaces and sample sizes more efficiently than kernel‑based SVMs.
- Serves as a strong linear baseline.

### 5.2 Random Forest

- Standard `RandomForestClassifier` with a moderate number of trees.
- Captures non‑linear relationships without explicit feature engineering.
- Is typically robust and computationally efficient on tabular data.

### 5.3 1D Convolutional Neural Network (CNN)

- Input is reshaped to `(batch_size, sequence_length, 1)`.
- The architecture comprises:
  - One or more `Conv1D` layers with ReLU activation to capture local patterns in the signal.
  - A `GlobalAveragePooling1D` layer to aggregate temporal information.
  - Fully connected layers leading to a final sigmoid neuron for binary classification.
- Trained with `binary_crossentropy` loss and the `adam` optimiser for a fixed number of epochs.

### 5.4 Temporal Convolutional Network (TCN‑style)

- Also operates on sequences of shape `(batch_size, sequence_length, 1)`.
- Uses several **dilated causal `Conv1D` layers**:
  - Causal padding ensures that the output at a given time step only depends on current and past inputs.
  - Dilation factors (e.g., 1, 2, 4, 8) expand the receptive field without extremely large kernels.
- A `GlobalAveragePooling1D` layer and dense layers follow, culminating in a sigmoid output.
- Trained with the same loss and optimiser as the CNN.

### 5.5 Conv‑Transformer Hybrid

- Starts with `Conv1D` layers to project the raw sequence into a higher‑dimensional representation (`d_model`).
- Applies a **Multi‑Head Self‑Attention** layer:
  - Allows each time step to attend to all others, modelling long‑range dependencies in the ultrasonic signal.
- A global pooling layer and fully connected layers map the attention output to a binary prediction.
- This architecture combines local feature extraction (convolutions) with global context modelling (attention).

### 5.6 Time Series Transformer

- Designed as a more canonical transformer for time‑series data.
- Steps:
  1. A `Conv1D` layer projects the input to `d_model` channels.
  2. A **learned positional embedding** is added so the model encodes temporal order.
  3. A transformer encoder block is applied:
     - Multi‑Head Self‑Attention.
     - Residual connections and `LayerNormalization`.
     - Position‑wise feed‑forward network.
  4. A `GlobalAveragePooling1D` layer aggregates the sequence into a fixed‑size representation.
  5. Final dense layers and a sigmoid neuron perform the binary classification.

All deep models use mini‑batch training with a fixed number of epochs and a validation split, providing both training progress and validation metrics.

---

## 6. Evaluation Methodology

Each model is evaluated on the held‑out test set using a consistent set of metrics:

1. **Predicted labels** are obtained on `X_test_scaled`.
2. The following quantities are computed:
   - **Accuracy** – proportion of correctly classified samples.
   - **Macro F1‑score** – average of F1‑scores for the two classes (object and human), treating them symmetrically.
   - **Confusion matrix** – 2×2 table summarising true/false positives and negatives.
   - **Classification report** – per‑class precision, recall, and F1‑score.
3. Accuracy and macro F1‑score are stored in the `model_accuracies` and `model_macro_f1` dictionaries.

This procedure provides both high‑level performance indicators and a more fine‑grained view of errors and class‑specific behaviour.

---

## 7. Comparative Summary and Visualisation

The final section of the notebook aggregates results across all models:

- A **summary DataFrame** is constructed with one row per model, containing:
  - Model name.
  - Test accuracy.
  - Macro F1‑score.
- The best‑performing model is identified according to:
  - Highest accuracy.
  - Highest macro F1‑score.
- A set of **bar charts** is generated:
  - Accuracy vs. model.
  - Macro F1 vs. model.

These summaries make it straightforward to assess:

- Whether more complex deep models (CNN, TCN, transformer‑based) offer a measurable benefit over simpler baselines (linear SVM, Random Forest).
- How consistent the ranking is across accuracy and macro F1‑score.

---

## 8. How to Execute the Pipeline

To reproduce the experiments:

1. Ensure the Python environment provides at least:
   - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow` (with Keras), and `openpyxl`.
2. Open `VER_pipeline.ipynb` in JupyterLab, Jupyter Notebook, VS Code, or Cursor.
3. Execute the cells sequentially under the section  
   **“VER Human+Object Preparation Pipeline (Dual‑Class)”**, covering:
   - Data loading, concatenation, and header application.
   - Column selection and construction of `X` and `y`.
   - Train/test split and feature scaling.
   - Training and evaluation of:
     - Linear SVM (SGDClassifier),
     - Random Forest,
     - CNN,
     - TCN,
     - Conv‑Transformer Hybrid,
     - Time Series Transformer.
   - The final summary and visualisation cell.

Upon completion, the notebook provides:

- Confusion matrices and classification reports for each model.
- A consolidated table of accuracy and macro F1‑scores.
- Visual comparisons that highlight which architectures are most suitable for human vs object discrimination with the given ultrasonic sensor data.


