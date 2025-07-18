# นำเข้าไลบรารีที่จำเป็นทั้งหมด
import pandas as pd
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
import io
from contextlib import redirect_stdout
import time # สำหรับจับเวลาการเทรนและสร้าง timestamp
import json # สำหรับบันทึกผลลัพธ์ในรูปแบบ JSON
import matplotlib.pyplot as plt # สำหรับพลอต Confusion Matrix
import seaborn as sns # สำหรับพลอต Confusion Matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix
import pandas.api.types as ptypes


def check_gpu():
    """Checks and prints GPU information if available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print("GPU is available and being used by TensorFlow.")
        except RuntimeError as e:
            print(e)
            print("Error setting memory growth for GPU.")
        except Exception as e:
            print(f"An unexpected error occurred during GPU check: {e}")
            print("No GPU available, using CPU.")
    else:
        print("No GPU available, using CPU.")



class preprocess:
    def __init__(self, dataframe, columns):
        self.data = dataframe.copy()
        self.columns = columns
        self.sequence_length = 5

    def set_sequence_length(self, seq_length):
         self.sequence_length = seq_length
         print(f"Sequence length set to: {self.sequence_length}")

    def select_columns(self):
        missing_cols = [col for col in self.columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        self.data = self.data[self.columns].copy()
        return self.data

    def label_encoding(self, mapping_path='category_mapping.json'):
        mapping = {}
        for column in self.data.columns:
            if self.data[column].dtype == 'object' or ptypes.is_categorical_dtype(self.data[column].dtype):
                 categories = self.data[column].astype('category')
                 self.data[column] = categories.cat.codes
                 mapping[column] = {int(i): str(cat) for i, cat in enumerate(categories.cat.categories)}
        mapping_dir = os.path.dirname(mapping_path)
        if mapping_dir and not os.path.exists(mapping_dir):
             os.makedirs(mapping_dir, exist_ok=True)
        try:
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=4, ensure_ascii=False)
            print(f"Saved category mapping to {mapping_path}")
        except Exception as e:
             print(f"Warning: Could not save category mapping to {mapping_path}. Error: {e}")
        return self.data, mapping

    def scale_data(self, scaler, exclude_columns=None, save_path='assets/scaler.pkl'):
        if exclude_columns is None:
            exclude_columns = []
        columns = self.data.columns
        cols_to_scale = [col for col in columns if col not in exclude_columns]
        scaled_df = self.data.copy()
        if cols_to_scale:
            try:
                cols_exist = [col for col in cols_to_scale if col in scaled_df.columns]
                if cols_exist:
                     scaled_df[cols_exist] = scaler.fit_transform(self.data[cols_exist])
                else:
                     print("Warning: No columns found to apply scaling.")
                scaler_dir = os.path.dirname(save_path)
                if scaler_dir and not os.path.exists(scaler_dir):
                    os.makedirs(scaler_dir, exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"Saved scaler to {save_path}")
            except Exception as e:
                 print(f"Warning: Could not scale data or save scaler. Error: {e}")
                 return self.data
        self.data = scaled_df
        return self.data

    def create_sequence(self, label_column_name, seq_length=None):
        if seq_length is None:
            seq_length = self.sequence_length
        if 'id' not in self.data.columns:
            raise ValueError("Column 'id' not found in the dataframe. It is required for grouping sequences.")
        store_id = self.data['id'].unique()
        all_X = []
        all_y = []
        try:
            features = self.data.columns.tolist()
            label_idx = features.index(label_column_name)
        except ValueError:
            print(f"Error: Label column '{label_column_name}' not found in the dataframe columns: {self.data.columns.tolist()}")
            expected_n_features = len(self.data.columns) - 2 if 'id' in self.data.columns else len(self.data.columns) - 1
            return np.empty((0, seq_length if seq_length else self.sequence_length, max(0, expected_n_features))), np.array([], dtype=int)

        for i in store_id:
            user_data = self.data[self.data['id'] == i].reset_index(drop=True)
            if len(user_data) > seq_length:
                X_seq, y_seq = self._create_sequence(user_data, seq_length, label_idx) # เงื่อนไขการสร้าง sequence คงเดิม
                if len(X_seq) > 0:
                    all_X.append(X_seq)
                    all_y.append(y_seq)
        if not all_X:
            print("No sequences created. Check data length per ID or sequence_length setting.")
            expected_n_features = len(self.data.columns) - 2 if 'id' in self.data.columns else len(self.data.columns) - 1
            return np.empty((0, seq_length if seq_length else self.sequence_length, max(0, expected_n_features))), np.array([], dtype=int)

        final_X = np.vstack(all_X)
        final_y = np.concatenate(all_y)
        print(f"Created {len(final_X)} sequences.")
        if final_X.ndim == 3:
             print(f"Input sequences shape (X): {final_X.shape}")
        else:
             print(f"Warning: Input sequences (X) shape is not as expected: {final_X.shape}")
        print(f"Target labels shape (y): {final_y.shape}")
        final_y = final_y.astype(int)
        print(f"Unique target labels in y: {np.unique(final_y)}")
        return final_X, final_y

    def _create_sequence(self, data, seq_length, label_idx):
        X = []
        y = []
        data_for_seq = data.copy()
        id_col_idx_original = -1
        if 'id' in data_for_seq.columns:
             try:
                  id_col_idx_original = data.columns.tolist().index('id')
             except ValueError:
                  pass
             data_for_seq = data_for_seq.drop(columns=['id'])

        data_array = data_for_seq.values
        if id_col_idx_original != -1 and id_col_idx_original < label_idx:
            adjusted_label_idx = label_idx - 1
        else:
            adjusted_label_idx = label_idx
        if adjusted_label_idx < 0 or adjusted_label_idx >= data_array.shape[1]:
             print(f"Error in _create_sequence: Adjusted label index ({adjusted_label_idx}) is out of bounds.")
             return np.array([]), np.array([], dtype=int)
        if len(data_array) < seq_length + 1:
             return np.array([]), np.array([], dtype=int)
        for i in range(len(data_array) - seq_length):
            seq_window = data_array[i:(i + seq_length), :]
            stress_values_in_window = seq_window[:, adjusted_label_idx]
            if len(set(stress_values_in_window)) == 1:
                next_step_index = i + seq_length
                if next_step_index < len(data_array):
                    next_step_label = data_array[next_step_index, adjusted_label_idx]
                    seq_features = np.delete(seq_window, adjusted_label_idx, axis=1)
                    X.append(seq_features)
                    y.append(next_step_label)
        return np.array(X), np.array(y, dtype=int)


# Class MyHyperModel ที่ปรับปรุงแล้ว (คงเดิม)
class MyHyperModel(kt.HyperModel):
    def __init__(self, n_features, sequence_length, n_classes):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        print(f"Initializing MyHyperModel with n_features={self.n_features}, sequence_length={self.sequence_length}, n_classes={self.n_classes}")
        if self.n_features <= 0:
             print(f"Warning: Initializing MyHyperModel with n_features={self.n_features}. This might cause issues during model building.")
        if self.sequence_length <= 0:
             print(f"Warning: Initializing MyHyperModel with sequence_length={self.sequence_length}. Ensure this is intended.")
        if self.n_classes <= 0:
             print(f"Warning: Initializing MyHyperModel with n_classes={self.n_classes}. Model requires at least 1 class for sigmoid or 2 for softmax.")

    def build(self, hp):
        print("\nBuilding model with hyperparameters:")
        model = keras.Sequential()

        if self.sequence_length is None or self.n_features is None or self.sequence_length <= 0 or self.n_features <= 0:
             raise ValueError(f"Invalid input shape for model: sequence_length={self.sequence_length}, n_features={self.n_features}")
        if self.n_classes <= 0 :
             raise ValueError(f"Invalid number of classes: n_classes={self.n_classes}. Must be > 0.")

        model.add(keras.Input(shape=(self.sequence_length, self.n_features)))

        hp_optimizer_type = hp.Choice('optimizer_type', values=['adam', 'sgd'])
        hp_lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
        hp_dropout_rate = hp.Float('dropout_rate', min_value=0.05, max_value=0.2, step=0.05)
        hp_num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=5, step=1)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256])
        hp_epochs = hp.Int('epochs', min_value=10, max_value=150, step=10)


        optimizer_type_value = hp.get('optimizer_type')
        lstm_units_value = hp.get('lstm_units')
        dropout_rate_value = hp.get('dropout_rate')
        num_lstm_layers_value = hp.get('num_lstm_layers')
        learning_rate_value = hp.get('learning_rate')
        epochs_value_for_hp_search = hp.get('epochs')
        batch_size_value_for_hp_search = hp.get('batch_size')


        print(f"  optimizer_type: {optimizer_type_value}")
        print(f"  lstm_units: {lstm_units_value}")
        print(f"  dropout_rate: {dropout_rate_value}")
        print(f"  learning_rate: {learning_rate_value}")
        print(f"  num_lstm_layers: {num_lstm_layers_value}")
        print(f"  epochs : {epochs_value_for_hp_search}")
        print(f"  batch_size : {batch_size_value_for_hp_search}")


        for i in range(num_lstm_layers_value):
            return_sequences = i < num_lstm_layers_value - 1
            model.add(layers.LSTM(units=lstm_units_value, return_sequences=return_sequences))
            model.add(layers.Dropout(rate=dropout_rate_value))

        if self.n_classes == 1:
             print("  Output layer: Dense units=1, activation='sigmoid' (Binary)")
             model.add(layers.Dense(1, activation='sigmoid'))
        elif self.n_classes >= 2:
             print(f"  Output layer: Dense units={self.n_classes}, activation='softmax' (Multi-class/Binary with Softmax)")
             model.add(layers.Dense(self.n_classes, activation='softmax'))

        if optimizer_type_value == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)
        elif optimizer_type_value == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_value)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type_value}")

        if self.n_classes == 1:
            loss_function = 'binary_crossentropy'
        elif self.n_classes >= 2:
            loss_function = 'categorical_crossentropy'

        print(f"  Compiling model with loss='{loss_function}', optimizer={optimizer_type_value}(lr={learning_rate_value}), metrics=['accuracy']")
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        return model



# --- Function สำหรับ Train และ Evaluate หนึ่ง Model (คงเดิม) ---
def train_evaluate_model(
    feature_columns, model_name, base_dataframe, label_column_name,
    columns_to_exclude_from_scaling, sequence_length, base_timestamp,
    results_base_dir, models_base_dir, tuner_base_dir, assets_base_dir,
    tuner_search_fixed_epochs=50, tuner_search_batch_size=32, tuner_max_trials=30,
    early_stopping_patience_tuner=10, early_stopping_patience_final=15
    ):
    """
    ทำการ Preprocess, Train, Tune, Evaluate, และ Save ผลลัพธ์สำหรับชุด feature ที่กำหนด
    """
    print(f"\n{'='*20} Starting Process for Model: {model_name} {'='*20}")

    # --- 2. โหลดและ Preprocess ข้อมูล ---
    print(f"\n--- 2. Preprocessing Data for {model_name} ---")
    preprocess_obj = preprocess(base_dataframe.copy(), feature_columns)
    dataframe = preprocess_obj.select_columns()
    mapping_path = os.path.join(assets_base_dir, f'category_mapping_{model_name}_{base_timestamp}.json')
    dataframe, mapping = preprocess_obj.label_encoding(mapping_path=mapping_path)
    class_names = []
    n_classes = 0
    if label_column_name in mapping and mapping[label_column_name]:
        try:
            class_items = sorted(mapping[label_column_name].items(), key=lambda item: item[0])
            class_names = [name for index, name in class_items]
            n_classes = len(class_names)
        except Exception as e:
             print(f"Error processing mapping for label column '{label_column_name}' ({model_name}): {e}")
    if n_classes == 0:
        unique_labels_in_df = dataframe[label_column_name].unique()
        unique_labels_in_df = [lbl for lbl in unique_labels_in_df if pd.notna(lbl)]
        if len(unique_labels_in_df) == 2:
            n_classes = 1
            class_names = [str(i) for i in sorted(unique_labels_in_df)]
            print(f"Binary classification detected ({model_name}). n_classes set to 1 for sigmoid output.")
        elif len(unique_labels_in_df) == 1:
            n_classes = 1
            class_names = [str(unique_labels_in_df[0])]
            print(f"Warning: Only one class '{class_names[0]}' found for '{label_column_name}' in {model_name}. Model might not train effectively.")
        elif len(unique_labels_in_df) > 2:
            n_classes = len(unique_labels_in_df)
            class_names = [str(i) for i in sorted(unique_labels_in_df)]
        else:
            print(f"Warning: Could not determine n_classes for '{label_column_name}' in {model_name}. Found unique labels: {unique_labels_in_df}")
            n_classes = 0

    print(f"\nClass Names ({model_name}, determined): {class_names}")
    print(f"Number of Output Neurons (n_classes for model) for {model_name}: {n_classes}")
    actual_num_distinct_classes = len(class_names) if class_names else 0
    print(f"Actual distinct classes for metrics/reporting: {actual_num_distinct_classes}")


    sc = StandardScaler()
    scaler_path = os.path.join(assets_base_dir, f'scaler_{model_name}_{base_timestamp}.pkl')
    current_exclude_scaling = [col for col in columns_to_exclude_from_scaling if col in dataframe.columns]
    dataframe = preprocess_obj.scale_data(sc, exclude_columns=current_exclude_scaling, save_path=scaler_path)

    # --- 3. สร้าง Sequence ข้อมูล ---
    print(f"\n--- 3. Creating Sequences for {model_name} ---")
    preprocess_obj.set_sequence_length(sequence_length)
    X, y = preprocess_obj.create_sequence(label_column_name)
    n_features = X.shape[-1] if X.ndim == 3 and X.shape[0] > 0 else 0
    print(f"\nNumber of Features per Time Step (n_features) for {model_name}: {n_features}")

    # --- 4. จัดการข้อมูลไม่สมดุลด้วย SMOTE และ Split Train/Test ---
    print(f"\n--- 4. SMOTE and Train/Test Split for {model_name} ---")
    X_train_resampled, y_train_resampled = np.array([]), np.array([], dtype=int)
    X_test, y_test = np.array([]), np.array([], dtype=int)
    y_train_eval_target = np.array([])
    X_pre_smote_train = np.array([])

    if X.shape[0] > 0 and y.shape[0] > 0 and X.shape[0] == y.shape[0]:
        X_pre_smote_train, X_test, y_pre_smote_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"  Train/Test split: 70% train, 30% test (test_size=0.3)")
        print(f"  Shape of X_pre_smote_train ({model_name}): {X_pre_smote_train.shape}")
        print(f"  Shape of y_pre_smote_train ({model_name}): {y_pre_smote_train.shape}")
        print(f"  Class distribution in y_pre_smote_train ({model_name}):\n{pd.Series(y_pre_smote_train).value_counts().sort_index()}")

        if X_pre_smote_train.shape[0] > 0:
            min_class_count = np.min(np.unique(y_pre_smote_train, return_counts=True)[1]) if len(np.unique(y_pre_smote_train)) > 1 else X_pre_smote_train.shape[0]
            k_neighbors = min(5, max(1, min_class_count - 1))
            if k_neighbors < 1 or len(np.unique(y_pre_smote_train)) < 2 :
                print(f"Warning: Cannot apply SMOTE for {model_name} (k_neighbors={k_neighbors}, unique_classes={len(np.unique(y_pre_smote_train))}). Using original training data.")
                X_train_resampled = X_pre_smote_train
                y_train_resampled = y_pre_smote_train
            else:
                 try:
                      print(f"Applying SMOTE with k_neighbors={k_neighbors} for {model_name}")
                      smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
                      X_pre_smote_train_flat = X_pre_smote_train.reshape(X_pre_smote_train.shape[0], -1)
                      X_train_resampled_flat, y_train_resampled = smote.fit_resample(X_pre_smote_train_flat, y_pre_smote_train)
                      X_train_resampled = X_train_resampled_flat.reshape(-1, X_pre_smote_train.shape[1], X_pre_smote_train.shape[2])
                 except ValueError as e:
                      print(f"Warning: SMOTE failed for {model_name} ('{e}'). Using original training data.")
                      X_train_resampled = X_pre_smote_train
                      y_train_resampled = y_pre_smote_train
            print(f"\nShape of X_train ({model_name}) after Split (SMOTE if applied): {X_train_resampled.shape}")
            print(f"Shape of y_train ({model_name}) after Split (SMOTE if applied): {y_train_resampled.shape}")
            print(f"Class distribution in y_train ({model_name}) after SMOTE:\n{pd.Series(y_train_resampled).value_counts().sort_index()}")

            if n_classes > 1 and y_train_resampled.size > 0:
                y_train_eval_target = tf.keras.utils.to_categorical(y_train_resampled, num_classes=n_classes)
                print(f"  y_train_eval_target (one-hot for softmax with {n_classes} neurons) shape: {y_train_eval_target.shape}")
            elif n_classes == 1 and y_train_resampled.size > 0:
                 y_train_eval_target = y_train_resampled
                 print(f"  y_train_eval_target (integer for sigmoid) shape: {y_train_eval_target.shape}")
            else:
                 y_train_eval_target = np.array([])

    # --- 5. สร้างโมเดลและ Tune Hyperparameter ---
    print(f"\n--- 5. Hyperparameter Tuning for {model_name} ---")
    tuner = None
    tuner_search_time_min_sec = "N/A"
    best_model = None
    best_hyperparameters_dict = {}
    actual_epochs_trained_final = 0
    final_model_train_time_min_sec = "N/A"

    tuner_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=early_stopping_patience_tuner,
        verbose=1,
        restore_best_weights=True
    )

    can_train_tuner = X_train_resampled.shape[0] > 0 and y_train_eval_target.shape[0] > 0 and \
                      X_train_resampled.shape[0] == y_train_eval_target.shape[0] and \
                      n_features > 0 and n_classes > 0

    if can_train_tuner:
        hypermodel = MyHyperModel(n_features=n_features, sequence_length=sequence_length, n_classes=n_classes)
        tuner_proj_dir = os.path.join(tuner_base_dir, model_name)
        project_name = f'stress_lstm_{model_name}_{base_timestamp}'
        try:
            tuner = kt.BayesianOptimization(
                hypermodel, objective="val_accuracy", max_trials=tuner_max_trials,
                executions_per_trial=1, directory=tuner_proj_dir,
                project_name=project_name, overwrite=True )
            print(f"Keras Tuner initialized for {model_name}.")
            start_time_tuner = time.time()
            tuner.search( X_train_resampled, y_train_eval_target,
                epochs=tuner_search_fixed_epochs,
                batch_size=tuner_search_batch_size,
                validation_split=0.2,
                callbacks=[tuner_early_stopping]
            )
            end_time_tuner = time.time()
            tuner_search_time_sec = end_time_tuner - start_time_tuner
            tuner_search_time_min_sec = f"{int(tuner_search_time_sec // 60)}m {int(tuner_search_time_sec % 60)}s"
            print(f"\nHyperparameter search completed in: {tuner_search_time_min_sec}")

            run_results_dir = os.path.join(results_base_dir, f"{model_name}_{base_timestamp}")
            os.makedirs(run_results_dir, exist_ok=True)
            tuner_summary_path = os.path.join(run_results_dir, 'tuner_results_summary.txt')
            f_io_tuner = io.StringIO()
            with redirect_stdout(f_io_tuner):
                tuner.results_summary(num_trials=10)
            summary_str_tuner = f_io_tuner.getvalue()
            print(summary_str_tuner)
            with open(tuner_summary_path, 'w', encoding='utf-8') as file_tuner:
                file_tuner.write(summary_str_tuner)
            print(f"Saved tuning summary to {tuner_summary_path}")

            best_hps_retrieved = tuner.get_best_hyperparameters(num_trials=1)
            if best_hps_retrieved:
                best_hps_tuner = best_hps_retrieved[0]
                best_hyperparameters_dict = best_hps_tuner.values
                print(f"\nBest Hyperparameters found by Tuner for {model_name}:")
                for param, value in best_hyperparameters_dict.items():
                    print(f"  {param}: {value}")

                print(f"\n--- Training Final Model with Best Hyperparameters for {model_name} ---")
                final_model_to_train = hypermodel.build(best_hps_tuner)

                final_train_epochs = best_hyperparameters_dict.get('epochs', tuner_search_fixed_epochs)
                final_train_batch_size = best_hyperparameters_dict.get('batch_size', tuner_search_batch_size)
                print(f"  Training with tuned epochs: {final_train_epochs}, tuned batch_size: {final_train_batch_size}")

                final_model_early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy' if X_train_resampled.shape[0] * 0.1 >= 10 else 'accuracy',
                    patience=early_stopping_patience_final,
                    verbose=1,
                    restore_best_weights=True
                )
                final_model_val_split = 0.1 if X_train_resampled.shape[0] >= 20 else 0.0

                start_time_final_train = time.time()
                history_final_model = final_model_to_train.fit(
                    X_train_resampled,
                    y_train_eval_target,
                    epochs=final_train_epochs,
                    batch_size=final_train_batch_size,
                    validation_split=final_model_val_split if final_model_val_split > 0 else None,
                    callbacks=[final_model_early_stopping],
                    verbose=1
                )
                end_time_final_train = time.time()
                final_model_train_time_sec = end_time_final_train - start_time_final_train
                final_model_train_time_min_sec = f"{int(final_model_train_time_sec // 60)}m {int(final_model_train_time_sec % 60)}s"
                print(f"Final model training completed in: {final_model_train_time_min_sec}")

                best_model = final_model_to_train
                if history_final_model and hasattr(final_model_early_stopping, 'stopped_epoch') and final_model_early_stopping.stopped_epoch > 0:
                     actual_epochs_trained_final = final_model_early_stopping.stopped_epoch + 1
                elif history_final_model:
                     actual_epochs_trained_final = len(history_final_model.history['loss'])
                print(f"  Final model effectively trained for {actual_epochs_trained_final} epochs.")
            else:
                print(f"Warning: Tuner did not return any best hyperparameters for {model_name}. Trying to get best model from search trials.")
                tuner_best_models = tuner.get_best_models(num_models=1)
                if tuner_best_models:
                    best_model = tuner_best_models[0]
                    best_hyperparameters_dict = best_model.get_config()
                    print("  Using best model directly from tuner search phase as fallback.")
                else:
                    print(f"Error: No models could be retrieved from tuner for {model_name} after search.")

        except Exception as e_tuner:
            print(f"An error occurred during hyperparameter search or final model training for {model_name}: {e_tuner}")
            import traceback; traceback.print_exc()
    else:
        print(f"Skipping Keras Tuner and final training: insufficient/invalid data (X_train:{X_train_resampled.shape}, y_train_eval:{y_train_eval_target.shape}, n_feat:{n_features}, n_class:{n_classes})")


    # --- 9. ประเมินผลโมเดลที่ดีที่สุดบน Test Set ---
    loss, accuracy = np.nan, np.nan
    f1_weighted, balanced_acc, roc_auc_metric = np.nan, np.nan, np.nan
    class_report_str = "Evaluation skipped or failed."
    conf_matrix_list = None
    y_true_classes_test = y_test

    y_test_target_eval = y_test
    if n_classes > 1 and y_test.size > 0:
        y_test_target_eval = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

    if best_model is not None and X_test.shape[0] > 0 and y_test.shape[0] > 0 and X_test.shape[0] == y_test.shape[0] and n_classes > 0:
        print(f"Evaluating on Test Set. X_test: {X_test.shape}, y_test_target_eval (for model.evaluate): {y_test_target_eval.shape}, y_true_classes_test (for metrics): {y_true_classes_test.shape}")
        try:
            loss, accuracy = best_model.evaluate(X_test, y_test_target_eval, verbose=0)
            y_pred_proba_test = best_model.predict(X_test)

            if n_classes > 1:
                y_pred_classes_test = np.argmax(y_pred_proba_test, axis=1)
                y_test_roc_one_hot = tf.keras.utils.to_categorical(y_true_classes_test, num_classes=actual_num_distinct_classes)
                if y_test_roc_one_hot.shape[0] > 0 and y_pred_proba_test.shape[0] > 0 and y_pred_proba_test.shape[1] == actual_num_distinct_classes:
                     roc_auc_metric = roc_auc_score(y_test_roc_one_hot, y_pred_proba_test, average='weighted', multi_class='ovr')
                elif y_test_roc_one_hot.shape[0] > 0 and y_pred_proba_test.shape[0] > 0 :
                     print(f"Warning: ROC AUC for multi-class might be inaccurate due to shape mismatch. y_pred_proba_test.shape[1]={y_pred_proba_test.shape[1]}, actual_num_distinct_classes={actual_num_distinct_classes}")
                     if y_pred_proba_test.shape[1] == 1 and actual_num_distinct_classes == 2:
                        roc_auc_metric = roc_auc_score(y_true_classes_test, y_pred_proba_test.flatten())
            else:
                y_pred_classes_test = (y_pred_proba_test > 0.5).astype(int).flatten()
                if y_true_classes_test.size > 0 and y_pred_proba_test.size > 0:
                    roc_auc_metric = roc_auc_score(y_true_classes_test, y_pred_proba_test.flatten())

            report_labels_indices = range(actual_num_distinct_classes) if actual_num_distinct_classes > 0 else None
            report_target_names = class_names if class_names and len(class_names) == actual_num_distinct_classes else [f"C_{i}" for i in report_labels_indices or []]

            f1_weighted = f1_score(y_true_classes_test, y_pred_classes_test, average='weighted', zero_division=0, labels=report_labels_indices)
            balanced_acc = balanced_accuracy_score(y_true_classes_test, y_pred_classes_test)
            class_report_str = classification_report(y_true_classes_test, y_pred_classes_test, target_names=report_target_names, zero_division=0, labels=report_labels_indices)
            conf_matrix_val_test = confusion_matrix(y_true_classes_test, y_pred_classes_test, labels=report_labels_indices)
            conf_matrix_list = conf_matrix_val_test.tolist() if conf_matrix_val_test is not None else None
        except Exception as e_eval:
            print(f"Error during test set evaluation for {model_name}: {e_eval}")
            import traceback; traceback.print_exc()

    # --- 10. เตรียมไดเรกทอรี ---
    run_results_dir_path = os.path.join(results_base_dir, f"{model_name}_{base_timestamp}")
    model_run_name_path = f"Lstm_{model_name}_{base_timestamp}"
    run_models_dir_path = os.path.join(models_base_dir, model_run_name_path)
    os.makedirs(run_results_dir_path, exist_ok=True)
    if best_model is not None: os.makedirs(run_models_dir_path, exist_ok=True)

    # --- 11. เก็บ Hyperparameters และผลลัพธ์ ---
    hyperparameters_used_log = {
        'model_name': model_name, 'csv_file_path': csv_file_path,
        'label_column_name': label_column_name, 'features_used': feature_columns,
        'columns_excluded_from_scaling': current_exclude_scaling, 'sequence_length': sequence_length,
        'n_features_in_sequence': n_features,
        'n_output_neurons_model': n_classes,
        'actual_distinct_classes_data': actual_num_distinct_classes,
        'class_names_used': class_names,
        'oversampling_method': 'SMOTE' if X_train_resampled.shape[0] > X_pre_smote_train.shape[0] and X_pre_smote_train.size > 0 else 'None',
        'best_model_hyperparameters_from_tuner': best_hyperparameters_dict if best_hyperparameters_dict else "N/A",
        'final_model_actual_epochs_trained': actual_epochs_trained_final if actual_epochs_trained_final > 0 else "N/A",
    }

    if tuner is not None and hasattr(tuner, 'oracle'):
        best_tuned_epochs_val = best_hyperparameters_dict.get('epochs', 'N/A')
        best_tuned_batch_size_val = best_hyperparameters_dict.get('batch_size', 'N/A')

        hyperparameters_used_log['tuner_summary_info'] = {
            'tuner_type': tuner.__class__.__name__, 'tuner_max_trials': tuner.oracle.max_trials,
            'tuner_objective': str(tuner.oracle.objective),
            'tuner_search_fixed_epochs_per_trial': tuner_search_fixed_epochs,
            'tuner_search_fixed_batch_size_per_trial': tuner_search_batch_size,
            'tuner_search_early_stopping_patience': early_stopping_patience_tuner,
            'best_tuned_epochs_hyperparam_value': best_tuned_epochs_val,
            'best_tuned_batch_size_hyperparam_value': best_tuned_batch_size_val,
            'final_model_early_stopping_patience': early_stopping_patience_final,
            'validation_split_tuner_search': 0.2,
            'validation_split_final_model_train': final_model_val_split if 'final_model_val_split' in locals() else "N/A",
        }
    else: hyperparameters_used_log['tuner_summary_info'] = "N/A"

    results_summary_log = {
        'Model_Name': model_name, 'Timestamp_Run_Start': base_timestamp,
        'Timestamp_Evaluation_End': time.strftime("%Y-%m-%d %H:%M:%S"),
        'Results_Directory': run_results_dir_path,
        'Model_Directory': run_models_dir_path if best_model is not None else "N/A",
        'Tuner_Search_Time': tuner_search_time_min_sec,
        'Final_Model_Training_Time': final_model_train_time_min_sec,
        'Hyperparameters_Used': hyperparameters_used_log,
        'Test_Set_Results': { 'Loss': float(loss) if not np.isnan(loss) else None,
            'Accuracy': float(accuracy) if not np.isnan(accuracy) else None,
            'F1_Weighted': float(f1_weighted) if not np.isnan(f1_weighted) else None,
            'Balanced_Accuracy': float(balanced_acc) if not np.isnan(balanced_acc) else None,
            'ROC_AUC': float(roc_auc_metric) if not np.isnan(roc_auc_metric) else None,
            'Num_Test_Samples': X_test.shape[0] },
        'Classification_Report': class_report_str, 'Confusion_Matrix': conf_matrix_list }

    # --- 12. บันทึกผลลัพธ์ ---
    json_path = os.path.join(run_results_dir_path, "test_results_summary.json")
    txt_path = os.path.join(run_results_dir_path, "test_summary.txt")
    cm_img_path = os.path.join(run_results_dir_path, "confusion_matrix.png")

    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(results_summary_log, f_json, indent=4, ensure_ascii=False)

    with open(txt_path, 'w', encoding='utf-8') as f_txt:
        f_txt.write(f"--- Test Set Evaluation Summary for {model_name} ---\n")
        f_txt.write(f"Run Start Timestamp: {results_summary_log['Timestamp_Run_Start']}\n")
        f_txt.write(f"Evaluation End Timestamp: {results_summary_log['Timestamp_Evaluation_End']}\n")
        f_txt.write(f"Results Directory: {results_summary_log['Results_Directory']}\n")
        f_txt.write(f"Model Directory: {results_summary_log['Model_Directory']}\n")
        f_txt.write(f"Tuner Search Time: {results_summary_log['Tuner_Search_Time']}\n")
        f_txt.write(f"Final Model Training Time: {results_summary_log['Final_Model_Training_Time']}\n")


        f_txt.write("\n--- Hyperparameters Used ---\n")
        hps_log_data = results_summary_log['Hyperparameters_Used']
        for k_hps, v_hps in hps_log_data.items():
            if k_hps == 'best_model_hyperparameters_from_tuner':
                f_txt.write(f"Best Model Hyperparameters (found by Tuner):\n")
                if isinstance(v_hps, dict):
                    for hpk, hpv in v_hps.items():
                        f_txt.write(f"  - {hpk}: {hpv}\n")
                else: f_txt.write(f"  {v_hps}\n")
            elif k_hps == 'tuner_summary_info':
                f_txt.write(f"Tuner and Final Model Training Info:\n")
                if isinstance(v_hps, dict):
                    for tsk, tsv in v_hps.items():
                        f_txt.write(f"  - {tsk}: {tsv}\n")
                else: f_txt.write(f"  {v_hps}\n")
            elif k_hps == 'features_used':
                f_txt.write(f"Features Used ({len(v_hps if isinstance(v_hps, list) else [])}):\n")
                if isinstance(v_hps, list):
                    for feature in v_hps: f_txt.write(f"  - {feature}\n")
            else: f_txt.write(f"{k_hps}: {v_hps}\n")

        f_txt.write("\n--- Test Set Results ---\n")
        test_res_data = results_summary_log['Test_Set_Results']
        f_txt.write(f"Number of Test Samples: {test_res_data.get('Num_Test_Samples', 'N/A')}\n")
        f_txt.write(f"Test Loss: {test_res_data.get('Loss', 'N/A')}\n")
        f_txt.write(f"Test Accuracy: {test_res_data.get('Accuracy', 'N/A')}\n")
        f_txt.write(f"Test F1 (Weighted): {test_res_data.get('F1_Weighted', 'N/A')}\n")
        f_txt.write(f"Test Balanced Acc: {test_res_data.get('Balanced_Accuracy', 'N/A')}\n")
        f_txt.write(f"Test ROC AUC: {test_res_data.get('ROC_AUC', 'N/A')}\n")

        f_txt.write("\n--- Classification Report ---\n")
        f_txt.write(f"{results_summary_log.get('Classification_Report', 'N/A')}\n")

        f_txt.write("\n--- Confusion Matrix ---\n")
        cm_list_data = results_summary_log.get('Confusion_Matrix')
        cm_txt_labels = class_names if class_names and len(class_names) == actual_num_distinct_classes else [f"C_{i}" for i in range(actual_num_distinct_classes if actual_num_distinct_classes > 0 else 0)]

        if cm_list_data and cm_txt_labels:
            header_labels_txt = [str(lbl)[:10].ljust(10) for lbl in cm_txt_labels]
            f_txt.write("True\\Pred".ljust(11) + " ".join(header_labels_txt) + "\n")
            f_txt.write("-" * (11 + len(header_labels_txt) * 11) + "\n")
            for i_cm, row_cm in enumerate(cm_list_data):
                 if i_cm < len(cm_txt_labels):
                     row_label_txt = str(cm_txt_labels[i_cm])[:10].ljust(10)
                     row_values_txt = [str(val_cm).ljust(10) for val_cm in row_cm]
                     f_txt.write(f"{row_label_txt} " + " ".join(row_values_txt) + "\n")
        else:
            f_txt.write("Confusion Matrix not available or could not be generated.\n")
        f_txt.write("\n" + "="*40 + "\n")

    cm_plot_labels_final = class_names if class_names and len(class_names) == actual_num_distinct_classes else [f"C_{i}" for i in range(actual_num_distinct_classes if actual_num_distinct_classes > 0 else 0)]
    if conf_matrix_list is not None and cm_plot_labels_final:
        try:
            plt.figure(figsize=(max(6, len(cm_plot_labels_final) * 0.8), max(5, len(cm_plot_labels_final) * 0.6)))
            sns.heatmap(np.array(conf_matrix_list), annot=True, fmt='d', cmap='Blues',
                        xticklabels=cm_plot_labels_final, yticklabels=cm_plot_labels_final)
            plt.xlabel('Predicted Label'); plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {model_name} Model'); plt.tight_layout()
            plt.savefig(cm_img_path); plt.close()
        except Exception as e_cm_plot: print(f"Error saving CM plot: {e_cm_plot}")

    # --- 13. บันทึกโมเดล ---
    if best_model is not None:
        try:
            tf.saved_model.save(best_model, run_models_dir_path)
            print(f"Model for {model_name} saved to {run_models_dir_path}")
        except Exception as e_save_model: print(f"Error saving model: {e_save_model}")

    return results_summary_log


print("--- 1. Checking GPU Availability ---"); check_gpu()
print("\n--- Setting up Configuration ---")
csv_file_path = "Data/labeled_stress_data.csv"
label_column_name = 'stress'
sequence_length = 30

# Tuner and Training Configuration
TUNER_MAX_TRIALS = 30
TUNER_SEARCH_FIXED_EPOCHS = 50
TUNER_SEARCH_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE_TUNER = 10
EARLY_STOPPING_PATIENCE_FINAL_MODEL = 15

columns_combined = [
    'HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_LFHF',
    'EDA_Phasic', 'SCR_Amplitude', 'EDA_Tonic', 'SCR_Onsets',
    'gender', 'bmi', 'sleep', 'type', 'stress', 'id'
]
columns_to_exclude_from_scaling = [label_column_name, 'id']

# Directory Setup
results_base_dir = "Results"; models_base_dir = "models"
tuner_base_dir = "keras_tuner_dir"; assets_base_dir = "assets"
for dir_path in [results_base_dir, models_base_dir, tuner_base_dir, assets_base_dir]:
    os.makedirs(dir_path, exist_ok=True)

base_timestamp = time.strftime("%Y%m%d_%H%M%S")
print(f"Base Timestamp for this run: {base_timestamp}")

print(f"\n--- Loading Base Data from {csv_file_path} ---")
try:
    if not os.path.exists(csv_file_path):
        alt_csv_path = os.path.join("..", csv_file_path)
        if os.path.exists(alt_csv_path):
            csv_file_path = alt_csv_path
            print(f"Using alternative CSV path: {csv_file_path}")
        else:
            raise FileNotFoundError(f"CSV file not found at primary path: {csv_file_path} or alt path: {alt_csv_path}")
    base_dataframe = pd.read_csv(csv_file_path)
    print("Base dataframe loaded successfully.")
    if 'id' not in base_dataframe.columns: raise ValueError("'id' column missing. It is required for creating sequences.")
    if label_column_name not in base_dataframe.columns: raise ValueError(f"Label column '{label_column_name}' missing.")
except Exception as e_load:
    print(f"Error loading CSV: {e_load}")
    import traceback; traceback.print_exc()
    exit()

all_results_final = {}

# ===== การเปลี่ยนแปลงส่วนที่ 2: เรียกใช้ฟังก์ชันเพื่อเทรนโมเดลเดียว =====
try:
    model_name_run = "Combined_Features"
    print(f"\n>>> PROCESSING MODEL: {model_name_run} ({base_timestamp})")
    results_run = train_evaluate_model(
        feature_columns=columns_combined,
        model_name=model_name_run,
        base_dataframe=base_dataframe,
        label_column_name=label_column_name,
        columns_to_exclude_from_scaling=columns_to_exclude_from_scaling,
        sequence_length=sequence_length,
        base_timestamp=base_timestamp,
        results_base_dir=results_base_dir,
        models_base_dir=models_base_dir,
        tuner_base_dir=tuner_base_dir,
        assets_base_dir=assets_base_dir,
        tuner_search_fixed_epochs=TUNER_SEARCH_FIXED_EPOCHS,
        tuner_search_batch_size=TUNER_SEARCH_BATCH_SIZE,
        tuner_max_trials=TUNER_MAX_TRIALS,
        early_stopping_patience_tuner=EARLY_STOPPING_PATIENCE_TUNER,
        early_stopping_patience_final=EARLY_STOPPING_PATIENCE_FINAL_MODEL
    )
    all_results_final[model_name_run] = results_run
except Exception as e_main:
    print(f"\nUnexpected error during model processing: {e_main}")
    import traceback; traceback.print_exc()
    all_results_final["Combined_Features"] = {"Error": str(e_main), "Traceback": traceback.format_exc()}


print("\n" + "="*30 + " Final Results " + "="*30)
print(f"{'Model':<20} | {'Test Acc':<12} | {'Test F1 (W)':<13} | {'Test Bal Acc':<14} | {'Test ROC AUC':<14} | {'Tuner Time':<15} | {'Final Train Time':<18}")
print("-" * 115) # Adjusted width
for model_key, res_val in all_results_final.items():
    if "Error" in res_val:
        print(f"{model_key:<20} | {'Error':<12} | {'Error':<13} | {'Error':<14} | {'Error':<14} | {'N/A':<15} | {'N/A':<18}")
    elif isinstance(res_val, dict) and 'Test_Set_Results' in res_val:
        test_res_print = res_val.get('Test_Set_Results', {})
        acc_p = test_res_print.get('Accuracy')
        f1_p = test_res_print.get('F1_Weighted')
        bal_p = test_res_print.get('Balanced_Accuracy')
        roc_p = test_res_print.get('ROC_AUC')
        time_tuner_p = res_val.get('Tuner_Search_Time', 'N/A')
        time_final_train_p = res_val.get('Final_Model_Training_Time', 'N/A')
        print(f"{model_key:<20} | {f'{acc_p:.4f}' if acc_p is not None else 'N/A':<12} | {f'{f1_p:.4f}' if f1_p is not None else 'N/A':<13} | "
              f"{f'{bal_p:.4f}' if bal_p is not None else 'N/A':<14} | {f'{roc_p:.4f}' if roc_p is not None else 'N/A':<14} | {str(time_tuner_p):<15} | {str(time_final_train_p):<18}")
print("\n--- Script Execution Finished ---")