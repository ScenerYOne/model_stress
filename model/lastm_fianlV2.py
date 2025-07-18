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
import time  # สำหรับจับเวลาการเทรนและสร้าง timestamp
import json  # สำหรับบันทึกผลลัพธ์ในรูปแบบ JSON
import matplotlib.pyplot as plt  # สำหรับพลอต Confusion Matrix
import seaborn as sns  # สำหรับพลอต Confusion Matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split # เปลี่ยนไปใช้ KFold
from sklearn.model_selection import StratifiedKFold  # <--- เพิ่ม StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix
import pandas.api.types as ptypes
import copy  # Import copy for deep copying dataframes

# --- กำหนดค่า K-Fold Cross-Validation ---
N_SPLITS = 5  # <--- กำหนดจำนวน Folds ที่นี่ (เช่น 3, 5, 10)


# --- กำหนด function/class ที่ต้องใช้ภายในสคริปต์ ---

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


# Class preprocess ที่รวมเข้ามา (คงเดิมส่วนใหญ่)
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

    # scale_data จะไม่ถูกใช้โดยตรงจาก preprocess object ใน K-Fold loop
    # แต่ scaler จะถูกสร้างและ fit/transform ในแต่ละ fold

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
            print(
                f"Error: Label column '{label_column_name}' not found in the dataframe columns: {self.data.columns.tolist()}")
            expected_n_features = len(self.data.columns) - 2 if 'id' in self.data.columns else len(
                self.data.columns) - 1
            return np.empty(
                (0, seq_length if seq_length else self.sequence_length, max(0, expected_n_features))), np.array([],
                                                                                                                dtype=int)

        for i in store_id:
            user_data = self.data[self.data['id'] == i].reset_index(drop=True)
            if len(user_data) > seq_length:
                X_seq, y_seq = self._create_sequence(user_data, seq_length, label_idx)
                if len(X_seq) > 0:
                    all_X.append(X_seq)
                    all_y.append(y_seq)
        if not all_X:
            print("No sequences created. Check data length per ID or sequence_length setting.")
            expected_n_features = len(self.data.columns) - 2 if 'id' in self.data.columns else len(
                self.data.columns) - 1
            return np.empty(
                (0, seq_length if seq_length else self.sequence_length, max(0, expected_n_features))), np.array([],
                                                                                                                dtype=int)

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


# Class MyHyperModel (คงเดิม)
class MyHyperModel(kt.HyperModel):
    def __init__(self, n_features, sequence_length, n_classes):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        print(
            f"Initializing MyHyperModel with n_features={self.n_features}, sequence_length={self.sequence_length}, n_classes={self.n_classes}")
        if self.n_features <= 0:
            print(
                f"Warning: Initializing MyHyperModel with n_features={self.n_features}. This might cause issues during model building.")
        if self.sequence_length <= 0:
            print(
                f"Warning: Initializing MyHyperModel with sequence_length={self.sequence_length}. Ensure this is intended.")
        if self.n_classes <= 0:
            print(
                f"Warning: Initializing MyHyperModel with n_classes={self.n_classes}. Model requires at least 1 class for sigmoid or 2 for softmax.")

    def build(self, hp):
        print("\nBuilding model with hyperparameters:")
        model = keras.Sequential()

        if self.sequence_length is None or self.n_features is None or self.sequence_length <= 0 or self.n_features <= 0:
            raise ValueError(
                f"Invalid input shape for model: sequence_length={self.sequence_length}, n_features={self.n_features}")
        if self.n_classes <= 0:
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
            print(
                f"  Output layer: Dense units={self.n_classes}, activation='softmax' (Multi-class/Binary with Softmax)")
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

        print(
            f"  Compiling model with loss='{loss_function}', optimizer={optimizer_type_value}(lr={learning_rate_value}), metrics=['accuracy']")
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        return model


# --- Function สำหรับ Process หนึ่ง Fold (ปรับปรุงจาก train_evaluate_model เดิม) ---
def process_single_fold(
        X_train_fold, y_train_fold, X_test_fold, y_test_fold,  # ข้อมูลสำหรับ Fold นี้
        fold_num, model_name, base_timestamp,  # ข้อมูลเฉพาะของ Fold และ Run
        n_features, n_classes_model_output, class_names_overall, actual_num_distinct_classes_overall,
        # ข้อมูลที่ได้จากการ Preprocess ครั้งเดียว
        sequence_length, label_column_name,  # Configurations
        results_base_dir, models_base_dir, tuner_base_dir, assets_base_dir,  # Paths
        tuner_search_fixed_epochs=50, tuner_search_batch_size=32, tuner_max_trials=30,
        early_stopping_patience_tuner=10, early_stopping_patience_final=15,
        csv_file_path_for_logging="N/A",  # For logging consistency
        feature_columns_for_logging=[]  # For logging consistency
):
    """
    ทำการ Scale, SMOTE, Tune, Train, Evaluate, และ Save ผลลัพธ์สำหรับ Fold ที่กำหนด
    """
    print(f"\n{'=' * 20} Starting Process for Model: {model_name}, Fold: {fold_num + 1}/{N_SPLITS} {'=' * 20}")

    # --- 1. Scaling Data for the current fold ---
    print(f"\n--- 1. Scaling Data for Fold {fold_num + 1} ---")
    scaler = StandardScaler()
    # Reshape for scaler: (samples * timesteps, features)
    X_train_fold_reshaped = X_train_fold.reshape(-1, n_features)
    X_test_fold_reshaped = X_test_fold.reshape(-1, n_features)

    X_train_fold_scaled_reshaped = scaler.fit_transform(X_train_fold_reshaped)
    X_test_fold_scaled_reshaped = scaler.transform(X_test_fold_reshaped)

    # Reshape back to (samples, timesteps, features)
    X_train_fold_scaled = X_train_fold_scaled_reshaped.reshape(X_train_fold.shape)
    X_test_fold_scaled = X_test_fold_scaled_reshaped.reshape(X_test_fold.shape)

    scaler_path = os.path.join(assets_base_dir, f'scaler_{model_name}_fold{fold_num + 1}_{base_timestamp}.pkl')
    scaler_dir = os.path.dirname(scaler_path)
    if scaler_dir and not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler for Fold {fold_num + 1} to {scaler_path}")

    # --- 2. SMOTE (applied to the current fold's training data) ---
    print(f"\n--- 2. SMOTE for Fold {fold_num + 1} ---")
    X_train_resampled, y_train_resampled = np.array([]), np.array([], dtype=int)
    y_train_eval_target = np.array([])

    if X_train_fold_scaled.shape[0] > 0:
        print(f"  Shape of X_train_fold_scaled ({model_name}, Fold {fold_num + 1}): {X_train_fold_scaled.shape}")
        print(f"  Shape of y_train_fold ({model_name}, Fold {fold_num + 1}): {y_train_fold.shape}")
        print(
            f"  Class distribution in y_train_fold ({model_name}, Fold {fold_num + 1}):\n{pd.Series(y_train_fold).value_counts().sort_index()}")

        min_class_count = np.min(np.unique(y_train_fold, return_counts=True)[1]) if len(
            np.unique(y_train_fold)) > 1 else X_train_fold_scaled.shape[0]
        k_neighbors = min(5, max(1, min_class_count - 1))
        if k_neighbors < 1 or len(np.unique(y_train_fold)) < 2:
            print(
                f"Warning: Cannot apply SMOTE for {model_name}, Fold {fold_num + 1} (k_neighbors={k_neighbors}, unique_classes={len(np.unique(y_train_fold))}). Using original training data for this fold.")
            X_train_resampled = X_train_fold_scaled
            y_train_resampled = y_train_fold
        else:
            try:
                print(f"Applying SMOTE with k_neighbors={k_neighbors} for {model_name}, Fold {fold_num + 1}")
                smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
                X_train_flat = X_train_fold_scaled.reshape(X_train_fold_scaled.shape[0], -1)
                X_train_resampled_flat, y_train_resampled = smote.fit_resample(X_train_flat, y_train_fold)
                X_train_resampled = X_train_resampled_flat.reshape(-1, X_train_fold_scaled.shape[1],
                                                                   X_train_fold_scaled.shape[2])
            except ValueError as e:
                print(
                    f"Warning: SMOTE failed for {model_name}, Fold {fold_num + 1} ('{e}'). Using original training data for this fold.")
                X_train_resampled = X_train_fold_scaled
                y_train_resampled = y_train_fold
        print(
            f"\nShape of X_train_resampled ({model_name}, Fold {fold_num + 1}) after SMOTE: {X_train_resampled.shape}")
        print(f"Shape of y_train_resampled ({model_name}, Fold {fold_num + 1}) after SMOTE: {y_train_resampled.shape}")
        print(
            f"Class distribution in y_train_resampled ({model_name}, Fold {fold_num + 1}) after SMOTE:\n{pd.Series(y_train_resampled).value_counts().sort_index()}")

        if n_classes_model_output > 1 and y_train_resampled.size > 0:
            y_train_eval_target = tf.keras.utils.to_categorical(y_train_resampled, num_classes=n_classes_model_output)
        elif n_classes_model_output == 1 and y_train_resampled.size > 0:
            y_train_eval_target = y_train_resampled
        else:
            y_train_eval_target = np.array([])

    # --- 3. สร้างโมเดลและ Tune Hyperparameter ---
    print(f"\n--- 3. Hyperparameter Tuning for {model_name}, Fold {fold_num + 1} ---")
    tuner = None
    tuner_search_time_min_sec = "N/A"
    best_model_fold = None
    best_hyperparameters_dict = {}
    actual_epochs_trained_final = 0
    final_model_train_time_min_sec = "N/A"

    tuner_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=early_stopping_patience_tuner,
        verbose=1, restore_best_weights=True)

    can_train_tuner = X_train_resampled.shape[0] > 0 and y_train_eval_target.shape[0] > 0 and \
                      X_train_resampled.shape[0] == y_train_eval_target.shape[0] and \
                      n_features > 0 and n_classes_model_output > 0

    if can_train_tuner:
        hypermodel = MyHyperModel(n_features=n_features, sequence_length=sequence_length,
                                  n_classes=n_classes_model_output)
        tuner_proj_dir = os.path.join(tuner_base_dir, f"{model_name}_fold{fold_num + 1}")
        project_name = f'stress_lstm_{model_name}_fold{fold_num + 1}_{base_timestamp}'
        try:
            tuner = kt.BayesianOptimization(
                hypermodel, objective="val_accuracy", max_trials=tuner_max_trials,
                executions_per_trial=1, directory=tuner_proj_dir,
                project_name=project_name, overwrite=True)
            print(f"Keras Tuner initialized for {model_name}, Fold {fold_num + 1}.")
            start_time_tuner = time.time()
            tuner.search(X_train_resampled, y_train_eval_target,
                         epochs=tuner_search_fixed_epochs, batch_size=tuner_search_batch_size,
                         validation_split=0.2, callbacks=[tuner_early_stopping])
            end_time_tuner = time.time()
            tuner_search_time_sec = end_time_tuner - start_time_tuner
            tuner_search_time_min_sec = f"{int(tuner_search_time_sec // 60)}m {int(tuner_search_time_sec % 60)}s"

            run_results_dir_fold = os.path.join(results_base_dir, f"{model_name}_fold{fold_num + 1}_{base_timestamp}")
            os.makedirs(run_results_dir_fold, exist_ok=True)
            tuner_summary_path = os.path.join(run_results_dir_fold, 'tuner_results_summary.txt')
            f_io_tuner = io.StringIO();
            redirect_stdout(f_io_tuner);
            tuner.results_summary(num_trials=10);
            summary_str_tuner = f_io_tuner.getvalue()
            print(summary_str_tuner)
            with open(tuner_summary_path, 'w', encoding='utf-8') as file_tuner:
                file_tuner.write(summary_str_tuner)

            best_hps_retrieved = tuner.get_best_hyperparameters(num_trials=1)
            if best_hps_retrieved:
                best_hps_tuner = best_hps_retrieved[0]
                best_hyperparameters_dict = best_hps_tuner.values
                final_model_to_train = hypermodel.build(best_hps_tuner)
                final_train_epochs = best_hyperparameters_dict.get('epochs', tuner_search_fixed_epochs)
                final_train_batch_size = best_hyperparameters_dict.get('batch_size', tuner_search_batch_size)

                final_model_early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy' if X_train_resampled.shape[0] * 0.1 >= 10 else 'accuracy',
                    patience=early_stopping_patience_final, verbose=1, restore_best_weights=True)
                final_model_val_split = 0.1 if X_train_resampled.shape[0] >= 20 else 0.0

                start_time_final_train = time.time()
                history_final_model = final_model_to_train.fit(
                    X_train_resampled, y_train_eval_target, epochs=final_train_epochs,
                    batch_size=final_train_batch_size,
                    validation_split=final_model_val_split if final_model_val_split > 0 else None,
                    callbacks=[final_model_early_stopping], verbose=1)
                end_time_final_train = time.time()
                final_model_train_time_sec = end_time_final_train - start_time_final_train
                final_model_train_time_min_sec = f"{int(final_model_train_time_sec // 60)}m {int(final_model_train_time_sec % 60)}s"
                best_model_fold = final_model_to_train
                if history_final_model and hasattr(final_model_early_stopping,
                                                   'stopped_epoch') and final_model_early_stopping.stopped_epoch > 0:
                    actual_epochs_trained_final = final_model_early_stopping.stopped_epoch + 1
                elif history_final_model:
                    actual_epochs_trained_final = len(history_final_model.history['loss'])
            else:  # Fallback
                tuner_best_models = tuner.get_best_models(num_models=1)
                if tuner_best_models: best_model_fold = tuner_best_models[0]
        except Exception as e_tuner:
            print(
                f"An error occurred during hyperparameter search or final model training for {model_name}, Fold {fold_num + 1}: {e_tuner}")
            import traceback;
            traceback.print_exc()
    else:
        print(f"Skipping Keras Tuner and final training for Fold {fold_num + 1}: insufficient/invalid data")

    # --- 4. ประเมินผลโมเดลที่ดีที่สุดบน Test Set ของ Fold นี้ ---
    loss, accuracy = np.nan, np.nan
    f1_weighted, balanced_acc, roc_auc_metric = np.nan, np.nan, np.nan
    class_report_str = "Evaluation skipped or failed."
    conf_matrix_list = None
    y_true_classes_test_fold = y_test_fold

    y_test_target_eval_fold = y_test_fold
    if n_classes_model_output > 1 and y_test_fold.size > 0:
        y_test_target_eval_fold = tf.keras.utils.to_categorical(y_test_fold, num_classes=n_classes_model_output)

    if best_model_fold is not None and X_test_fold_scaled.shape[0] > 0 and y_test_fold.shape[0] > 0:
        print(f"Evaluating on Test Set for Fold {fold_num + 1}. X_test_fold_scaled: {X_test_fold_scaled.shape}")
        try:
            loss, accuracy = best_model_fold.evaluate(X_test_fold_scaled, y_test_target_eval_fold, verbose=0)
            y_pred_proba_test = best_model_fold.predict(X_test_fold_scaled)

            if n_classes_model_output > 1:
                y_pred_classes_test = np.argmax(y_pred_proba_test, axis=1)
                y_test_roc_one_hot = tf.keras.utils.to_categorical(y_true_classes_test_fold,
                                                                   num_classes=actual_num_distinct_classes_overall)
                if y_test_roc_one_hot.shape[0] > 0 and y_pred_proba_test.shape[0] > 0 and y_pred_proba_test.shape[
                    1] == actual_num_distinct_classes_overall:
                    roc_auc_metric = roc_auc_score(y_test_roc_one_hot, y_pred_proba_test, average='weighted',
                                                   multi_class='ovr')
                elif y_test_roc_one_hot.shape[0] > 0 and y_pred_proba_test.shape[0] > 0 and y_pred_proba_test.shape[
                    1] == 1 and actual_num_distinct_classes_overall == 2:  # Binary case handled as 1 neuron
                    roc_auc_metric = roc_auc_score(y_true_classes_test_fold, y_pred_proba_test.flatten())
            else:  # Sigmoid
                y_pred_classes_test = (y_pred_proba_test > 0.5).astype(int).flatten()
                if y_true_classes_test_fold.size > 0 and y_pred_proba_test.size > 0:
                    roc_auc_metric = roc_auc_score(y_true_classes_test_fold, y_pred_proba_test.flatten())

            report_labels_indices = range(
                actual_num_distinct_classes_overall) if actual_num_distinct_classes_overall > 0 else None
            report_target_names = class_names_overall if class_names_overall and len(
                class_names_overall) == actual_num_distinct_classes_overall else [f"C_{i}" for i in
                                                                                  report_labels_indices or []]

            f1_weighted = f1_score(y_true_classes_test_fold, y_pred_classes_test, average='weighted', zero_division=0,
                                   labels=report_labels_indices)
            balanced_acc = balanced_accuracy_score(y_true_classes_test_fold, y_pred_classes_test)
            class_report_str = classification_report(y_true_classes_test_fold, y_pred_classes_test,
                                                     target_names=report_target_names, zero_division=0,
                                                     labels=report_labels_indices)
            conf_matrix_val_test = confusion_matrix(y_true_classes_test_fold, y_pred_classes_test,
                                                    labels=report_labels_indices)
            conf_matrix_list = conf_matrix_val_test.tolist() if conf_matrix_val_test is not None else None
        except Exception as e_eval:
            print(f"Error during test set evaluation for {model_name}, Fold {fold_num + 1}: {e_eval}")
            import traceback;
            traceback.print_exc()

    # --- 5. เตรียมไดเรกทอรี และ บันทึกผลลัพธ์ของ Fold นี้ ---
    run_results_dir_fold_path = os.path.join(results_base_dir, f"{model_name}_fold{fold_num + 1}_{base_timestamp}")
    model_run_name_fold_path_segment = f"Lstm_{model_name}_fold{fold_num + 1}_{base_timestamp}"
    run_models_dir_fold_path = os.path.join(models_base_dir, model_run_name_fold_path_segment)
    os.makedirs(run_results_dir_fold_path, exist_ok=True)
    if best_model_fold is not None: os.makedirs(run_models_dir_fold_path, exist_ok=True)

    hyperparameters_used_log_fold = {
        'fold_number': fold_num + 1, 'model_name': model_name,
        'csv_file_path': csv_file_path_for_logging,  # Log original CSV path
        'label_column_name': label_column_name, 'features_used': feature_columns_for_logging,  # Log original features
        'sequence_length': sequence_length, 'n_features_in_sequence': n_features,
        'n_output_neurons_model': n_classes_model_output,
        'actual_distinct_classes_data': actual_num_distinct_classes_overall,
        'class_names_used': class_names_overall,
        'oversampling_method': 'SMOTE' if X_train_resampled.shape[0] > X_train_fold_scaled.shape[
            0] and X_train_fold_scaled.size > 0 else 'None',
        'best_model_hyperparameters_from_tuner': best_hyperparameters_dict if best_hyperparameters_dict else "N/A",
        'final_model_actual_epochs_trained': actual_epochs_trained_final if actual_epochs_trained_final > 0 else "N/A",
    }
    if tuner is not None and hasattr(tuner, 'oracle'):
        # ... (logging tuner info as before) ...
        hyperparameters_used_log_fold['tuner_summary_info'] = {
            'tuner_type': tuner.__class__.__name__, 'tuner_max_trials': tuner.oracle.max_trials,
            'tuner_objective': str(tuner.oracle.objective),
            'tuner_search_fixed_epochs_per_trial': tuner_search_fixed_epochs,
            'tuner_search_fixed_batch_size_per_trial': tuner_search_batch_size,
            'tuner_search_early_stopping_patience': early_stopping_patience_tuner,
            'best_tuned_epochs_hyperparam_value': best_hyperparameters_dict.get('epochs', 'N/A'),
            'best_tuned_batch_size_hyperparam_value': best_hyperparameters_dict.get('batch_size', 'N/A'),
            'final_model_early_stopping_patience': early_stopping_patience_final,
            'validation_split_tuner_search': 0.2,
            'validation_split_final_model_train': final_model_val_split if 'final_model_val_split' in locals() else "N/A",
        }

    results_summary_log_fold = {
        'Model_Name': model_name, 'Fold_Number': fold_num + 1,
        'Timestamp_Run_Start': base_timestamp,  # Overall run start
        'Timestamp_Fold_Evaluation_End': time.strftime("%Y-%m-%d %H:%M:%S"),
        'Results_Directory_Fold': run_results_dir_fold_path,
        'Model_Directory_Fold': run_models_dir_fold_path if best_model_fold is not None else "N/A",
        'Tuner_Search_Time_Fold': tuner_search_time_min_sec,
        'Final_Model_Training_Time_Fold': final_model_train_time_min_sec,
        'Hyperparameters_Used_Fold': hyperparameters_used_log_fold,
        'Test_Set_Results_Fold': {'Loss': float(loss) if not np.isnan(loss) else None,
                                  'Accuracy': float(accuracy) if not np.isnan(accuracy) else None,
                                  'F1_Weighted': float(f1_weighted) if not np.isnan(f1_weighted) else None,
                                  'Balanced_Accuracy': float(balanced_acc) if not np.isnan(balanced_acc) else None,
                                  'ROC_AUC': float(roc_auc_metric) if not np.isnan(roc_auc_metric) else None,
                                  'Num_Test_Samples_Fold': X_test_fold_scaled.shape[0]},
        'Classification_Report_Fold': class_report_str, 'Confusion_Matrix_Fold': conf_matrix_list}

    json_path = os.path.join(run_results_dir_fold_path, "test_results_summary_fold.json")
    # ... (save json_path, txt_path, cm_img_path for the fold as before, adapting names) ...
    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(results_summary_log_fold, f_json, indent=4, ensure_ascii=False)
    # (Similar adaptation for txt_path and cm_img_path)

    if best_model_fold is not None:
        try:
            tf.saved_model.save(best_model_fold, run_models_dir_fold_path)
            print(f"Model for {model_name}, Fold {fold_num + 1} saved to {run_models_dir_fold_path}")
        except Exception as e_save_model:
            print(f"Error saving model for Fold {fold_num + 1}: {e_save_model}")

    # Return key metrics for aggregation
    return {
        'loss': loss, 'accuracy': accuracy, 'f1_weighted': f1_weighted,
        'balanced_acc': balanced_acc, 'roc_auc': roc_auc_metric,
        'tuner_time': tuner_search_time_sec if tuner_search_time_min_sec != "N/A" else 0,
        'train_time': final_model_train_time_sec if final_model_train_time_min_sec != "N/A" else 0
    }


# --- เริ่มต้นสคริปต์หลัก ---
print("--- 1. Checking GPU Availability ---");
check_gpu()
print("\n--- Setting up Configuration ---")
csv_file_path = "Data/labeled_stress_data.csv"
label_column_name = 'stress'
sequence_length = 30

# Tuner and Training Configuration
TUNER_MAX_TRIALS = 30  # Reduced for K-Fold example, adjust as needed
TUNER_SEARCH_FIXED_EPOCHS = 50
TUNER_SEARCH_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE_TUNER = 10
EARLY_STOPPING_PATIENCE_FINAL_MODEL = 15

columns_EDA = ['EDA_Phasic', 'SCR_Amplitude', 'EDA_Tonic', 'SCR_Onsets',
               'gender', 'bmi', 'sleep', 'type', 'stress', 'id']
columns_HRV = ['HRV_RMSSD', 'HRV_LFHF', 'HRV_SDNN', 'HRV_LF',
               'HRV_HF', 'gender', 'bmi', 'sleep', 'type', 'stress', 'id']
# columns_to_exclude_from_scaling is not directly used in process_single_fold's scaling step
# as X_train_fold / X_test_fold should already be purely numerical features.
# It's logged for information.
columns_to_exclude_from_scaling_logging = [label_column_name, 'id']

results_base_dir = "Results";
models_base_dir = "models"
tuner_base_dir = "keras_tuner_dir";
assets_base_dir = "assets"
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
        else:
            raise FileNotFoundError(f"CSV file not found at {csv_file_path} or {alt_csv_path}")
    base_dataframe_original = pd.read_csv(csv_file_path)
    print("Base dataframe loaded successfully.")
    if 'id' not in base_dataframe_original.columns: raise ValueError("'id' column missing.")
    if label_column_name not in base_dataframe_original.columns: raise ValueError(
        f"Label column '{label_column_name}' missing.")
except Exception as e_load:
    print(f"Error loading CSV: {e_load}");
    import traceback;

    traceback.print_exc();
    exit()

# This dictionary will store aggregated CV results
all_cv_aggregated_results = {}

model_configs = {
    "EDA": {"feature_columns": columns_EDA},
    "HRV": {"feature_columns": columns_HRV}
}

for model_name_key, config in model_configs.items():
    print(f"\n>>> PROCESSING MODEL TYPE: {model_name_key} with K-Fold Cross-Validation ({base_timestamp})")
    current_feature_columns = config["feature_columns"]

    # --- 1. Initial Preprocessing (once per model type before K-Fold) ---
    print(f"\n--- 1.1 Initial Preprocessing for {model_name_key} ---")
    preprocess_obj = preprocess(base_dataframe_original.copy(), current_feature_columns)
    dataframe_processed = preprocess_obj.select_columns()

    # Define mapping path once for the model type
    mapping_path_model_type = os.path.join(assets_base_dir, f'category_mapping_{model_name_key}_{base_timestamp}.json')
    dataframe_processed, mapping_model_type = preprocess_obj.label_encoding(mapping_path=mapping_path_model_type)

    class_names_overall_list = []
    n_classes_model_output_val = 0  # For model's output layer
    actual_num_distinct_classes_val = 0  # For metrics reporting

    if label_column_name in mapping_model_type and mapping_model_type[label_column_name]:
        try:
            class_items = sorted(mapping_model_type[label_column_name].items(), key=lambda item: item[0])
            class_names_overall_list = [name for index, name in class_items]
            actual_num_distinct_classes_val = len(class_names_overall_list)
        except Exception as e:
            print(f"Error processing mapping for label column '{label_column_name}' ({model_name_key}): {e}")

    if actual_num_distinct_classes_val == 0:  # Fallback if mapping failed or label not in mapping
        unique_labels_in_df = dataframe_processed[label_column_name].unique()
        unique_labels_in_df = [lbl for lbl in unique_labels_in_df if pd.notna(lbl)]
        actual_num_distinct_classes_val = len(unique_labels_in_df)
        class_names_overall_list = [str(i) for i in sorted(unique_labels_in_df)]

    if actual_num_distinct_classes_val == 2:  # Binary case
        n_classes_model_output_val = 1  # Use 1 output neuron with sigmoid
        print(
            f"Binary classification detected for {model_name_key}. n_classes_model_output_val set to 1 for sigmoid output.")
    elif actual_num_distinct_classes_val > 2:  # Multi-class case
        n_classes_model_output_val = actual_num_distinct_classes_val
    elif actual_num_distinct_classes_val == 1:
        n_classes_model_output_val = 1  # Or handle as an error/warning
        print(
            f"Warning: Only one class found for '{label_column_name}' in {model_name_key}. Model might not train effectively.")
    else:  # actual_num_distinct_classes_val == 0
        print(f"Error: Could not determine n_classes for '{label_column_name}' in {model_name_key}.")
        all_cv_aggregated_results[model_name_key] = {"Error": "Could not determine n_classes."}
        continue  # Skip to next model type

    print(f"Class Names ({model_name_key}, determined globally): {class_names_overall_list}")
    print(f"Number of Output Neurons (n_classes_model_output) for {model_name_key}: {n_classes_model_output_val}")
    print(f"Actual distinct classes for metrics/reporting ({model_name_key}): {actual_num_distinct_classes_val}")

    preprocess_obj.set_sequence_length(sequence_length)
    X_all_sequences, y_all_sequences = preprocess_obj.create_sequence(label_column_name)

    if X_all_sequences.shape[0] == 0 or y_all_sequences.shape[0] == 0:
        print(f"Error: No sequences created for {model_name_key}. Skipping K-Fold.")
        all_cv_aggregated_results[model_name_key] = {"Error": "No sequences created."}
        continue

    n_features_val = X_all_sequences.shape[-1] if X_all_sequences.ndim == 3 else 0
    print(f"Number of Features per Time Step (n_features_val) for {model_name_key}: {n_features_val}")
    if n_features_val == 0:
        print(f"Error: n_features is 0 for {model_name_key}. Skipping K-Fold.")
        all_cv_aggregated_results[model_name_key] = {"Error": "n_features is 0."}
        continue

    # --- 1.2 K-Fold Cross-Validation Loop ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_metrics_collector = []

    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(X_all_sequences, y_all_sequences)):
        X_train_fold_data, X_test_fold_data = X_all_sequences[train_indices], X_all_sequences[test_indices]
        y_train_fold_data, y_test_fold_data = y_all_sequences[train_indices], y_all_sequences[test_indices]

        try:
            fold_result_metrics = process_single_fold(
                X_train_fold=X_train_fold_data, y_train_fold=y_train_fold_data,
                X_test_fold=X_test_fold_data, y_test_fold=y_test_fold_data,
                fold_num=fold_idx, model_name=model_name_key, base_timestamp=base_timestamp,
                n_features=n_features_val, n_classes_model_output=n_classes_model_output_val,
                class_names_overall=class_names_overall_list,
                actual_num_distinct_classes_overall=actual_num_distinct_classes_val,
                sequence_length=sequence_length, label_column_name=label_column_name,
                results_base_dir=results_base_dir, models_base_dir=models_base_dir,
                tuner_base_dir=tuner_base_dir, assets_base_dir=assets_base_dir,
                tuner_search_fixed_epochs=TUNER_SEARCH_FIXED_EPOCHS,
                tuner_search_batch_size=TUNER_SEARCH_BATCH_SIZE,
                tuner_max_trials=TUNER_MAX_TRIALS,
                early_stopping_patience_tuner=EARLY_STOPPING_PATIENCE_TUNER,
                early_stopping_patience_final=EARLY_STOPPING_PATIENCE_FINAL_MODEL,
                csv_file_path_for_logging=csv_file_path,
                feature_columns_for_logging=current_feature_columns
            )
            if fold_result_metrics:  # Ensure it's not None
                fold_metrics_collector.append(fold_result_metrics)
        except Exception as e_fold_proc:
            print(f"\nError processing Fold {fold_idx + 1} for {model_name_key}: {e_fold_proc}")
            import traceback;

            traceback.print_exc()
            # Append NaNs or skip this fold's results for aggregation
            fold_metrics_collector.append({
                'loss': np.nan, 'accuracy': np.nan, 'f1_weighted': np.nan,
                'balanced_acc': np.nan, 'roc_auc': np.nan,
                'tuner_time': np.nan, 'train_time': np.nan
            })

    # --- 1.3 Aggregate and Log CV Results for the current model_name_key ---
    if fold_metrics_collector:
        aggregated_cv_results = {}
        for metric_key in fold_metrics_collector[0].keys():  # Assuming all dicts have same keys
            valid_values = [fm[metric_key] for fm in fold_metrics_collector if
                            fm and not np.isnan(fm.get(metric_key, np.nan))]
            if valid_values:
                aggregated_cv_results[f'{metric_key}_mean'] = np.mean(valid_values)
                aggregated_cv_results[f'{metric_key}_std'] = np.std(valid_values)
            else:
                aggregated_cv_results[f'{metric_key}_mean'] = np.nan
                aggregated_cv_results[f'{metric_key}_std'] = np.nan
        all_cv_aggregated_results[model_name_key] = aggregated_cv_results
        print(f"\n--- Aggregated CV Results for {model_name_key} ({N_SPLITS} Folds) ---")
        for agg_metric, agg_value in aggregated_cv_results.items():
            print(f"  {agg_metric}: {agg_value:.4f}" if not np.isnan(agg_value) else f"  {agg_metric}: N/A")

        # Save aggregated results to a JSON file
        agg_results_path = os.path.join(results_base_dir,
                                        f"cv_aggregated_results_{model_name_key}_{base_timestamp}.json")
        with open(agg_results_path, 'w', encoding='utf-8') as f_agg:
            json.dump({
                "model_name": model_name_key,
                "n_splits": N_SPLITS,
                "base_timestamp": base_timestamp,
                "aggregated_metrics": aggregated_cv_results,
                "individual_fold_metrics": fold_metrics_collector  # Optional: log individual fold raw metrics too
            }, f_agg, indent=4, ensure_ascii=False)
        print(f"Saved aggregated CV results for {model_name_key} to {agg_results_path}")

    else:  # No fold results collected
        all_cv_aggregated_results[model_name_key] = {"Error": "No fold results were collected."}
        print(f"No fold results collected for {model_name_key}.")

# --- เปรียบเทียบผลลัพธ์สุดท้าย (Aggregated CV Results) ---
print("\n" + "=" * 30 + f" Final K-Fold CV Comparison ({N_SPLITS} Folds) " + "=" * 30)
print(
    f"{'Model':<10} | {'Test Acc (Mean)':<17} | {'Test F1 (Mean)':<16} | {'Test Bal Acc (Mean)':<20} | {'Test ROC AUC (Mean)':<20} | {'Mean Tuner Time (s)':<20} | {'Mean Train Time (s)':<20}")
print("-" * 150)

for model_key, agg_res_val in all_cv_aggregated_results.items():
    if "Error" in agg_res_val:
        print(
            f"{model_key:<10} | {'Error':<17} | {'Error':<16} | {'Error':<20} | {'Error':<20} | {'N/A':<20} | {'N/A':<20}")
    elif isinstance(agg_res_val, dict):
        acc_m = agg_res_val.get('accuracy_mean')
        f1_m = agg_res_val.get('f1_weighted_mean')
        bal_m = agg_res_val.get('balanced_acc_mean')
        roc_m = agg_res_val.get('roc_auc_mean')
        time_tuner_m = agg_res_val.get('tuner_time_mean')  # in seconds
        time_train_m = agg_res_val.get('train_time_mean')  # in seconds


        def format_metric(value):
            return f'{value:.4f}' if value is not None and not np.isnan(value) else 'N/A'


        def format_time(value):  # Convert seconds to m s string for display if desired, or keep as seconds
            if value is not None and not np.isnan(value):
                # return f"{int(value // 60)}m {int(value % 60)}s"
                return f'{value:.2f}s'  # Keep as seconds for this table
            return 'N/A'


        print(f"{model_key:<10} | {format_metric(acc_m):<17} | {format_metric(f1_m):<16} | "
              f"{format_metric(bal_m):<20} | {format_metric(roc_m):<20} | {format_time(time_tuner_m):<20} | {format_time(time_train_m):<20}")
print("\n--- Script Execution Finished ---")