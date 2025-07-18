# Import necessary libraries with error handling
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.impute import SimpleImputer, KNNImputer
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    print(f"Analysis run on: 2025-04-17 16:42:50 UTC")
except ImportError as e:
    print(f"Error: {e}")
    print("Please install missing packages using:")
    print("pip install pandas numpy matplotlib seaborn scipy scikit-learn")
    import sys
    sys.exit(1)

# Load the data (assuming the data is in a CSV file)
df = pd.read_csv('Combined/eda_ppg/combined_eda_ppg_data_20250417_220939.csv')

# Convert DateTime to proper datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

#-------- 1. Basic Data Examination --------#
print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Print statistics about missing values
print("\nMissing values summary:")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Percentage of missing data: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")

# Data types
print("\nData types:")
print(df.dtypes)

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

#-------- 2. EDA Feature Category Analysis --------#
# Group features by category
eda_features = [col for col in df.columns if col.startswith('EDA_')]
scr_features = [col for col in df.columns if col.startswith('SCR_')]
ppg_features = [col for col in df.columns if col.startswith('PPG_')]
hrv_features = [col for col in df.columns if col.startswith('HRV_')]

print("\nEDA features:", eda_features)
print("\nSCR features:", scr_features)
print("\nPPG features:", ppg_features)
print(f"\nHRV features: {len(hrv_features)} features")  # There are many HRV features

#-------- 3. Distribution Analysis --------#
# Create a function to plot histograms for key features
def plot_distributions(df, features, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    for i, feature in enumerate(features):
        if feature in df.columns:  # Check if feature exists
            plt.subplot(len(features)//3 + 1, 3, i+1)
            sns.histplot(df[feature].dropna(), kde=True)
            plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

# Select key features from each category - check they exist in the dataframe
all_candidate_features = ['EDA_Phasic', 'SCR_Amplitude', 'NumPeaks', 'PPG_Rate', 
                        'HRV_SDNN', 'HRV_RMSSD', 'HRV_LFHF', 'HR']
key_features = [f for f in all_candidate_features if f in df.columns]

if key_features:
    plot_distributions(df, key_features)
else:
    print("None of the key features were found in the dataframe.")

#-------- 4. Correlation Analysis --------#
# Calculate correlations for key physiological measures
candidate_selected_features = key_features + ['RMSSD', 'SDNN', 'RR_Mean', 
                                'HRV_SD1', 'HRV_SD2', 'HRV_SampEn', 
                                'HRV_DFA_alpha1']

# Filter to include only features that exist in the dataframe
selected_features = [f for f in candidate_selected_features if f in df.columns]

# If you have a stress_level column, include it
if 'stress_level' in df.columns:
    selected_features.append('stress_level')

if selected_features:
    # Replace missing values with mean for correlation analysis
    df_corr = df[selected_features].copy()
    
    # Calculate means for each column
    column_means = df_corr.mean()
    
    # Fill missing values with respective column means
    df_corr = df_corr.fillna(column_means)
    
    # Correlation matrix
    corr_matrix = df_corr.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Key Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("No selected features were found in the dataframe for correlation analysis.")

#-------- 5. Gender, BMI, and Sleep Analysis --------#
# Check demographic patterns if available
if 'gender' in df.columns:
    print("\nGender distribution:")
    print(df['gender'].value_counts())
    
    # Compare physiological measures by gender
    gender_features = [f for f in key_features[:6] if f in df.columns]
    if gender_features:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(gender_features):
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='gender', y=feature, data=df)
            plt.title(f'{feature} by Gender')
        plt.tight_layout()
        plt.show()

if 'bmi_category' in df.columns:
    print("\nBMI category distribution:")
    print(df['bmi_category'].value_counts())
    
    # BMI category vs key measures
    bmi_features = [f for f in key_features[:6] if f in df.columns]
    if bmi_features:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(bmi_features):
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='bmi_category', y=feature, data=df)
            plt.title(f'{feature} by BMI Category')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if 'sleep' in df.columns:
    # Check if sleep column is numeric or can be converted to numeric
    try:
        sleep_numeric = pd.to_numeric(df['sleep'], errors='coerce')
        if not sleep_numeric.isna().all():
            print("\nSleep statistics:")
            print(sleep_numeric.describe())
            
            # Scatter plots of sleep vs key features
            sleep_features = [f for f in key_features[:6] if f in df.columns]
            if sleep_features:
                plt.figure(figsize=(15, 10))
                for i, feature in enumerate(sleep_features):
                    plt.subplot(2, 3, i+1)
                    sns.scatterplot(x=sleep_numeric, y=df[feature])
                    plt.title(f'{feature} vs Sleep')
                plt.tight_layout()
                plt.show()
        else:
            print("\nSleep column exists but cannot be converted to numeric format.")
    except:
        print("\nSleep column exists but is not numeric.")
        print(df['sleep'].value_counts())

#-------- 6. Principal Component Analysis --------#
# Prepare data for PCA
# Select numerical columns only (excluding DateTime and categorical variables)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove ID column if it exists
if 'id' in numerical_cols:
    numerical_cols.remove('id')

# Create a copy of the data for PCA
df_pca = df[numerical_cols].copy()

# Check percentage of missing values per column
missing_percent = df_pca.isnull().mean() * 100
print("\nPercentage of missing values per column (top 10):")
print(missing_percent[missing_percent > 0].sort_values(ascending=False).head(10))

# Option 1: Remove columns with too many missing values (e.g., >30%)
threshold = 30.0  # adjust as needed
high_missing_cols = missing_percent[missing_percent > threshold].index.tolist()
if high_missing_cols:
    print(f"\nRemoving {len(high_missing_cols)} columns with >{threshold}% missing values")
    df_pca = df_pca.drop(columns=high_missing_cols)

# Option 2: Impute remaining missing values
# For fewer missing values, KNN imputation can give better results
# For more missing values, SimpleImputer might be more appropriate
if df_pca.isnull().sum().sum() / (df_pca.shape[0] * df_pca.shape[1]) < 0.2:
    print("\nPerforming KNN imputation for missing values...")
    imputer = KNNImputer(n_neighbors=5)
else:
    print("\nPerforming mean imputation for missing values...")
    imputer = SimpleImputer(strategy='mean')

# Apply imputation
df_pca_imputed = pd.DataFrame(
    imputer.fit_transform(df_pca),
    columns=df_pca.columns
)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_pca_imputed)

# Apply PCA
pca = PCA()
try:
    pca_result = pca.fit_transform(scaled_data)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Components')
    plt.grid(True)
    plt.show()

    # Plot first two principal components
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Result')
    plt.show()

    # If you have type information, color by that
    if 'type' in df.columns:
        plt.figure(figsize=(10, 8))
        # Create a smaller sample if the dataset is large
        sample_size = min(10000, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        
        sns.scatterplot(
            x=pca_result[sample_indices, 0], 
            y=pca_result[sample_indices, 1], 
            hue=df['type'].iloc[sample_indices]
        )
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Result by Type')
        plt.show()
        
except Exception as e:
    print(f"PCA failed with error: {e}")
    print("Trying robust PCA with more aggressive filtering...")
    
    # More aggressive approach - keep only columns with less than 5% missing values
    low_missing_cols = missing_percent[missing_percent < 5].index.tolist()
    if low_missing_cols:
        print(f"Performing PCA with {len(low_missing_cols)} columns having <5% missing values")
        df_pca_strict = df[low_missing_cols].copy()
        
        # Simple mean imputation
        imputer_strict = SimpleImputer(strategy='mean')
        df_pca_strict_imputed = pd.DataFrame(
            imputer_strict.fit_transform(df_pca_strict),
            columns=df_pca_strict.columns
        )
        
        # Standardize and apply PCA
        scaler_strict = StandardScaler()
        scaled_data_strict = scaler_strict.fit_transform(df_pca_strict_imputed)
        
        pca_strict = PCA()
        pca_result_strict = pca_strict.fit_transform(scaled_data_strict)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca_strict.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Components (Robust PCA)')
        plt.grid(True)
        plt.show()
        
        # Plot first two principal components
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result_strict[:, 0], pca_result_strict[:, 1], alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Robust PCA Result')
        plt.show()

#-------- 7. Feature Importance Analysis --------#
# If you have a target variable (stress_level), you can check feature importance
if 'stress_level' in df.columns:
    # Prepare data and handle missing values
    X = df[numerical_cols].drop('stress_level', axis=1) if 'stress_level' in numerical_cols else df[numerical_cols]
    
    # Use a robust imputer for X
    imp = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    
    y = df['stress_level'].copy()
    
    # Handle missing values in y if any
    if y.isnull().any():
        # For numeric target, impute with mean
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.mean())
        # For categorical target, use mode
        else:
            y = y.fillna(y.mode()[0])
    
    try:
        # Use SelectKBest for feature importance
        selector = SelectKBest(f_regression, k=min(20, X_imputed.shape[1]))
        selector.fit(X_imputed, y)
        
        # Get feature importance scores
        feature_scores = pd.DataFrame({
            'Feature': X_imputed.columns,
            'Score': selector.scores_
        })
        
        # Sort by importance
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Plot top 20 important features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Score', y='Feature', data=feature_scores.head(20))
        plt.title('Top 20 Important Features for Stress Level')
        plt.tight_layout()
        plt.show()
        
        print("\nTop 20 important features:")
        print(feature_scores.head(20))
    except Exception as e:
        print(f"Feature importance analysis failed: {e}")

#-------- 8. Time Series Analysis (if applicable) --------#
if 'DateTime' in df.columns:
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Set DateTime as index
    df_ts = df.copy()
    df_ts.set_index('DateTime', inplace=True)
    
    # Select key features for time series analysis
    ts_features = [f for f in key_features if f in df_ts.columns]
    
    if ts_features:
        # Handle missing values in time series with forward fill
        df_ts_filled = df_ts[ts_features].fillna(method='ffill').fillna(method='bfill')
        
        # Plot time series for key features
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(ts_features[:min(4, len(ts_features))]):
            plt.subplot(min(4, len(ts_features)), 1, i+1)
            df_ts_filled[feature].plot()
            plt.title(f'Time Series of {feature}')
        plt.tight_layout()
        plt.show()
        
        # Resample to detect patterns (optional)
        print("\nDaily aggregation of key features:")
        daily_agg = df_ts_filled[ts_features].resample('D').mean()
        print(daily_agg.head())
        
        # Plot daily aggregation
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(ts_features[:min(4, len(ts_features))]):
            plt.subplot(min(4, len(ts_features)), 1, i+1)
            daily_agg[feature].plot()
            plt.title(f'Daily Average of {feature}')
        plt.tight_layout()
        plt.show()

#-------- 9. Relationship between HRV metrics --------#
# Select some important HRV metrics
hrv_key_metrics_candidates = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SampEn']
hrv_key_metrics = [metric for metric in hrv_key_metrics_candidates if metric in df.columns]

if len(hrv_key_metrics) >= 2:  # Need at least 2 metrics for pairplot
    # Create a dataframe with only the selected metrics and handle missing values
    df_hrv = df[hrv_key_metrics].copy()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_hrv_imputed = pd.DataFrame(
        imputer.fit_transform(df_hrv),
        columns=df_hrv.columns
    )
    
    # Create a pairplot for these metrics
    # Use a smaller sample if the dataset is large
    if len(df_hrv_imputed) > 5000:
        print("\nSampling 5000 rows for HRV pairplot due to large dataset size")
        df_hrv_sample = df_hrv_imputed.sample(5000, random_state=42)
        sns.pairplot(df_hrv_sample)
    else:
        sns.pairplot(df_hrv_imputed)
    
    plt.suptitle('Relationships Between Key HRV Metrics', y=1.02)
    plt.show()
else:
    print("\nNot enough HRV metrics found in the dataset for pairplot analysis")

#-------- 10. Identify Outliers --------#
# Select key features that exist in the data
outlier_features = [f for f in key_features if f in df.columns]

if outlier_features:
    # Create a copy and fill missing values for outlier detection
    df_outliers = df[outlier_features].copy()
    df_outliers = df_outliers.fillna(df_outliers.mean())
    
    # Z-score method for outlier detection
    z_scores = stats.zscore(df_outliers)
    abs_z_scores = np.abs(z_scores)
    outliers = (abs_z_scores > 3).any(axis=1)
    print(f"\nNumber of outliers detected: {sum(outliers)} ({sum(outliers)/len(df)*100:.2f}% of data)")
    
    # Plot boxplots to visualize outliers
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_outliers)
    plt.title('Boxplot for Key Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary of outliers per feature
    print("\nOutliers per feature (z-score > 3):")
    for feature in outlier_features:
        feature_z = np.abs(stats.zscore(df_outliers[feature]))
        n_outliers = sum(feature_z > 3)
        print(f"{feature}: {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
        
    # Save outlier indices for potential removal in future analyses
    outlier_indices = np.where(outliers)[0]
    print(f"\nFirst 10 outlier indices: {outlier_indices[:10]}")
    
    # Optional: Create a cleaned dataset without outliers
    df_no_outliers = df[~outliers].copy()
    print(f"\nShape after outlier removal: {df_no_outliers.shape} (removed {len(df) - len(df_no_outliers)} rows)")
else:
    print("\nNo key features found for outlier detection")

print("\nData exploration completed successfully!")