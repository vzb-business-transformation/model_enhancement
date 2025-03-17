import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import category_encoders as ce
import joblib
from sklearn.model_selection import train_test_split
from load_data import train_data

class DataProcessor:
    """
    Class for handling all data processing tasks for circuit prediction.
    """

    def __init__(self, random_state=42):
        """
        Initialize the data processor.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.robust_scaler = None
        self.standard_scaler = None
        self.target_encoder = None
        self.knn_imputer = None

    def load_data(self, df = train_data):
        return df
    # def load_data(self, filepath):
    #     """
    #     Load data from a file.
    #
    #     Args:
    #         filepath (str): Path to the data file
    #
    #     Returns:
    #         pd.DataFrame: Loaded data
    #     """
    #     print(f"Loading data from {filepath}")
    #
    #     # Determine file type and load accordingly
    #     if filepath.endswith('.csv'):
    #         df = pd.read_csv(filepath)
    #     elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
    #         df = pd.read_excel(filepath)
    #     elif filepath.endswith('.parquet'):
    #         df = pd.read_parquet(filepath)
    #     else:
    #         raise ValueError(f"Unsupported file format: {filepath}")
    #
    #     print(f"Loaded data with shape: {df.shape}")
    #     return df

    def get_evenly_distributed_circuits(self, df, is_training=True):
        """
        Create an evenly distributed dataset balancing circuits with and without disconnect order numbers.

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): Whether this is for training or prediction

        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        if not is_training:
            return df.copy()

        required_cols = ['PROD_YR_MTH', 'DISCO_DATE_BCOM', 'DISCO_DATE_ORDERING_STRT', 'DISCO_DURATION',
                         'DISCO_ORD_NUM']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter valid disconnects
        valid_disconnects = df[
            (df['DISCO_DATE_BCOM'].notna()) &
            (df['DISCO_DATE_ORDERING_STRT'].notna()) &
            (df['DISCO_DURATION'] > 0) &
            (df['DISCO_DURATION'] <= 365)  # Filter unreasonable durations
            ].copy()

        print(f'\nFiltered to {len(valid_disconnects)} circuits with valid disconnect data')

        valid_df = valid_disconnects
        # Focus on specific time period
        train_period = valid_df[
            (valid_df['PROD_YR_MTH'] >= 202301) &
            (valid_df['PROD_YR_MTH'] < 202501)
            ]

        # Group by month and disconnect order number presence
        def sample_balanced_circuits(month_group):
            """Sample equally from circuits with and without disconnect order numbers"""
            # Split by disconnect order number presence
            with_order_num = month_group[month_group['DISCO_ORD_NUM'].notna() & (month_group['DISCO_ORD_NUM'] != '')]
            without_order_num = month_group[month_group['DISCO_ORD_NUM'].isna() | (month_group['DISCO_ORD_NUM'] == '')]

            print(
                f"Month {month_group['PROD_YR_MTH'].iloc[0]}: {len(with_order_num)} with order num, {len(without_order_num)} without")

            # Determine sample size (minimum monthly count)
            min_monthly_count = 2000  # Can be adjusted
            samples_per_category = min_monthly_count

            # Sample from each category
            sampled_circuits = []

            # With order numbers
            if len(with_order_num) > 0:
                sample_size = min(samples_per_category, len(with_order_num))
                try:
                    # Try stratified sampling by duration
                    duration_bins = pd.qcut(with_order_num['DISCO_DURATION'], q=5, duplicates='drop')
                    sampled = with_order_num.groupby(duration_bins, observed=True).apply(
                        lambda x: x.sample(
                            n=max(1, int(sample_size * len(x) / len(with_order_num))),
                            random_state=self.random_state
                        )
                    )
                except Exception as e:
                    print(f"Warning: Falling back to random sampling for with_order_num: {str(e)}")
                    sampled = with_order_num.sample(
                        n=sample_size,
                        random_state=self.random_state
                    )
                sampled_circuits.append(sampled)

            # Without order numbers
            if len(without_order_num) > 0:
                sample_size = min(samples_per_category, len(without_order_num))
                sampled = without_order_num.sample(
                    n=sample_size,
                    random_state=self.random_state
                )
                sampled_circuits.append(sampled)

            # Combine samples
            if sampled_circuits:
                return pd.concat(sampled_circuits)
            else:
                return pd.DataFrame()  # Return empty DataFrame if no samples

        # Apply sampling to each month
        evenly_distributed = train_period.groupby('PROD_YR_MTH', group_keys=False).apply(sample_balanced_circuits)

        # Check order number distribution
        with_num = evenly_distributed[
            evenly_distributed['DISCO_ORD_NUM'].notna() & (evenly_distributed['DISCO_ORD_NUM'] != '')].shape[0]
        without_num = evenly_distributed[
            evenly_distributed['DISCO_ORD_NUM'].isna() | (evenly_distributed['DISCO_ORD_NUM'] == '')].shape[0]

        print("\nFinal distribution:")
        print(f"Circuits with disconnect order number: {with_num}")
        print(f"Circuits without disconnect order number: {without_num}")

        # Check monthly distribution
        new_counts = evenly_distributed.groupby('PROD_YR_MTH').size()
        print(f"\nNew counts per month:\n{new_counts}")
        print(f'Total circuits: {len(evenly_distributed)}')

        return evenly_distributed

    def count_unique_vendor(self, vendor_str):
        """
        Count unique vendors in a vendor string.

        Args:
            vendor_str (str): Vendor string

        Returns:
            int: Count of unique vendors
        """
        if not isinstance(vendor_str, str):
            return 0
        vendors = re.findall(r'[^,]+(?:,\s*LLC)?', vendor_str)
        unique_vendors = set()
        for v in vendors:
            v = v.strip()
            if any(x in v for x in ['QWT', 'CTL', 'LV3']):
                unique_vendors.add('LUM')
            else:
                unique_vendors.add(v)
        return len(unique_vendors)

    def engineer_features(self, df):
        """
        Create advanced engineered features to improve model performance.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        print("Engineering features...")
        # Create a copy to avoid modifying the original
        df = df.copy()

        # --- Time-based features ---

        # Basic time features
        df['MONTH'] = df['PROD_YR_MTH'].astype(str).str[-2:].astype(int)
        df['YEAR'] = df['PROD_YR_MTH'].astype(str).str[:4].astype(int)
        df['IS_QUARTER_END'] = df['MONTH'].isin([3, 6, 9, 12]).astype(int)
        df['IS_YEAR_END'] = (df['MONTH'] == 12).astype(int)
        df['VENDOR_COUNT'] = df['VENDOR'].apply(lambda x: self.count_unique_vendor(x)).astype(int)

        # Cyclic encoding of time features
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)

        # Calculate time features if date columns exist
        date_cols = ['CIR_INST_DATE', 'CIR_DISC_DATE', 'DISCO_DATE_BCOM', 'DISCO_DATE_ORDERING_STRT']
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    print(f"Warning: Could not convert {col} to datetime")

        # Calculate circuit age and other time-based features
        if 'CIR_INST_DATE' in df.columns and pd.api.types.is_datetime64_dtype(df['CIR_INST_DATE']):
            # Age of circuit at prediction time
            reference_date = pd.Timestamp.now()
            df['CIRCUIT_AGE_DAYS'] = (reference_date - df['CIR_INST_DATE']).dt.days

            # Create periodic features
            df['INST_MONTH'] = df['CIR_INST_DATE'].dt.month
            df['INST_QUARTER'] = df['CIR_INST_DATE'].dt.quarter
            df['INST_DAY_OF_WEEK'] = df['CIR_INST_DATE'].dt.dayofweek

        # --- Geographic features ---

        # Calculate distances if coordinates are available
        coord_cols = ['FIRST_LAT', 'FIRST_LGNTD', 'SECOND_LAT', 'SECOND_LGNTD']
        if all(col in df.columns for col in coord_cols):
            # Function to calculate distance between coordinates
            def haversine_distance(lat1, lon1, lat2, lon2):
                # Convert decimal degrees to radians
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

                # Haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                c = 2 * np.arcsin(np.sqrt(a))
                r = 6371  # Radius of earth in kilometers
                return c * r

            # Apply distance calculation
            coord_mask = (df['FIRST_LAT'].notna() & df['FIRST_LGNTD'].notna() &
                          df['SECOND_LAT'].notna() & df['SECOND_LGNTD'].notna())

            if coord_mask.sum() > 0:
                df.loc[coord_mask, 'ENDPOINT_DISTANCE_KM'] = haversine_distance(
                    df.loc[coord_mask, 'FIRST_LAT'],
                    df.loc[coord_mask, 'FIRST_LGNTD'],
                    df.loc[coord_mask, 'SECOND_LAT'],
                    df.loc[coord_mask, 'SECOND_LGNTD']
                )

                # Create distance buckets
                if 'ENDPOINT_DISTANCE_KM' in df.columns:
                    try:
                        df['DISTANCE_BUCKET'] = pd.qcut(
                            df['ENDPOINT_DISTANCE_KM'].fillna(df['ENDPOINT_DISTANCE_KM'].median()),
                            q=5,
                            labels=['Very_Close', 'Close', 'Medium', 'Far', 'Very_Far'],
                            duplicates='drop'
                        )
                    except ValueError as e:
                        print(f"Warning: Could not create distance buckets: {str(e)}")
                        # Use a simpler alternative method
                        df['DISTANCE_BUCKET'] = pd.cut(
                            df['ENDPOINT_DISTANCE_KM'].fillna(df['ENDPOINT_DISTANCE_KM'].median()),
                            bins=5,
                            labels=['Very_Close', 'Close', 'Medium', 'Far', 'Very_Far']
                        )

        # --- Speed features ---

        # Speed tiers
        speed_cols = [col for col in df.columns if 'SPEED' in col]
        for col in speed_cols:
            if df[col].nunique() > 1 and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    quintiles = df[col].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
                    quintiles = sorted(list(dict.fromkeys(quintiles)))

                    if len(quintiles) >= 2:
                        bins = [float('-inf')] + quintiles + [float('inf')]
                        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'][:len(bins) - 1]
                        df[f'{col}_TIER'] = pd.cut(
                            df[col],
                            bins=bins,
                            labels=labels,
                            include_lowest=True
                        )
                    else:
                        df[f'{col}_IS_HIGH'] = (df[col] > df[col].median()).astype(int)
                except Exception as e:
                    print(f"Warning: Could not create tiers for {col}: {str(e)}")
                    df[f'{col}_IS_HIGH'] = (df[col] > df[col].median()).astype(int)

        # Speed variations and interactions
        if 'PORT_SPEED' in df.columns and 'ACCESS_SPEED' in df.columns:
            df['SPEED_RATIO'] = df['PORT_SPEED'] / df['ACCESS_SPEED'].replace(0, 1)

            # Create speed variation if multiple speed columns exist
            speed_cols_numeric = [col for col in speed_cols if pd.api.types.is_numeric_dtype(df[col])]
            if len(speed_cols_numeric) > 1:
                df['SPEED_VARIATION'] = df[speed_cols_numeric].std(axis=1) / df[speed_cols_numeric].mean(
                    axis=1).replace(0, 1)

            # Highest speed in the circuit
            df['MAX_SPEED'] = df[speed_cols_numeric].max(axis=1)

            # Speed tiers based on industry standards
            df['SPEED_TIER'] = pd.cut(
                df['MAX_SPEED'],
                bins=[0, 10000, 100000, 1000000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Ultra']
            )

        # --- Disconnect features ---

        # Combine disconnect types into meaningful groups
        disconnect_cols = ['ACCESS_DISC', 'PORT_DISC', 'PVC_CAR_DISC', 'OTHER_DISC']
        if all(col in df.columns for col in disconnect_cols):
            # Total disconnects
            df['TOTAL_DISC'] = df[disconnect_cols].sum(axis=1)
            df['HAS_MULTIPLE_DISC'] = (df['TOTAL_DISC'] > 0).astype(int)

            # Disconnect rates relative to speeds
            if 'ACCESS_SPEED' in df.columns and df['ACCESS_SPEED'].sum() > 0:
                df['ACCESS_DISC_RATE'] = df['ACCESS_DISC'] / df['ACCESS_SPEED'].replace(0, 1) * 100

            if 'PORT_SPEED' in df.columns and df['PORT_SPEED'].sum() > 0:
                df['PORT_DISC_RATE'] = df['PORT_DISC'] / df['PORT_SPEED'].replace(0, 1) * 100

            # Disconnect complexity (number of different disconnect types applied)
            df['DISCONNECT_COMPLEXITY'] = df[disconnect_cols].apply(
                lambda x: sum(1 for val in x if val > 0),
                axis=1
            )

        # --- Interaction features ---

        # Create interaction features between important variables
        if 'MARGIN' in df.columns:
            if 'MAX_SPEED' in df.columns:
                df['MARGIN_SPEED_INTERACTION'] = df['MARGIN'] * np.log1p(df['MAX_SPEED'])

            if 'TOTAL_DISC' in df.columns:
                df['MARGIN_DISCONNECT_RATIO'] = df['MARGIN'] / df['TOTAL_DISC'].replace(0, 1)

        # For categorical interactions
        if 'CIR_TECH_TYPE' in df.columns and 'FIRST_ONNET' in df.columns:
            df['TECH_ONNET_COMBO'] = df['CIR_TECH_TYPE'] + '_' + df['FIRST_ONNET']

        if 'CIR_TECH_TYPE' in df.columns and 'CIR_BILL_TYPE' in df.columns:
            df['TECH_BILL_COMBO'] = df['CIR_TECH_TYPE'] + '_' + df['CIR_BILL_TYPE']

        if 'VRTCL_MRKT_NAME' in df.columns and 'SALES_TIER' in df.columns:
            df['MARKET_TIER_COMBO'] = df['VRTCL_MRKT_NAME'] + '_' + df['SALES_TIER']

        # --- Polynomial features for key numerics ---

        # Add polynomial terms for key numeric features
        numeric_features = ['MARGIN', 'ACCESS_SPEED', 'PORT_SPEED', 'TOTAL_DISC']
        numeric_features = [f for f in numeric_features if f in df.columns]

        for feat in numeric_features:
            df[f'{feat}_SQUARED'] = df[feat] ** 2
            df[f'LOG_{feat}'] = np.log1p(df[feat].clip(lower=0))

        print(f"Feature engineering complete. New shape: {df.shape}")

        return df

    def preprocess_data(self, df, is_training=True, target_col='DISCO_DURATION'):
        """
        Preprocess the data for model training or prediction.

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): Whether this is for training or prediction
            target_col (str): Target column name

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        print("Starting data preprocessing...")

        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()

        # Initial target column check
        if target_col in df.columns:
            print(f"Initial - Target column '{target_col}' exists with {df[target_col].notna().sum()} non-NaN values")
        else:
            print(f"WARNING: Target column '{target_col}' not found in initial dataframe")

        # 1. Distribution and initial filtering
        if is_training:
            df = self.get_evenly_distributed_circuits(df.copy(), is_training=True)
        else:
            df = self.get_evenly_distributed_circuits(df.copy(), is_training=False)

        # Check target after distribution
        if target_col in df.columns:
            print(
                f"After distribution - Target column has {df[target_col].notna().sum()} non-NaN values out of {len(df)} rows")
        else:
            print(f"WARNING: Target column '{target_col}' lost after distribution")

        # 2. Handle speed columns with improved parsing
        speed_cols = [col for col in df.columns if 'SPEED' in col]
        for col in speed_cols:
            if col in df.columns:
                # First convert to string
                df[col] = df[col].astype(str)

                # Handle 'None', 'nan', etc.
                df[col] = df[col].replace(['None', 'nan', 'NaN', 'null', 'NULL', 'NA', ''], '0')

                # Extract numeric components
                def extract_numeric_speed(speed_str):
                    # Remove non-numeric chars except decimal points
                    speed_str = re.sub(r'[^\d.]', '', str(speed_str))
                    if speed_str == '':
                        return 0

                    # Convert to float first to handle decimals
                    try:
                        return float(speed_str)
                    except:
                        return 0

                # Apply numeric extraction
                df[col] = df[col].apply(extract_numeric_speed)

                # Convert units (Kbps, Mbps, Gbps) based on column context
                if 'Kbps' in ''.join(df[col].astype(str).tolist()):
                    # Values are likely in Kbps
                    pass  # Keep as is
                elif 'Gbps' in ''.join(df[col].astype(str).tolist()):
                    # Convert Gbps to Kbps for consistency
                    df[col] = df[col] * 1000000
                else:
                    # Assume Mbps and convert to Kbps
                    df[col] = df[col] * 1000

                # Finally convert to integer
                df[col] = df[col].astype(int)

        # 3. Process circuit type columns
        if 'SR_CRCTP' in df.columns:
            df['SR_CRCTP'] = df['SR_CRCTP'].apply(lambda x: str(x).split('x')[-1])
        if 'RPTG_CRCTP' in df.columns:
            df['RPTG_CRCTP'] = df['RPTG_CRCTP'].apply(lambda x: str(x).split('x')[-1])

        # 4. Add engineered features

        df = self.engineer_features(df)
        print(f"After feature engineering - Target column NaN count: {df[target_col].isna().sum()}")

        # 5. Drop unnecessary columns
        columns_to_drop = [
            'CIR_ID', 'NASP_ID', 'NASP_NM', 'FIRST_CITY', 'FIRST_STATE',
            'FIRST_ZIP', 'DISCO_ORD_NUM', 'DIV_PORT', 'ACCESS_SPEED_UP',
            'DIV_ACCESS', 'CIR_INST_DATE', 'CIR_DISC_DATE', 'MIG_STATUS',
            'FIRST_LATA', 'VENDOR', 'DISCO_DATE_BCOM', 'DISCO_DATE_ORDERING_STRT',
            'SECOND_LAT', 'SECOND_LGNTD'
        ]

        # Only drop columns that exist
        df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True)

        # 6. Handle missing values in categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                # For categorical columns with missing values
                mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_value)

        # 7. Advanced categorical encoding for high-cardinality features
        if is_training and target_col in df.columns:
            high_cardinality_cols = [
                col for col in categorical_cols
                if col in df.columns and df[col].nunique() > 10
            ]

            if high_cardinality_cols:
                self.target_encoder = ce.TargetEncoder(cols=high_cardinality_cols)
                # Fit and transform
                df_encoded = self.target_encoder.fit_transform(
                    df[high_cardinality_cols],
                    df[target_col]
                )
                # Replace original columns
                for col in high_cardinality_cols:
                    df[col] = df_encoded[col]
        elif not is_training and hasattr(self, 'target_encoder') and self.target_encoder is not None:
            # Apply existing target encoder in prediction mode
            high_cardinality_cols = self.target_encoder.cols
            df_encoded = self.target_encoder.transform(df[high_cardinality_cols])
            for col in high_cardinality_cols:
                df[col] = df_encoded[col]

        # 8. One-hot encoding - CRITICAL POINT
        print(f"Before one-hot encoding - Target column exists: {target_col in df.columns}")
        if target_col in df.columns:
            print(f"  Target column type: {df[target_col].dtype}")
            print(f"  Target column non-NaN values: {df[target_col].notna().sum()}")
            # Save target column separately
            target_values = df[target_col].copy()

        # Apply one-hot encoding more carefully
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]

        print(f"Categorical columns to encode: {list(categorical_cols)}")


        if len(categorical_cols) > 0:
            # Encode only categorical columns
            encoded_cats = pd.get_dummies(df[categorical_cols], drop_first=True)

            # Drop original categorical columns
            df = df.drop(columns=categorical_cols)

            # Join encoded columns
            df = pd.concat([df, encoded_cats], axis=1)

        # Verify target column after encoding
        if target_col not in df.columns:
            print(f"Target column lost after encoding, restoring it")
            df[target_col] = target_values

        print(f"After one-hot encoding - Target column exists: {target_col in df.columns}")
        if target_col in df.columns:
            print(f"  Target column non-NaN values: {df[target_col].notna().sum()}")

        # 9. Handle outliers with robust scaling instead of removal
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if is_training:
            self.robust_scaler = RobustScaler()
            df[numeric_cols] = self.robust_scaler.fit_transform(df[numeric_cols])
        else:
            # In prediction mode, use the pre-fit scaler
            if hasattr(self, 'robust_scaler') and self.robust_scaler is not None:
                df[numeric_cols] = self.robust_scaler.transform(df[numeric_cols])
            else:
                print("Warning: RobustScaler not fitted. Skipping transformation.")


        # 10. KNN imputation for remaining missing values
        if is_training:
            # Select numeric columns excluding target
            numeric_cols_to_impute = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if
                                      col != target_col]

            # Save target column
            target_values = df[target_col].copy()

            # Impute other numeric columns
            if numeric_cols_to_impute:
                self.knn_imputer = KNNImputer(n_neighbors=5)
                df_imputed = df[numeric_cols_to_impute].copy()
                df_imputed_values = self.knn_imputer.fit_transform(df_imputed)

                # Convert back to DataFrame and update original
                df_imputed = pd.DataFrame(df_imputed_values, columns=numeric_cols_to_impute, index=df.index)
                for col in numeric_cols_to_impute:
                    df[col] = df_imputed[col]

            # Restore target column
            df[target_col] = target_values

        # 11. Convert all columns to float32 for efficiency
        target_values = df[target_col].copy()

        # Convert other columns
        for col in df.columns:
            if col != target_col:  # Skip target column
                df[col] = df[col].astype(np.float32)

        # Restore target column separately to ensure it's preserved exactly
        df[target_col] = target_values

        print(f"Preprocessing complete. Final shape: {df.shape}")
        print(f"After distribution - Target column NaN count: {df[target_col].isna().sum()}")
        print(f"Final preprocessed data - Target column exists: {target_col in df.columns}")
        if target_col in df.columns:
            print(f"  Target column non-NaN values: {df[target_col].notna().sum()} out of {len(df)} rows")

        return df

    def prepare_train_test_data(self, df, target_col, test_size=0.2):
        """
        Prepare training and testing data.

        Args:
            df (pd.DataFrame): Input DataFrame
            target_col (str): Target column name
            test_size (float): Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
        """
        print("Preparing train/test split...")

        # Split features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Print target column info
        print(f"Target column '{target_col}' stats:")
        print(f"  - Data type: {df[target_col].dtype}")
        print(f"  - NaN count: {df[target_col].isna().sum()}")
        print(f"  - Total rows: {len(df)}")

        if df[target_col].notna().sum() > 0:
            print(f"  - Min value: {df[target_col].min()}")
            print(f"  - Max value: {df[target_col].max()}")
            print(f"  - Mean value: {df[target_col].mean()}")

        # Check if all values are NaN
        if df[target_col].isna().all():
            raise ValueError(f"All values in target column '{target_col}' are NaN. Please check your data.")

        # Check for NaN values in target column
        nan_count = df[target_col].isna().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in target column '{target_col}'. Dropping these rows.")
            df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Create a stratified split based on target quantiles
        try:
            target_bins = pd.qcut(y, q=5, duplicates='drop')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=target_bins
            )
        except:
            # Fallback to regular split if stratification fails
            print("Warning: Stratified split failed. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        # Scale data
        self.standard_scaler = StandardScaler()
        X_train_scaled = self.standard_scaler.fit_transform(X_train)
        X_test_scaled = self.standard_scaler.transform(X_test)

        print(f"Train set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def save_preprocessor(self, filepath="results/models/data_processor.joblib"):
        """
        Save the preprocessor for later use.

        Args:
            filepath (str): Path to save the preprocessor
        """


        # Create a dictionary of fitted components
        preprocessor_data = {
            'robust_scaler': self.robust_scaler,
            'standard_scaler': self.standard_scaler,
            'target_encoder': self.target_encoder,
            'knn_imputer': self.knn_imputer,
            'random_state': self.random_state
        }

        # Save to file
        joblib.dump(preprocessor_data, filepath)
        print(f"Data processor saved to {filepath}")

    def load_preprocessor(self, filepath="results/models/data_processor.joblib"):
        """
        Load a saved preprocessor.

        Args:
            filepath (str): Path to the saved preprocessor
        """


        # Load from file
        preprocessor_data = joblib.load(filepath)

        # Set preprocessor components
        self.robust_scaler = preprocessor_data['robust_scaler']
        self.standard_scaler = preprocessor_data['standard_scaler']
        self.target_encoder = preprocessor_data['target_encoder']
        self.knn_imputer = preprocessor_data['knn_imputer']
        self.random_state = preprocessor_data['random_state']

        print(f"Data processor loaded from {filepath}")