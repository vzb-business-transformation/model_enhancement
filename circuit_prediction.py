class CircuitPredictionPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_models = {}
        self.feature_names = {}
        self.feature_importance = {}
        self.xgb_feature_importance = None

    def get_evenly_distributed_circuits(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create an evenly distributed dataset balancing circuits with and without disconnect order numbers

        Args:
            df: DataFrame
            is_training: if this is for training (True) or prediction (False)
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
        # valid_actives = df[
        #     (df['DISCO_DATE_BCOM'].isna()) &
        #     (df['DISCO_DATE_ORDERING_STRT'].isna()) &
        #     (df['DISCO_ORD_NUM'].isna())
        # ].copy()

        print(f'\nFiltered to {len(valid_disconnects)} circuits with valid disconnect data')
        # print(f'\nFiltered to {len(valid_actives)} circuits with valid active data')

        # valid_df = pd.concat([valid_disconnects, valid_actives])
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
            without_order_num = month_group[month_group['DISCO_ORD_NUM'].isna()]

            print(
                f"Month {month_group['PROD_YR_MTH'].iloc[0]}: {len(with_order_num)} with order num, {len(without_order_num)} without")

            # Determine sample size (half of minimum monthly count)
            monthly_counts = train_period.groupby('PROD_YR_MTH').size()
            min_monthly_count = 2000  # monthly_counts.min()
            # samples_per_category = min_monthly_count // 2  # Half for each category
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

    def handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[
                    (df[col] >= Q1 - 1.5 * IQR) &
                    (df[col] <= Q3 + 1.5 * IQR)
                    ]
        return df

    def count_unique_vendor(self, vendor_str):
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

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to improve model performance"""
        # Time-based features
        df['MONTH'] = df['PROD_YR_MTH'].astype(str).str[-2:].astype(int)
        df['YEAR'] = df['PROD_YR_MTH'].astype(str).str[:4].astype(int)
        df['IS_QUARTER_END'] = df['MONTH'].isin([3, 6, 9, 12]).astype(int)
        df['IS_YEAR_END'] = (df['MONTH'] == 12).astype(int)
        df['VENDOR_COUNT'] = df['VENDOR'].apply(lambda x: self.count_unique_vendor(x)).astype(int)

        # Speed features
        for col in df.columns:
            if 'SPEED' in col:
                df[col] = df[col].astype(int)
                if df[col].nunique() > 1:
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

        # Speed ratios and interactions
        df['SPEED_RATIO'] = df['PORT_SPEED'] / df['ACCESS_SPEED'].replace(0, 1)
        df['TOTAL_DISC'] = df['ACCESS_DISC'] + df['PORT_DISC'] + df['PVC_CAR_DISC'] + df['OTHER_DISC']
        df['HAS_MULTIPLE_DISC'] = (df['TOTAL_DISC'] > 0).astype(int)
        df['SPEED_DISC_INTERACTION'] = df['TOTAL_DISC'] * df['ACCESS_SPEED']

        # Categorical combinations
        if 'CIR_TECH_TYPE' in df.columns and 'CIR_BILL_TYPE' in df.columns:
            df['TECH_BILL_COMBO'] = df['CIR_TECH_TYPE'] + '_' + df['CIR_BILL_TYPE']

        return df

    def enhanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features to improve model performance"""
        # Keep original engineered features
        df = self.add_engineered_features(df.copy())

        # --- Time-based features ---

        # Convert date columns to datetime if they exist
        date_cols = ['CIR_INST_DATE', 'CIR_DISC_DATE', 'DISCO_DATE_BCOM', 'DISCO_DATE_ORDERING_STRT']
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    print(f"Warning: Could not convert {col} to datetime")

        # Calculate time features
        if 'CIR_INST_DATE' in df.columns and pd.api.types.is_datetime64_dtype(df['CIR_INST_DATE']):
            # Age of circuit at prediction time
            reference_date = pd.Timestamp.now()
            df['CIRCUIT_AGE_DAYS'] = (reference_date - df['CIR_INST_DATE']).dt.days

            # Create periodic features (captures seasonality)
            df['INST_MONTH'] = df['CIR_INST_DATE'].dt.month
            df['INST_QUARTER'] = df['CIR_INST_DATE'].dt.quarter
            df['INST_DAY_OF_WEEK'] = df['CIR_INST_DATE'].dt.dayofweek

            # Cyclic encoding of time features (better captures periodicity)
            df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
            df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
            df['QUARTER_SIN'] = np.sin(2 * np.pi * df['INST_QUARTER'] / 4)
            df['QUARTER_COS'] = np.cos(2 * np.pi * df['INST_QUARTER'] / 4)

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
                    df['DISTANCE_BUCKET'] = pd.qcut(
                        df['ENDPOINT_DISTANCE_KM'].fillna(df['ENDPOINT_DISTANCE_KM'].median()),
                        q=5,
                        labels=['Very_Close', 'Close', 'Medium', 'Far', 'Very_Far'],
                        duplicates='drop'
                    )

        # --- Enhanced financial features ---

        # Combine disconnect types into meaningful groups
        disconnect_cols = ['ACCESS_DISC', 'PORT_DISC', 'PVC_CAR_DISC', 'OTHER_DISC']
        if all(col in df.columns for col in disconnect_cols):
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

        # --- Customer segmentation features ---

        # If we have a customer identifier, create aggregation features
        if 'NASP_ID' in df.columns:
            # Count circuits per customer
            customer_circuit_counts = df.groupby('NASP_ID').size()
            df['CUSTOMER_CIRCUIT_COUNT'] = df['NASP_ID'].map(customer_circuit_counts)

            # Customer average metrics
            if 'MARGIN' in df.columns:
                customer_avg_margin = df.groupby('NASP_ID')['MARGIN'].mean()
                df['CUSTOMER_AVG_MARGIN'] = df['NASP_ID'].map(customer_avg_margin)
                df['MARGIN_VS_CUSTOMER_AVG'] = df['MARGIN'] - df['CUSTOMER_AVG_MARGIN']

        # --- Circuit complexity features ---

        # Create features representing circuit complexity
        speed_cols = [col for col in df.columns if 'SPEED' in col]
        if len(speed_cols) > 0:
            # Speed variation within circuit
            speed_cols_numeric = [col for col in speed_cols if pd.api.types.is_numeric_dtype(df[col])]
            if len(speed_cols_numeric) > 1:
                df['SPEED_VARIATION'] = df[speed_cols_numeric].std(axis=1) / df[speed_cols_numeric].mean(
                    axis=1).replace(0, 1)

            # Highest speed in the circuit
            df['MAX_SPEED'] = df[speed_cols_numeric].max(axis=1)

            # Speed tiers based on industry standards (arbitrary thresholds - should be adjusted)
            df['SPEED_TIER'] = pd.cut(
                df['MAX_SPEED'],
                bins=[0, 10000, 100000, 1000000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Ultra']
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

        if 'VRTCL_MRKT_NAME' in df.columns and 'SALES_TIER' in df.columns:
            df['MARKET_TIER_COMBO'] = df['VRTCL_MRKT_NAME'] + '_' + df['SALES_TIER']

        # --- Missing value indicators ---

        # Create missing value indicators for potentially important columns
        important_cols = ['MARGIN', 'ACCESS_SPEED', 'PORT_SPEED', 'CIR_TECH_TYPE', 'VENDOR']
        for col in important_cols:
            if col in df.columns:
                df[f'{col}_IS_MISSING'] = df[col].isna().astype(int)

        # --- Polynomial features for key numerics ---

        # Add polynomial terms for key numeric features
        numeric_features = ['MARGIN', 'ACCESS_SPEED', 'PORT_SPEED', 'TOTAL_DISC']
        numeric_features = [f for f in numeric_features if f in df.columns]

        for feat in numeric_features:
            df[f'{feat}_SQUARED'] = df[feat] ** 2
            df[f'LOG_{feat}'] = np.log1p(df[feat].clip(lower=0))

        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.001) -> pd.DataFrame:
        """Select important features using Random Forest importance scores"""
        selector = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state
        )
        selector.fit(X, y)

        importance = pd.Series(selector.feature_importances_, index=X.columns)
        important_features = importance[importance > threshold].index

        print(f"\nSelected {len(important_features)} features out of {len(X.columns)}")
        return X[important_features]

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess the circuit data

        Args:
            is_training: whether this is training (true) or prediction (false)
        """

        if is_training:
            df = self.get_evenly_distributed_circuits(df.copy(), is_training=True)
        else:
            df = self.get_evenly_distributed_circuits(df.copy(), is_training=False)
        print(f"After distribution: {df.shape}")

        # Handle speed columns
        speed_replacements = {
            '.00': '0',
            'None': '0',
            'OTHER': '0',
            ' Kbps': '0000',
            ' Mbps': '000000',
            '1Mbps': '000000',
            'Mbps': '000000',
            ' Gbps': '000000000'
        }

        for col in df.columns:
            if 'SPEED' in col:
                for old, new in speed_replacements.items():
                    df[col] = df[col].astype(str).str.replace(old, new)
                df[col] = df[col].str.strip()
                df[col] = df[col].apply(lambda x: str(x).split('.')[0])
                df[col] = df[col].apply(lambda x: str(x).replace('\x00', ''))
                df[col] = df[col].apply(lambda x: '0' if x == '' else x)
                df[col] = df[col].astype('int64')

        # Add engineered features
        df = self.add_engineered_features(df)

        # Process circuit type columns
        df['SR_CRCTP'] = df['SR_CRCTP'].apply(lambda x: str(x).split('x')[-1])
        df['RPTG_CRCTP'] = df['RPTG_CRCTP'].apply(lambda x: str(x).split('x')[-1])

        # Remove outliers from numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df = self.handle_outliers(df, numeric_cols)

        # Drop unnecessary columns
        columns_to_drop = [
            'CIR_ID', 'NASP_ID', 'NASP_NM', 'FIRST_CITY', 'FIRST_STATE',
            'FIRST_ZIP', 'DISCO_ORD_NUM', 'DIV_PORT', 'ACCESS_SPEED_UP',
            'DIV_ACCESS', 'CIR_INST_DATE', 'CIR_DISC_DATE', 'MIG_STATUS',
            'FIRST_LATA', 'VENDOR', 'DISCO_DATE_BCOM', 'DISCO_DATE_ORDERING_STRT',
            'SECOND_LAT', 'SECOND_LGNTD'
        ]
        df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True)

        # Handle missing values
        for col in df.columns:
            if df[col].dtype.name == 'category' or df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].astype(str).fillna('Unknown')

        # One-hot encoding
        df = pd.get_dummies(df, drop_first=True)

        for col in df.columns:
            df[col] = df[col].astype(np.float32)

        return df

    def drop_highly_correlated_features(self, X, threshold=0.95):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

        print(f'Dripping {len(to_drop)} highly correlated features: {to_drop}')
        return X.drop(columns=to_drop)

    def get_model_params(self) -> Dict[str, Any]:
        """Define focused hyperparameter search spaces"""
        rf_params = {
            'n_estimators': [500, 1000, 1500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        svr_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2]
        }

        xgb_params = {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 500, 1000],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        gb_params = {'max_depth': [3, 5, 7, 9],
                     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                     'n_estimators': [100, 300, 500],
                     'subsample': [0.8, 0.9, 1.0],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4],
                     'max_features': ['sqrt', 'log2', None]
                     }
        cat_params = {'depth': [4, 6, 8, 10],
                      'learning_rate': [0.01, 0.05, 0.1],
                      'iterations': [500, 1000, 1500],
                      'l2_leaf_reg': [1, 3, 5, 7]
                      }

        lgb_params = {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 500, 1000],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'min_data_in_leaf': [10, 20, 30]
        }

        return {'rf': rf_params, 'svr': svr_params, 'xgb': xgb_params, 'gb': gb_params, 'cat': cat_params,
                'lgb': lgb_params}

    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train and evaluate Random Forest and XGBoost models"""
        models = {
            'Random Forest': RandomForestRegressor(random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_state),
            'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=self.random_state),
            'CatBoost': CatBoostRegressor(verbose=0, random_state=self.random_state),
            'LightGBM': LGBMRegressor(random_state=self.random_state, verbose=-1, force_col_wise=True)
        }
        results = {}
        params = self.get_model_params()
        best_score = -float('inf')

        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()

            if name == 'Random Forest':
                search = RandomizedSearchCV(
                    model,
                    params['rf'],
                    n_iter=100,
                    cv=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    scoring='r2',
                    verbose=1
                )
            elif name == 'Gradient Boosting':
                search = RandomizedSearchCV(
                    model,
                    params['gb'],
                    n_iter=100,
                    cv=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    scoring='r2',
                    verbose=1
                )
            elif name == 'CatBoost':
                search = RandomizedSearchCV(
                    model,
                    params['cat'],
                    n_iter=100,
                    cv=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    scoring='r2',
                    verbose=1
                )
            elif name == 'LightGBM':
                search = RandomizedSearchCV(
                    model,
                    params['lgb'],
                    n_iter=100,
                    cv=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    scoring='r2',
                    verbose=0
                )
            else:  # XGBoost
                search = RandomizedSearchCV(
                    model,
                    params['xgb'],
                    n_iter=100,
                    cv=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    scoring='r2',
                    verbose=1
                )

            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            results[name] = self.calculate_metrics(y_test, y_pred)

            if results[name]['R2'] > best_score:
                best_score = results[name]['R2']
                self.best_model = search.best_estimator_

            if name == 'Random Forest':
                self.feature_importance = pd.Series(
                    search.best_estimator_.feature_importances_,
                    index=self.feature_names
                ).sort_values(ascending=False)

                # Store best model parameters
                self.best_rf_params = search.best_params_
            else:
                # Store XGBoost feature importance as well
                self.xgb_feature_importance = pd.Series(
                    search.best_estimator_.feature_importances_,
                    index=self.feature_names
                ).sort_values(ascending=False)

                # Store best model parameters
                self.best_xgb_params = search.best_params_

            print(f"{name} completed in {time.time() - start_time:.2f} seconds")
            print(f"Best parameters: {search.best_params_}")
            print(f"Best cross-validation score: {search.best_score_:.4f}")

        return results

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False),
            'R2': r2_score(y_true, y_pred)
        }

    def fit_predict(self, df: pd.DataFrame, target_col: str = 'DISCO_DURATION') -> Dict[str, Dict[str, float]]:
        """Main method to fit models and make predictions"""
        # Preprocess data
        processed_df = self.preprocess_data(df)

        # Split features and target
        X = processed_df.drop(columns=[target_col])
        y = processed_df[target_col]

        # Feature selection
        X = self.select_features(X, y, threshold=0.001)
        self.feature_names = X.columns

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        correlated_features = self.drop_highly_correlated_features(X_train).columns
        print(f'\nFinal feature set after removing correlation: {len(correlated_features)} features')

        X_train = X_train[correlated_features]
        X_test = X_test[correlated_features]

        self.feature_names = X_train.columns

        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train and evaluate models
        results = self.train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

        return results

    def get_feature_importance(self) -> pd.Series:
        """Return feature importance from the Random Forest model."""
        if self.feature_importance is None:
            raise ValueError("Model hasn't been trained yet. Call fit_predict first.")
        return self.feature_importance

    def plot_monthly_distribution(self, df: pd.DataFrame, title: str) -> pd.Series:
        """Plot the distribution of circuits across months."""
        monthly_counts = df.groupby('PROD_YR_MTH').size()

        plt.figure(figsize=(15, 5))
        monthly_counts.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Year-Month')
        plt.ylabel('Number of Circuits')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return monthly_counts


    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance for each model"""
        if not hasattr(self, 'best_models') or not self.best_models:
            raise ValueError("Models haven't been trained yet. Call fit_predict first.")

        # Collect feature importances from all models that support it
        importance_data = {}

        for model_name, model in self.best_models.items():
            # Check if the model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(
                    model.feature_importances_,
                    index=self.feature_names
                ).sort_values(ascending=False)
                importance_data[model_name] = importance
            elif model_name == 'CatBoost' and hasattr(model, 'get_feature_importance'):
                # Special handling for CatBoost
                importance = pd.Series(
                    model.get_feature_importance(),
                    index=self.feature_names
                ).sort_values(ascending=False)
                importance_data[model_name] = importance

        # If no models with feature importance were found
        if not importance_data:
            print("None of the trained models support feature importance visualization.")
            return

        # Create a figure with subplots based on the number of models with feature importance
        num_models = len(importance_data)
        fig, axes = plt.subplots(num_models, 1, figsize=(12, 6 * num_models), constrained_layout=True)

        # Handle the case when there's only one model (axes is not an array)
        if num_models == 1:
            axes = [axes]

        # Plot feature importance for each model
        for ax, (model_name, importance) in zip(axes, importance_data.items()):
            importance.head(top_n).plot(kind='barh', ax=ax)
            ax.set_title(f'{model_name} Feature Importance')
            ax.set_xlabel('Importance Score')

        plt.show()

        return importance_data

    def save_results(self, results: Dict, data_stats: Dict) -> str:
        """Save model results and statistics to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/improved/circuit_prediction_results_{timestamp}.txt'

        with open(filename, 'w') as f:
            f.write("=== Circuit Disconnection Duration Prediction Results ===\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=== Data Statistics ===\n")
            f.write(f"Original dataset size: {data_stats['original_size']}\n")
            f.write(f"Balanced dataset size: {data_stats['balanced_size']}\n")
            f.write(f"Minimum circuits per month: {data_stats['min_per_month']}\n\n")

            f.write("=== Model Performance ===\n")
            for model_name, metrics in results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * (len(model_name) + 1) + "\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")

            f.write("\n=== Best Model Parameters ===\n")
            if hasattr(self, 'best_model'):
                for model_name, best_model in self.best_models.items():
                    f.write(f'\n{model_name}:\n')
                    if hasattr(best_model, 'get_params'):
                        f.write(str(best_model.get_params()))
                    else:
                        f.write('No parameter available.\n')
            else:
                f.write('No best models saved.\n')

            f.write("\n\n=== Top 20 Important Features for Each Model ===\n")
            if hasattr(self, 'feature_importances'):
                for model_name, importance in self.feature_importance.items():
                    f.write(f'\n{model_name} Feature Importance:\n')
                    if isinstance(importance, pd.Series):
                        f.write(importance.head(20).to_string())
                    else:
                        f.write('No feature importance available.\n')
            else:
                f.write('No feature importance saved.\n')

        return filename

    def save_pipeline(self, filename='results/models/circuit_prediction_pipeline_v3.joblib'):
        """Save the trained pipeline"""
        if self.best_model is None:
            raise ValueError('Cannot save pipeline - no trained model found')
        joblib.dump({'best_model': self.best_model,
                     'scaler': self.scaler,
                     'feature_names': self.feature_names}, filename)
        print(f'Pipeline saved to {filename}')