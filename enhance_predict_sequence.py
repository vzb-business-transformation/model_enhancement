# enhance_predict_sequence.py
# enhance_predict_sequence.py
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_processing import DataProcessor
from ml_models import MLModels
from src.run_query import run_sql

def main():
    """Main function to run the sequential prediction and enhancement workflow"""
    print("\n=== Starting Circuit Prediction Enhancement Pipeline ===\n")
    start_time = time.time()
    
    # Set up directories
    base_output_dir = "results"
    jan_output_dir = os.path.join(base_output_dir, "january")
    enhanced_output_dir = os.path.join(base_output_dir, "enhanced")
    models_dir = os.path.join(base_output_dir, "models")
    
    os.makedirs(jan_output_dir, exist_ok=True)
    os.makedirs(enhanced_output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize data processor and ML models
    data_processor = DataProcessor(random_state=42)
    ml_models = MLModels(random_state=42)
    
    # Check if models already exist
    base_model_exists = os.path.exists(os.path.join(models_dir, 'rf_model.joblib'))
    enhanced_model_exists = os.path.exists(os.path.join(enhanced_output_dir, 'models/rf_model.joblib'))
    
    # STEP 1: Load or train base model
    if not base_model_exists:
        print("\n=== STEP 1: Training initial model with 2023-2024 data ===\n")
        
        # Load training data (completed disconnects from 2023-2024)
        train_query = """
        select distinct a.CIR_ID,a.NASP_ID, NASP_NM, DOM_INTL_FLAG, INT_EXT_FLAG, LVL_4_PRD_NM,CIR_INST_DATE,CIR_DISC_DATE,VRTCL_MRKT_NAME,
        SALES_TIER, SR_CRCTP, RPTG_CRCTP, RPTG_CIR_SPEED, ACCESS_SPEED, ACCESS_SPEED_UP, 
        PORT_SPEED, PVC_CAR_SPEED, ACCESS_DISC, PORT_DISC, PVC_CAR_DISC, OTHER_DISC, OTHER_ADJ,
        TOTAL_REV-(TOTAL_COST + OCC_COST_NONAFLT_M6364) MARGIN,
        REV_TYPE_FLAG, COST_TYPE_FLAG, FIRST_ADDR_TYPE_CODE,
        FIRST_CITY, FIRST_STATE, FIRST_ZIP,
        FIRST_ONNET, FIRST_INREGION, FIRST_XOLIT, FIRST_LATA,
        ONNET_PROV, COMPANY_CODE, VRD_FLAG, OPCO_REV, IN_FOOTPRINT, NASP_TYPE,
        CIR_TECH_TYPE, CIR_BILL_TYPE, PROGRAM, VENDOR,
        BIZ_CASE, FIRST_LAT, FIRST_LGNTD,MIG_STATUS,
        SECOND_LAT, SECOND_LGNTD, IEN_PROV, DIV_PORT, DIV_ACCESS, a.PROD_YR_MTH,
        DISCO_ORD_NUM,DISCO_DATE_BCOM,DISCO_DATE_ORDERING_STRT,DISCO_DATE_BCOM - DISCO_DATE_ORDERING_STRT as DISCO_DURATION
        from edw_sr_vw.rt_cir_single_row_addr a
        inner join (
            select conv_naspid NASP_ID,CIR_ID,INST_ORD_NUM,CHG_ORD_NUM,MIG_STATUS,DISCO_ORD_NUM,DISCO_DATE_BCOM,DISCO_DATE_ORDERING_STRT
            from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
            where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
            qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
        ) b ON a.CIR_ID = b.CIR_ID
        where PROD_YR_MTH >= 202301 and PROD_YR_MTH <= 202412
        and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
        and DISCO_DATE_BCOM is not null
        and DISCO_DATE_ORDERING_STRT is not null
        and DISCO_ORD_NUM <> ''
        and DISCO_DURATION > 0 
        and DISCO_DURATION <= 365  -- Filter unreasonable durations
        """
        
        print("Loading training data...")
        train_data = run_sql(train_query)
        print(f"Training data: {len(train_data)} records")
        
        # Process training data
        train_processed = data_processor.preprocess_data(train_data, is_training=True, target_col='DISCO_DURATION')
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = data_processor.prepare_train_test_data(
            train_processed, target_col='DISCO_DURATION', test_size=0.2
        )
        
        # Train models
        models_to_train = ['rf', 'gb', 'xgb']  # More stable algorithms
        trained_models = ml_models.train_all_models(X_train, y_train, models_to_train=models_to_train)
        
        # Evaluate models
        print("Evaluating models on test data...")
        model_results = ml_models.evaluate_models(X_test, y_test)
        
        # Identify best model
        best_model_name = max(model_results.items(), key=lambda x: x[1]['R2'])[0]
        print(f"Best model is {best_model_name} with R² = {model_results[best_model_name]['R2']:.4f}")
        
        # Save models and preprocessor
        data_processor.save_preprocessor(os.path.join(models_dir, 'data_processor.joblib'))
        ml_models.save_models(models_dir)
    else:
        print("\n=== STEP 1: Loading existing base model ===\n")
        # Load preprocessor and models
        data_processor.load_preprocessor(os.path.join(models_dir, 'data_processor.joblib'))
        ml_models.load_models(models_dir)
        print("Base models loaded successfully")
    
    # STEP 2: Load January prediction data and make predictions
    print("\n=== STEP 2: Predicting January disconnect durations ===\n")
    
    # January data to predict (active circuits that may be disconnected)
    jan_query = """
    select distinct a.CIR_ID,a.NASP_ID, NASP_NM, DOM_INTL_FLAG, INT_EXT_FLAG, LVL_4_PRD_NM,CIR_INST_DATE,CIR_DISC_DATE,VRTCL_MRKT_NAME,
    SALES_TIER, SR_CRCTP, RPTG_CRCTP, RPTG_CIR_SPEED, ACCESS_SPEED, ACCESS_SPEED_UP, 
    PORT_SPEED, PVC_CAR_SPEED, ACCESS_DISC, PORT_DISC, PVC_CAR_DISC, OTHER_DISC, OTHER_ADJ,
    TOTAL_REV-(TOTAL_COST + OCC_COST_NONAFLT_M6364) MARGIN,
    REV_TYPE_FLAG, COST_TYPE_FLAG, FIRST_ADDR_TYPE_CODE,
    FIRST_CITY, FIRST_STATE, FIRST_ZIP,
    FIRST_ONNET, FIRST_INREGION, FIRST_XOLIT, FIRST_LATA,
    ONNET_PROV, COMPANY_CODE, VRD_FLAG, OPCO_REV, IN_FOOTPRINT, NASP_TYPE,
    CIR_TECH_TYPE, CIR_BILL_TYPE, PROGRAM, VENDOR,
    BIZ_CASE, FIRST_LAT, FIRST_LGNTD,MIG_STATUS,
    SECOND_LAT, SECOND_LGNTD, IEN_PROV, DIV_PORT, DIV_ACCESS, a.PROD_YR_MTH
    from edw_sr_vw.rt_cir_single_row_addr a
    inner join (
        select conv_naspid NASP_ID,CIR_ID,INST_ORD_NUM,CHG_ORD_NUM,MIG_STATUS,DISCO_ORD_NUM,DISCO_DATE_ORDERING_STRT
        from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
        where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
        qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
    ) b ON a.CIR_ID = b.CIR_ID
    where PROD_YR_MTH = 202501
    and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
    and DISCO_ORD_NUM is null
    """
    
    print("Loading January data...")
    jan_data = run_sql(jan_query)
    print(f"January active circuits: {len(jan_data)} records")
    
    # Add dummy DISCO_DURATION column for preprocessing
    jan_data['DISCO_DURATION'] = 0
    
    # Process January data
    jan_processed = data_processor.preprocess_data(jan_data, is_training=False, target_col='DISCO_DURATION')
    
    # Handle feature compatibility
    # Get model's expected features
    model_features = ml_models.get_feature_names() if hasattr(ml_models, 'get_feature_names') else None
    
    if model_features is not None:
        # Add missing features
        for feature in model_features:
            if feature not in jan_processed.columns and feature != 'DISCO_DURATION':
                jan_processed[feature] = 0
        
        # Remove extra features
        extra_features = [col for col in jan_processed.columns 
                          if col not in model_features and col != 'DISCO_DURATION']
        jan_processed = jan_processed.drop(columns=extra_features, errors='ignore')
        
        # Ensure same column order
        feature_order = [col for col in model_features if col in jan_processed.columns]
        if 'DISCO_DURATION' not in feature_order:
            feature_order.append('DISCO_DURATION')
        jan_processed = jan_processed[feature_order]
    
    # Make predictions
    X_jan = jan_processed.drop(columns=['DISCO_DURATION'])
    
    # Initialize results DataFrame
    jan_predictions = pd.DataFrame({'CIR_ID': jan_data['CIR_ID']})
    
    # Make predictions with each model
    for name, model in ml_models.best_models.items():
        try:
            preds = model.predict(X_jan)
            jan_predictions[f'prediction_{name}'] = preds
            print(f"Successfully predicted with {name} model")
        except Exception as e:
            print(f"Error predicting with {name} model: {str(e)}")
    
    # Create ensemble prediction
    pred_columns = [col for col in jan_predictions.columns if col.startswith('prediction_')]
    if pred_columns:
        jan_predictions['PREDICTED_DURATION'] = jan_predictions[pred_columns].mean(axis=1)
    
    # Save January predictions
    jan_output_file = os.path.join(jan_output_dir, 'january_predictions.csv')
    jan_predictions.to_csv(jan_output_file, index=False)
    print(f"January predictions saved to {jan_output_file}")
    
    # STEP 3: Error analysis and model enhancement (if we later get actual data)
    print("\n=== STEP 3: Analyzing January prediction errors ===\n")
    
    # For now, we're just noting that this would be done later when actual
    # disconnect durations for January circuits are available
    print("Note: To perform error analysis, we'll need to wait until actual disconnect")
    print("durations are available for the January circuits we're predicting.")
    print("When that data is available, rerun this step to enhance the model.")
    
    # STEP 4: Make February predictions using base model
    print("\n=== STEP 4: Predicting February disconnect durations ===\n")
    
    # Load February data
    feb_query = """
    select distinct a.CIR_ID,a.NASP_ID, NASP_NM, DOM_INTL_FLAG, INT_EXT_FLAG, LVL_4_PRD_NM,CIR_INST_DATE,CIR_DISC_DATE,VRTCL_MRKT_NAME,
    SALES_TIER, SR_CRCTP, RPTG_CRCTP, RPTG_CIR_SPEED, ACCESS_SPEED, ACCESS_SPEED_UP, 
    PORT_SPEED, PVC_CAR_SPEED, ACCESS_DISC, PORT_DISC, PVC_CAR_DISC, OTHER_DISC, OTHER_ADJ,
    TOTAL_REV-(TOTAL_COST + OCC_COST_NONAFLT_M6364) MARGIN,
    REV_TYPE_FLAG, COST_TYPE_FLAG, FIRST_ADDR_TYPE_CODE,
    FIRST_CITY, FIRST_STATE, FIRST_ZIP,
    FIRST_ONNET, FIRST_INREGION, FIRST_XOLIT, FIRST_LATA,
    ONNET_PROV, COMPANY_CODE, VRD_FLAG, OPCO_REV, IN_FOOTPRINT, NASP_TYPE,
    CIR_TECH_TYPE, CIR_BILL_TYPE, PROGRAM, VENDOR,
    BIZ_CASE, FIRST_LAT, FIRST_LGNTD,MIG_STATUS,
    SECOND_LAT, SECOND_LGNTD, IEN_PROV, DIV_PORT, DIV_ACCESS, a.PROD_YR_MTH
    from edw_sr_vw.rt_cir_single_row_addr a
    inner join (
        select conv_naspid NASP_ID,CIR_ID,INST_ORD_NUM,CHG_ORD_NUM,MIG_STATUS,DISCO_ORD_NUM,DISCO_DATE_ORDERING_STRT
        from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
        where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
        qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
    ) b ON a.CIR_ID = b.CIR_ID
    where PROD_YR_MTH = 202502
    and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
    and DISCO_ORD_NUM is null
    """
    
    print("Loading February data...")
    feb_data = run_sql(feb_query)
    print(f"February active circuits: {len(feb_data)} records")
    
    # Add dummy DISCO_DURATION column for preprocessing
    feb_data['DISCO_DURATION'] = 0
    
    # Process February data
    feb_processed = data_processor.preprocess_data(feb_data, is_training=False, target_col='DISCO_DURATION')
    
    # Handle feature compatibility (same as with January data)
    if model_features is not None:
        # Add missing features
        for feature in model_features:
            if feature not in feb_processed.columns and feature != 'DISCO_DURATION':
                feb_processed[feature] = 0
        
        # Remove extra features
        extra_features = [col for col in feb_processed.columns 
                          if col not in model_features and col != 'DISCO_DURATION']
        feb_processed = feb_processed.drop(columns=extra_features, errors='ignore')
        
        # Ensure same column order
        feature_order = [col for col in model_features if col in feb_processed.columns]
        if 'DISCO_DURATION' not in feature_order:
            feature_order.append('DISCO_DURATION')
        feb_processed = feb_processed[feature_order]
    
    # Make predictions
    X_feb = feb_processed.drop(columns=['DISCO_DURATION'])
    
    # Initialize results DataFrame
    feb_predictions = pd.DataFrame({'CIR_ID': feb_data['CIR_ID']})
    
    # Make predictions with each model
    for name, model in ml_models.best_models.items():
        try:
            preds = model.predict(X_feb)
            feb_predictions[f'prediction_{name}'] = preds
            print(f"Successfully predicted with {name} model")
        except Exception as e:
            print(f"Error predicting with {name} model: {str(e)}")
    
    # Create ensemble prediction
    pred_columns = [col for col in feb_predictions.columns if col.startswith('prediction_')]
    if pred_columns:
        feb_predictions['PREDICTED_DURATION'] = feb_predictions[pred_columns].mean(axis=1)
    
    # Save February predictions
    feb_output_file = os.path.join(base_output_dir, 'february_predictions.csv')
    feb_predictions.to_csv(feb_output_file, index=False)
    print(f"February predictions saved to {feb_output_file}")
    
    # STEP 5: Save summary report
    print("\n=== STEP 5: Generating summary report ===\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(base_output_dir, f'prediction_summary_{timestamp}.txt')
    
    with open(summary_file, 'w') as f:
        f.write("=== Circuit Disconnect Duration Prediction Summary ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Data Counts ===\n")
        f.write(f"January active circuits: {len(jan_data)}\n")
        f.write(f"February active circuits: {len(feb_data)}\n\n")
        
        f.write("=== Prediction Summary ===\n")
        
        # January predictions summary
        if 'PREDICTED_DURATION' in jan_predictions.columns:
            f.write("\nJanuary Predictions:\n")
            f.write(f"  Average predicted duration: {jan_predictions['PREDICTED_DURATION'].mean():.2f} days\n")
            f.write(f"  Min predicted duration: {jan_predictions['PREDICTED_DURATION'].min():.2f} days\n")
            f.write(f"  Max predicted duration: {jan_predictions['PREDICTED_DURATION'].max():.2f} days\n")
            
            # Distribution by duration range
            ranges = [(0, 30), (31, 60), (61, 90), (91, 180), (181, 365)]
            f.write("\n  Distribution by duration range:\n")
            for lower, upper in ranges:
                count = ((jan_predictions['PREDICTED_DURATION'] >= lower) & 
                         (jan_predictions['PREDICTED_DURATION'] <= upper)).sum()
                pct = count / len(jan_predictions) * 100
                f.write(f"    {lower}-{upper} days: {count} circuits ({pct:.2f}%)\n")
        
        # February predictions summary
        if 'PREDICTED_DURATION' in feb_predictions.columns:
            f.write("\nFebruary Predictions:\n")
            f.write(f"  Average predicted duration: {feb_predictions['PREDICTED_DURATION'].mean():.2f} days\n")
            f.write(f"  Min predicted duration: {feb_predictions['PREDICTED_DURATION'].min():.2f} days\n")
            f.write(f"  Max predicted duration: {feb_predictions['PREDICTED_DURATION'].max():.2f} days\n")
            
            # Distribution by duration range
            f.write("\n  Distribution by duration range:\n")
            for lower, upper in ranges:
                count = ((feb_predictions['PREDICTED_DURATION'] >= lower) & 
                         (feb_predictions['PREDICTED_DURATION'] <= upper)).sum()
                pct = count / len(feb_predictions) * 100
                f.write(f"    {lower}-{upper} days: {count} circuits ({pct:.2f}%)\n")
        
        f.write(f"\n\nPipeline completed in {time.time() - start_time:.2f} seconds\n")
    
    print(f"Summary report saved to {summary_file}")
    print(f"\nPrediction pipeline completed in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
