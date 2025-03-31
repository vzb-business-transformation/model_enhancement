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
    print("\n=== Starting Sequential Circuit Prediction Enhancement Pipeline ===\n")
    start_time = time.time()

    # Set up directories
    base_output_dir = "results"
    jan_output_dir = os.path.join(base_output_dir, "january")
    enhanced_output_dir = os.path.join(base_output_dir, "enhanced")
    models_dir = os.path.join(base_output_dir, "models")

    os.makedirs(jan_output_dir, exist_ok=True)
    os.makedirs(enhanced_output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load data
    print("Loading training data (2023-2024)...")

    # Training data (completed disconnects from 2023-2024)
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

    # January 2025 data for evaluation (completed disconnects)
    jan_actual_query = """
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
        select conv_naspid NASP_ID,CIR_ID,INST_ORD_NUM,CHG_ORD_NUM,MIG_STATUS,DISCO_ORD_NUM,DISCO_DATE_BCOM,DISCO_DATE_ORDERING_STRT
        from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
        where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
        qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
    ) b ON a.CIR_ID = b.CIR_ID
    where PROD_YR_MTH = 202501
    and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
    and DISCO_ORD_NUM is null
    """

    # January 2025 data to predict (in-progress disconnects)
    jan_predict_query = """
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

    # February 2025 data to predict (in-progress disconnects)
    feb_predict_query = """
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

    # Execute queries
    print("Executing queries...")
    train_data = run_sql(train_query)
    jan_actual_data = run_sql(jan_actual_query)
    jan_predict_data = run_sql(jan_predict_query)
    feb_predict_data = run_sql(feb_predict_query)

    print(f"Training data: {len(train_data)} records")
    print(f"January completed disconnects: {len(jan_actual_data)} records")
    print(f"January in-progress disconnects: {len(jan_predict_data)} records")
    print(f"February in-progress disconnects: {len(feb_predict_data)} records")

    # Add dummy DISCO_DURATION column for preprocessing
    jan_predict_data['DISCO_DURATION'] = 0
    feb_predict_data['DISCO_DURATION'] = 0

    # Initialize data processor
    data_processor = DataProcessor(random_state=42)

    # STEP 1: Train initial model with 2023-2024 data
    print("\n=== STEP 1: Training initial model with 2023-2024 data ===\n")

    # Process training data
    train_processed = data_processor.preprocess_data(train_data, is_training=True, target_col='DISCO_DURATION')

    # Prepare train/test split
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = data_processor.prepare_train_test_data(
        train_processed, target_col='DISCO_DURATION', test_size=0.2
    )

    # Initialize ML models
    ml_models = MLModels(random_state=42)

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

    # STEP 2: Make predictions for January in-progress disconnects
    print("\n=== STEP 2: Predicting January in-progress disconnect durations ===\n")

    # Process January data for prediction
    jan_predict_processed = data_processor.preprocess_data(jan_predict_data, is_training=False,
                                                           target_col='DISCO_DURATION')

    # Ensure feature compatibility
    train_features = set(train_processed.columns)
    jan_predict_features = set(jan_predict_processed.columns)

    # Find and fix feature mismatches
    missing_features = train_features - jan_predict_features
    extra_features = jan_predict_features - train_features

    print(f"Features missing in January data: {len(missing_features)}")
    print(f"Extra features in January data: {len(extra_features)}")

    # Add missing features
    for feature in missing_features:
        if feature != 'DISCO_DURATION':
            jan_predict_processed[feature] = 0

    # Remove extra features
    jan_predict_processed = jan_predict_processed.drop(columns=list(extra_features - {'DISCO_DURATION'}),
                                                       errors='ignore')

    # Ensure same column order
    common_columns = list(train_features & jan_predict_features)
    if 'DISCO_DURATION' in common_columns:
        common_columns.remove('DISCO_DURATION')
    jan_predict_processed = jan_predict_processed[common_columns + ['DISCO_DURATION']]

    # Make predictions
    X_jan_predict = jan_predict_processed.drop(columns=['DISCO_DURATION'])

    # Initialize results DataFrame
    jan_predictions = pd.DataFrame({'CIR_ID': jan_predict_data['CIR_ID']})

    # Make predictions with each model
    for name, model in ml_models.best_models.items():
        try:
            preds = model.predict(X_jan_predict)
            jan_predictions[f'prediction_{name}'] = preds
            print(f"Successfully predicted with {name} model")
        except Exception as e:
            print(f"Error predicting with {name} model: {str(e)}")

    # Create ensemble prediction
    pred_columns = [col for col in jan_predictions.columns if col.startswith('prediction_')]
    if pred_columns:
        jan_predictions['PREDICTED_DURATION'] = jan_predictions[pred_columns].mean(axis=1)

    # Add start date for reference
    jan_predictions['DISCO_DATE_ORDERING_STRT'] = jan_predict_data['DISCO_DATE_ORDERING_STRT']

    # Calculate estimated completion date
    if 'PREDICTED_DURATION' in jan_predictions.columns:
        jan_predictions['ESTIMATED_COMPLETION_DATE'] = pd.to_datetime(jan_predictions['DISCO_DATE_ORDERING_STRT']) + \
                                                       pd.to_timedelta(jan_predictions['PREDICTED_DURATION'], unit='D')

    # Save January predictions
    jan_output_file = os.path.join(jan_output_dir, 'january_predictions.csv')
    jan_predictions.to_csv(jan_output_file, index=False)
    print(f"January predictions saved to {jan_output_file}")

    # STEP 3: Evaluate with January completed disconnects
    print("\n=== STEP 3: Evaluating model with January completed disconnects ===\n")

    # Process January actual data
    jan_actual_processed = data_processor.preprocess_data(jan_actual_data, is_training=False,
                                                          target_col='DISCO_DURATION')

    # Ensure feature compatibility
    jan_actual_features = set(jan_actual_processed.columns)

    # Find and fix feature mismatches
    missing_features = train_features - jan_actual_features
    extra_features = jan_actual_features - train_features

    # Add missing features
    for feature in missing_features:
        if feature != 'DISCO_DURATION':
            jan_actual_processed[feature] = 0

    # Remove extra features
    jan_actual_processed = jan_actual_processed.drop(columns=list(extra_features - {'DISCO_DURATION'}), errors='ignore')

    # Ensure same column order
    jan_actual_processed = jan_actual_processed[common_columns + ['DISCO_DURATION']]

    # Evaluate the model
    X_jan_actual = jan_actual_processed.drop(columns=['DISCO_DURATION'])
    y_jan_actual = jan_actual_processed['DISCO_DURATION']

    # Initialize evaluation DataFrame
    jan_evaluation = pd.DataFrame(
        {'CIR_ID': jan_actual_data['CIR_ID'], 'ACTUAL_DURATION': jan_actual_data['DISCO_DURATION']})

    # Make predictions with each model
    for name, model in ml_models.best_models.items():
        try:
            preds = model.predict(X_jan_actual)
            jan_evaluation[f'prediction_{name}'] = preds
        except Exception as e:
            print(f"Error evaluating with {name} model: {str(e)}")

    # Create ensemble prediction
    pred_columns = [col for col in jan_evaluation.columns if col.startswith('prediction_')]
    if pred_columns:
        jan_evaluation['PREDICTED_DURATION'] = jan_evaluation[pred_columns].mean(axis=1)

        # Calculate errors
        jan_evaluation['ERROR'] = jan_evaluation['PREDICTED_DURATION'] - jan_evaluation['ACTUAL_DURATION']
        jan_evaluation['ABS_ERROR'] = np.abs(jan_evaluation['ERROR'])
        jan_evaluation['PCT_ERROR'] = (jan_evaluation['ABS_ERROR'] / jan_evaluation['ACTUAL_DURATION'].replace(0,
                                                                                                               1)) * 100

        # Calculate metrics
        mae = jan_evaluation['ABS_ERROR'].mean()
        rmse = np.sqrt((jan_evaluation['ERROR'] ** 2).mean())
        r2 = 1 - (np.sum(jan_evaluation['ERROR'] ** 2) / np.sum(
            (jan_evaluation['ACTUAL_DURATION'] - jan_evaluation['ACTUAL_DURATION'].mean()) ** 2))

        print(f"January evaluation metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")

    # Save January evaluation
    jan_eval_file = os.path.join(jan_output_dir, 'january_evaluation.csv')
    jan_evaluation.to_csv(jan_eval_file, index=False)
    print(f"January evaluation saved to {jan_eval_file}")

    # STEP 4: Enhance model with January feedback
    print("\n=== STEP 4: Enhancing model with January feedback ===\n")

    # Combine training data with January completed disconnects
    combined_data = pd.concat([train_data, jan_actual_data], ignore_index=True)
    print(f"Combined training data: {len(combined_data)} records")

    # Process combined data
    combined_processed = data_processor.preprocess_data(combined_data, is_training=True, target_col='DISCO_DURATION')

    # Prepare train/test split
    X_enhanced, X_test_enhanced, y_enhanced, y_test_enhanced, _, _ = data_processor.prepare_train_test_data(
        combined_processed, target_col='DISCO_DURATION', test_size=0.2
    )

    # Retrain models
    enhanced_models = ml_models.train_all_models(X_enhanced, y_enhanced, models_to_train=models_to_train)

    # Evaluate enhanced models
    enhanced_results = ml_models.evaluate_models(X_test_enhanced, y_test_enhanced)

    # Identify best enhanced model
    best_enhanced_model = max(enhanced_results.items(), key=lambda x: x[1]['R2'])[0]
    print(f"Best enhanced model is {best_enhanced_model} with R² = {enhanced_results[best_enhanced_model]['R2']:.4f}")

    # Save enhanced models
    ml_models.save_models(os.path.join(enhanced_output_dir, 'models'))
    data_processor.save_preprocessor(os.path.join(enhanced_output_dir, 'data_processor.joblib'))

    # STEP 5: Make predictions for February in-progress disconnects
    print("\n=== STEP 5: Predicting February in-progress disconnect durations ===\n")

    # Process February data for prediction
    feb_predict_processed = data_processor.preprocess_data(feb_predict_data, is_training=False,
                                                           target_col='DISCO_DURATION')

    # Ensure feature compatibility
    enhanced_features = set(combined_processed.columns)
    feb_predict_features = set(feb_predict_processed.columns)

    # Find and fix feature mismatches
    missing_features = enhanced_features - feb_predict_features
    extra_features = feb_predict_features - enhanced_features

    # Add missing features
    for feature in missing_features:
        if feature != 'DISCO_DURATION':
            feb_predict_processed[feature] = 0

    # Remove extra features
    feb_predict_processed = feb_predict_processed.drop(columns=list(extra_features - {'DISCO_DURATION'}),
                                                       errors='ignore')

    # Ensure same column order
    common_columns = list(enhanced_features & feb_predict_features)
    if 'DISCO_DURATION' in common_columns:
        common_columns.remove('DISCO_DURATION')
    feb_predict_processed = feb_predict_processed[common_columns + ['DISCO_DURATION']]

    # Make predictions
    X_feb_predict = feb_predict_processed.drop(columns=['DISCO_DURATION'])

    # Initialize results DataFrame
    feb_predictions = pd.DataFrame({'CIR_ID': feb_predict_data['CIR_ID']})

    # Make predictions with each model
    for name, model in ml_models.best_models.items():
        try:
            preds = model.predict(X_feb_predict)
            feb_predictions[f'prediction_{name}'] = preds
            print(f"Successfully predicted with enhanced {name} model")
        except Exception as e:
            print(f"Error predicting with enhanced {name} model: {str(e)}")

    # Create ensemble prediction
    pred_columns = [col for col in feb_predictions.columns if col.startswith('prediction_')]
    if pred_columns:
        feb_predictions['PREDICTED_DURATION'] = feb_predictions[pred_columns].mean(axis=1)

    # Add start date for reference
    feb_predictions['DISCO_DATE_ORDERING_STRT'] = feb_predict_data['DISCO_DATE_ORDERING_STRT']

    # Calculate estimated completion date
    if 'PREDICTED_DURATION' in feb_predictions.columns:
        feb_predictions['ESTIMATED_COMPLETION_DATE'] = pd.to_datetime(feb_predictions['DISCO_DATE_ORDERING_STRT']) + \
                                                       pd.to_timedelta(feb_predictions['PREDICTED_DURATION'], unit='D')

    # Save February predictions
    feb_output_file = os.path.join(enhanced_output_dir, 'february_predictions.csv')
    feb_predictions.to_csv(feb_output_file, index=False)
    print(f"February predictions saved to {feb_output_file}")

    # STEP 6: Save summary report
    print("\n=== STEP 6: Generating summary report ===\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(base_output_dir, f'enhancement_summary_{timestamp}.txt')

    with open(summary_file, 'w') as f:
        f.write("=== Circuit Prediction Enhancement Summary ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=== Initial Model Performance (2023-2024 Data) ===\n")
        for model_name, metrics in model_results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * (len(model_name) + 1) + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

        f.write("\n\n=== January Evaluation Results ===\n")
        if 'PREDICTED_DURATION' in jan_evaluation.columns:
            f.write(f"Total evaluations: {len(jan_evaluation)}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")

        f.write("\n\n=== Enhanced Model Performance ===\n")
        for model_name, metrics in enhanced_results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * (len(model_name) + 1) + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

        f.write(f"\n\n=== Prediction Counts ===\n")
        f.write(f"January predictions: {len(jan_predictions)}\n")
        f.write(f"February predictions: {len(feb_predictions)}\n")

        f.write(f"\n\nPipeline completed in {time.time() - start_time:.2f} seconds\n")

    print(f"Summary report saved to {summary_file}")
    print(f"\nSequential prediction pipeline completed in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
