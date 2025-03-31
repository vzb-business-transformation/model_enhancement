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
from model_enhancer import ModelEnhancer
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

    # 1. Load training data (2023-2024)
    print("Loading original training data...")
    train_query = """
    select distinct a.CIR_ID, a.NASP_ID, NASP_NM, DOM_INTL_FLAG, INT_EXT_FLAG, LVL_4_PRD_NM,
    CIR_INST_DATE, CIR_DISC_DATE, VRTCL_MRKT_NAME, SALES_TIER, SR_CRCTP, RPTG_CRCTP,
    RPTG_CIR_SPEED, ACCESS_SPEED, ACCESS_SPEED_UP, PORT_SPEED, PVC_CAR_SPEED,
    ACCESS_DISC, PORT_DISC, PVC_CAR_DISC, OTHER_DISC, OTHER_ADJ,
    TOTAL_REV-(TOTAL_COST + OCC_COST_NONAFLT_M6364) MARGIN,
    REV_TYPE_FLAG, COST_TYPE_FLAG, FIRST_ADDR_TYPE_CODE,
    FIRST_CITY, FIRST_STATE, FIRST_ZIP, FIRST_ONNET, FIRST_INREGION, FIRST_XOLIT, FIRST_LATA,
    ONNET_PROV, COMPANY_CODE, VRD_FLAG, OPCO_REV, IN_FOOTPRINT, NASP_TYPE,
    CIR_TECH_TYPE, CIR_BILL_TYPE, PROGRAM, VENDOR, BIZ_CASE, FIRST_LAT, FIRST_LGNTD,
    MIG_STATUS, SECOND_LAT, SECOND_LGNTD, IEN_PROV, DIV_PORT, DIV_ACCESS, a.PROD_YR_MTH,
    DISCO_ORD_NUM, DISCO_DATE_BCOM, DISCO_DATE_ORDERING_STRT, 
    DISCO_DATE_BCOM - DISCO_DATE_ORDERING_STRT as DISCO_DURATION
    from edw_sr_vw.rt_cir_single_row_addr a
    inner join (
        select conv_naspid NASP_ID, CIR_ID, INST_ORD_NUM, CHG_ORD_NUM, MIG_STATUS,
        DISCO_ORD_NUM, DISCO_DATE_BCOM, DISCO_DATE_ORDERING_STRT
        from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
        where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
        qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
    ) b ON a.CIR_ID = b.CIR_ID
    where PROD_YR_MTH >= 202301 and PROD_YR_MTH <= 202412
    and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
    and DISCO_ORD_NUM <> ''
    and DISCO_DURATION >= 0
    """

    train_data = run_sql(train_query)

    # 2. Load January 2025 data (without predictions)
    print("Loading January 2025 data...")
    jan_query = """
    select distinct a.CIR_ID, a.NASP_ID, NASP_NM, DOM_INTL_FLAG, INT_EXT_FLAG, LVL_4_PRD_NM,
    CIR_INST_DATE, CIR_DISC_DATE, VRTCL_MRKT_NAME, SALES_TIER, SR_CRCTP, RPTG_CRCTP,
    RPTG_CIR_SPEED, ACCESS_SPEED, ACCESS_SPEED_UP, PORT_SPEED, PVC_CAR_SPEED,
    ACCESS_DISC, PORT_DISC, PVC_CAR_DISC, OTHER_DISC, OTHER_ADJ,
    TOTAL_REV-(TOTAL_COST + OCC_COST_NONAFLT_M6364) MARGIN,
    REV_TYPE_FLAG, COST_TYPE_FLAG, FIRST_ADDR_TYPE_CODE, 
    FIRST_CITY, FIRST_STATE, FIRST_ZIP, FIRST_ONNET, FIRST_INREGION, FIRST_XOLIT, FIRST_LATA,
    ONNET_PROV, COMPANY_CODE, VRD_FLAG, OPCO_REV, IN_FOOTPRINT, NASP_TYPE,
    CIR_TECH_TYPE, CIR_BILL_TYPE, PROGRAM, VENDOR, BIZ_CASE, FIRST_LAT, FIRST_LGNTD,
    MIG_STATUS, SECOND_LAT, SECOND_LGNTD, IEN_PROV, DIV_PORT, DIV_ACCESS, a.PROD_YR_MTH,
    DISCO_ORD_NUM, DISCO_DATE_BCOM, DISCO_DATE_ORDERING_STRT,
    DISCO_DATE_BCOM - DISCO_DATE_ORDERING_STRT as DISCO_DURATION
    from edw_sr_vw.rt_cir_single_row_addr a
    inner join (
        select conv_naspid NASP_ID, CIR_ID, INST_ORD_NUM, CHG_ORD_NUM, MIG_STATUS,
        DISCO_ORD_NUM, DISCO_DATE_BCOM, DISCO_DATE_ORDERING_STRT
        from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
        where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
        qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
    ) b ON a.CIR_ID = b.CIR_ID
    where PROD_YR_MTH = 202501
    and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
    and DISCO_DATE_BCOM is not null
    and DISCO_DATE_ORDERING_STRT is not null
    and DISCO_ORD_NUM <> ''
    and DISCO_DURATION >= 0
    """

    jan_data = run_sql(jan_query)

    # 3. Load February 2025 data
    print("Loading February 2025 data...")
    feb_query = """
    select distinct a.CIR_ID, a.NASP_ID, NASP_NM, DOM_INTL_FLAG, INT_EXT_FLAG, LVL_4_PRD_NM,
    CIR_INST_DATE, CIR_DISC_DATE, VRTCL_MRKT_NAME, SALES_TIER, SR_CRCTP, RPTG_CRCTP,
    RPTG_CIR_SPEED, ACCESS_SPEED, ACCESS_SPEED_UP, PORT_SPEED, PVC_CAR_SPEED,
    ACCESS_DISC, PORT_DISC, PVC_CAR_DISC, OTHER_DISC, OTHER_ADJ,
    TOTAL_REV-(TOTAL_COST + OCC_COST_NONAFLT_M6364) MARGIN,
    REV_TYPE_FLAG, COST_TYPE_FLAG, FIRST_ADDR_TYPE_CODE,
    FIRST_CITY, FIRST_STATE, FIRST_ZIP, FIRST_ONNET, FIRST_INREGION, FIRST_XOLIT, FIRST_LATA,
    ONNET_PROV, COMPANY_CODE, VRD_FLAG, OPCO_REV, IN_FOOTPRINT, NASP_TYPE,
    CIR_TECH_TYPE, CIR_BILL_TYPE, PROGRAM, VENDOR, BIZ_CASE, FIRST_LAT, FIRST_LGNTD,
    MIG_STATUS, SECOND_LAT, SECOND_LGNTD, IEN_PROV, DIV_PORT, DIV_ACCESS, a.PROD_YR_MTH,
    DISCO_ORD_NUM, DISCO_DATE_BCOM, DISCO_DATE_ORDERING_STRT,
    DISCO_DATE_BCOM - DISCO_DATE_ORDERING_STRT as DISCO_DURATION
    from edw_sr_vw.rt_cir_single_row_addr a
    inner join (
        select conv_naspid NASP_ID, CIR_ID, INST_ORD_NUM, CHG_ORD_NUM, MIG_STATUS,
        DISCO_ORD_NUM, DISCO_DATE_BCOM, DISCO_DATE_ORDERING_STRT
        from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
        where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
        qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
    ) b ON a.CIR_ID = b.CIR_ID
    where PROD_YR_MTH = 202502
    and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
    and DISCO_DATE_BCOM is not null
    and DISCO_DATE_ORDERING_STRT is not null
    and DISCO_ORD_NUM <> ''
    """

    feb_data = run_sql(feb_query)

    # 4. Initialize components
    print("Initializing model components...")
    data_processor = DataProcessor(random_state=42)
    ml_models = MLModels(random_state=42)

    # 5. Train the model with 2023-2024 data
    print("\n=== STEP 1: Training initial model with 2023-2024 data ===\n")

    # Preprocess the training data
    processed_train_df = data_processor.preprocess_data(train_data, is_training=True, target_col='DISCO_DURATION')

    # Prepare train/test split
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = data_processor.prepare_train_test_data(
        processed_train_df, target_col='DISCO_DURATION', test_size=0.2
    )

    # Feature selection and training
    X_train_selected = ml_models.select_features(X_train, y_train, threshold=0.001, method='rf')
    X_test_selected = X_test[X_train_selected.columns]

    # Remove highly correlated features
    X_train_final = ml_models.drop_correlated_features(X_train_selected, threshold=0.95)
    X_test_final = X_test_selected[X_train_final.columns]

    # Train all models
    models_to_train = ['rf', 'gb', 'xgb', 'lgb', 'cat']
    base_models = ml_models.train_all_models(X_train_final, y_train, models_to_train=models_to_train)

    # Create ensembles
    stacked_model, stacked_metrics = ml_models.create_stacked_ensemble(
        X_train_final, X_test_final, y_train, y_test
    )

    voting_model, voting_metrics = ml_models.create_voting_ensemble(
        X_train_final, X_test_final, y_train, y_test
    )

    # Evaluate all models
    base_results = ml_models.evaluate_models(X_test_final, y_test)

    # Save the models and preprocessor
    data_processor.save_preprocessor(os.path.join(models_dir, 'data_processor.joblib'))
    ml_models.save_models(models_dir)

    # 6. Make January predictions
    print("\n=== STEP 2: Predicting January 2025 data ===\n")

    # Preprocess January data
    processed_jan_df = data_processor.preprocess_data(jan_data, is_training=False, target_col='DISCO_DURATION')

    # Generate predictions
    jan_predictions = {}
    jan_results_df = pd.DataFrame({'CIR_ID': jan_data['CIR_ID']})
    jan_results_df['ACTUAL'] = jan_data['DISCO_DURATION']

    for name, model in ml_models.best_models.items():
        try:
            X_jan = processed_jan_df.drop(
                columns=['DISCO_DURATION']) if 'DISCO_DURATION' in processed_jan_df.columns else processed_jan_df
            preds = model.predict(X_jan)
            jan_predictions[name] = preds
            jan_results_df[f'prediction_{name}'] = preds
        except Exception as e:
            print(f"Error predicting with model {name}: {str(e)}")

    # Create ensemble prediction
    if len(jan_predictions) > 1:
        ensemble_preds = np.mean([jan_predictions[name] for name in jan_predictions], axis=0)
        jan_results_df['prediction_ensemble'] = ensemble_preds

    # Add prediction column (using ensemble or best model)
    if 'prediction_ensemble' in jan_results_df.columns:
        jan_results_df['PREDICTION'] = jan_results_df['prediction_ensemble']
    else:
        # Use the first available model
        first_model = next(iter(jan_predictions))
        jan_results_df['PREDICTION'] = jan_results_df[f'prediction_{first_model}']

    # Calculate basic error metrics for January predictions
    jan_results_df['ERROR'] = jan_results_df['PREDICTION'] - jan_results_df['ACTUAL']
    jan_results_df['ABS_ERROR'] = np.abs(jan_results_df['ERROR'])
    jan_results_df['PCT_ERROR'] = (jan_results_df['ABS_ERROR'] / jan_results_df['ACTUAL'].replace(0, 1)) * 100

    # Print January prediction metrics
    mae = jan_results_df['ABS_ERROR'].mean()
    rmse = np.sqrt(np.mean(jan_results_df['ERROR'] ** 2))
    r2 = 1 - (np.sum(jan_results_df['ERROR'] ** 2) / np.sum(
        (jan_results_df['ACTUAL'] - jan_results_df['ACTUAL'].mean()) ** 2))

    print(f"January prediction metrics:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

    # Save January predictions
    jan_output_file = os.path.join(jan_output_dir, 'january_predictions.csv')
    jan_results_df.to_csv(jan_output_file, index=False)
    print(f"January predictions saved to {jan_output_file}")

    # 7. Enhance model with January feedback
    print("\n=== STEP 3: Enhancing model with January feedback ===\n")

    # Create a full January dataset with predictions and original features
    jan_with_preds = jan_data.copy()
    jan_with_preds['PREDICTION'] = jan_results_df['PREDICTION']

    # Initialize the model enhancer
    enhancer = ModelEnhancer(
        data_processor=data_processor,
        ml_models=ml_models,
        original_model_path=models_dir,
        output_dir=enhanced_output_dir,
        random_state=42
    )

    # Analyze January prediction errors
    error_patterns, high_errors = enhancer.analyze_errors(
        jan_predictions=jan_results_df[['CIR_ID', 'prediction_ensemble']].rename(
            columns={'prediction_ensemble': 'prediction_model'}),
        jan_actuals=jan_results_df['ACTUAL'],
        features_df=jan_data.drop(columns=['DISCO_DURATION']),
        threshold=0.3
    )

    # Retrain with January feedback
    enhanced_model_results = enhancer.retrain_with_feedback(
        train_data=train_data,
        jan_data=jan_with_preds,
        target_col='DISCO_DURATION'
    )

    # 8. Predict February with enhanced model
    print("\n=== STEP 4: Predicting February 2025 with enhanced model ===\n")

    # Generate February predictions with enhanced model
    feb_predictions = enhancer.predict_future(
        feb_data=feb_data,
        target_col='DISCO_DURATION'
    )

    # 9. Generate summary report
    print("\n=== STEP 5: Generating summary report ===\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(base_output_dir, f'prediction_pipeline_summary_{timestamp}.txt')

    with open(summary_file, 'w') as f:
        f.write("=== Sequential Circuit Prediction Pipeline Summary ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=== Initial Model Performance (2023-2024 Data) ===\n")
        for model_name, metrics in base_results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * (len(model_name) + 1) + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

        f.write("\n\n=== January 2025 Prediction Performance ===\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")

        f.write("\n\n=== Key Error Patterns in January Predictions ===\n")
        for model_name, insights in error_patterns.items():
            f.write(f"\nModel: {model_name}\n")
            f.write("-" * (len(model_name) + 8) + "\n")
            f.write(f"High-error cases: {insights['high_error_count']}\n")
            f.write(f"Error bias: {insights['error_bias']:.4f}\n")

            f.write("\nTop features correlated with error:\n")
            for feature, corr in insights['correlated_features'].items():
                f.write(f"  {feature}: {corr:.4f}\n")

        f.write("\n\n=== Enhanced Model Performance ===\n")
        for model_name, metrics in enhanced_model_results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * (len(model_name) + 1) + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

        f.write(f"\n\nPipeline completed in {time.time() - start_time:.2f} seconds\n")

    print(f"Summary report saved to {summary_file}")
    print(f"\nSequential prediction pipeline completed in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()