import os
from src.run_query import run_sql, run_sql1
import pandas as pd
from datetime import datetime
import copy
import hashlib
import warnings

warnings.filterwarnings('ignore')

train_query = '''
select distinct a.CIR_ID,a.NASP_ID, NASP_NM, DOM_INTL_FLAG,	INT_EXT_FLAG,	LVL_4_PRD_NM,CIR_INST_DATE,CIR_DISC_DATE,VRTCL_MRKT_NAME,
SALES_TIER,	SR_CRCTP,	RPTG_CRCTP,	RPTG_CIR_SPEED,	ACCESS_SPEED,	ACCESS_SPEED_UP,	
PORT_SPEED,	PVC_CAR_SPEED,		ACCESS_DISC,		PORT_DISC,	PVC_CAR_DISC,		OTHER_DISC,	OTHER_ADJ,		

-- MILEAGE_COST,	MILEAGE_COST_DISC,	CHANTERM_COST,	CHANTERM_COST_DISC,	OTHER_COST,	OTHER_COST_DISC,OTHER_REV,ACCESS_REV,PORT_REV,PVC_CAR_REV,
	TOTAL_REV-(TOTAL_COST + OCC_COST_NONAFLT_M6364) MARGIN,
/*    
MILEAGE_COST_NONAFLT,	MILEAGE_COST_DISC_NONAFLT,	CHANTERM_COST_NONAFLT,	CHANTERM_COST_DISC_NONAFLT,
OTHER_COST_NONAFLT,	OTHER_COST_DISC_NONAFLT,	TOTAL_COST_NONAFLT,	MILEAGE_COST_AFLT,	
MILEAGE_COST_DISC_AFLT,	CHANTERM_COST_AFLT,	CHANTERM_COST_DISC_AFLT,	OTHER_COST_AFLT,	OTHER_COST_DISC_AFLT,	
TOTAL_COST_AFLT,	SHARED_NETWORK_COST_NONAFLT,	SHARED_NETWORK_COST_AFLT,	SHARED_NETWORK_COST,	OCC_COST_NONAFLT,	
OCC_COST_AFLT,	TOTAL_OCC_COST,	
*/

REV_TYPE_FLAG,	COST_TYPE_FLAG,	FIRST_ADDR_TYPE_CODE,
FIRST_CITY,	FIRST_STATE,	FIRST_ZIP,
FIRST_ONNET,	FIRST_INREGION,	FIRST_XOLIT,	FIRST_LATA,
ONNET_PROV,	COMPANY_CODE,	VRD_FLAG,	OPCO_REV,	IN_FOOTPRINT,	NASP_TYPE,
CIR_TECH_TYPE,	CIR_BILL_TYPE,	PROGRAM,	VENDOR,
BIZ_CASE, FIRST_LAT,	FIRST_LGNTD,MIG_STATUS,
SECOND_LAT,	SECOND_LGNTD,	IEN_PROV,	DIV_PORT,	DIV_ACCESS, a.PROD_YR_MTH
,	DISCO_ORD_NUM,DISCO_DATE_BCOM,DISCO_DATE_ORDERING_STRT,DISCO_DATE_BCOM - DISCO_DATE_ORDERING_STRT as DISCO_DURATION

       from edw_sr_vw.rt_cir_single_row_addr a
inner join
(select conv_naspid NASP_ID,CIR_ID,INST_ORD_NUM,CHG_ORD_NUM,MIG_STATUS,DISCO_ORD_NUM,DISCO_DATE_BCOM,DISCO_DATE_ORDERING_STRT
                     from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW
                    where report_date < (select max(report_date) from EDW_GLOB_OPS_VW.CIRCUIT_TDM_TD_VW)
                    -- where (report_date >= '2023-01-31' and report_date <= '2024-01-31')
                    -- where report_date >= '2024-01-31'
                    --and DISCO_ORD_NUM <> ''
                    qualify row_number() over(partition by DISCO_ORD_NUM order by REPORT_DATE desc) = 1
                    ) b
                    ON a.CIR_ID = b.CIR_ID

                    where 
                    PROD_YR_MTH >= 202301 and PROD_YR_MTH <= 202412
                    and REV_LOC_DIV_CODE in ('LRG','PUB','WHL','SAM')
                     -- and DISCO_DATE_BCOM is not null
                    -- and DISCO_DATE_ORDERING_STRT is not null
                     and DISCO_ORD_NUM <> ''
                     and DISCO_DURATION >= 0

                    '''

train_data = run_sql(train_query)