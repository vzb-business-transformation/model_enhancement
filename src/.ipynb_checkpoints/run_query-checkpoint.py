import os
import sys
import teradatasql
import oracledb
import pandas as pd
from dotenv import load_dotenv
from config.utils import timing_decorator
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

edw_user = os.getenv('EDW_USER')
edw_pass = os.getenv('EDW_PASS')
bgw_user = os.getenv('BGW_USER')
bgw_pass = os.getenv('BGW_PASS')

@timing_decorator
def run_sql(query):
    with teradatasql.connect(host='tddp.tdc.vzwcorp.com', user=edw_user, password=edw_pass, logmech='ldap') as connect:
        query = query
        df = pd.read_sql(query, connect)
    return df


# Set batch size for fetching large amounts of data
BATCH_SIZE = 10000  # Adjust based on your environment


@timing_decorator
def run_sql1(query, batch_size=BATCH_SIZE):
    """
    Run Teradata SQL query with optimized batch fetching.
    """
    with teradatasql.connect(host='tddp.tdc.vzwcorp.com', user=edw_user, password=edw_pass, logmech='ldap') as connect:
        df_list = []
        try:
            for chunk in pd.read_sql(query, connect, chunksize=batch_size):
                df_list.append(chunk)
            df = pd.concat(df_list, ignore_index=True)
        except Exception as e:
            print(f"Error fetching data: {e}")
            df = pd.DataFrame()  # Return an empty DataFrame on failure
    return df


@timing_decorator
def execute_sql(query):
    """
    Run Teradata multiple SQL query statements
    """
    with teradatasql.connect(host='tddp.tdc.vzwcorp.com', user=edw_user, password=edw_pass, logmech='ldap') as connect:
        try:
            with connect.cursor() as cursor:
                cursor.execute(query)
            print(f"Query executed successfully: {query[:50]}...")
        except Exception as e:
            print(f"Error executing query: {e}")


@timing_decorator
def run_oracle(query):
    with oracledb.connect(user=bgw_user, password=bgw_pass, host="tpapx1-scan", port=1521,
                          service_name="BGWDR") as connect:
        query = query
        df = pd.read_sql(query, connect)
    return df


