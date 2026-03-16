
#Data loading and initial cleaning utilities.

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and perform initial cleaning of appointment data."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        #Standardize column names.
        df = df.copy()
        
        # Rename columns to snake cases
        column_mapping = {
            'PatientId': 'patient_id',
            'AppointmentID': 'appointment_id',
            'Gender': 'gender',
            'ScheduledDay': 'scheduled_day',
            'AppointmentDay': 'appointment_day',
            'Age': 'age',
            'Neighbourhood': 'neighbourhood',
            'Scholarship': 'scholarship',
            'Hipertension': 'hypertension',
            'Diabetes': 'diabetes',
            'Alcoholism': 'alcoholism',
            'Handcap': 'handicap',
            'SMS_received': 'sms_received',
            'No-show': 'no_show'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        logger.info("Column names standardized")
        return df
    
    def handle_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data quality issues."""
        df = df.copy()
        initial_rows = len(df)
        
        #remove invalid ages
        df = df[df['age'] >= 0]
        df = df[df['age'] <= 115]
        logger.info(f"Removed {initial_rows - len(df)} rows with invalid ages")
        
        #convert date columns
        df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
        df['appointment_day'] = pd.to_datetime(df['appointment_day'])
        
        #remove appointments where scheduled_day > appointment_day (data errors)
        df = df[df['scheduled_day'] <= df['appointment_day']]
        
        #convert no_show to binary (Yes=1, No=0)
        df['no_show'] = (df['no_show'] == 'Yes').astype(int)
        
        #handle missing values (if any)
        df = df.dropna()
        
        logger.info(f"Final dataset after cleaned: {len(df):,} rows")
        return df
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        #save cleaned data to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to the {output_path}")
        
    def load_and_clean(self, save_path: str = None) -> pd.DataFrame:
        # execute full loading and cleaning pipeline
        df = self.load_raw_data()
        df = self.clean_column_names(df)
        df = self.handle_data_quality_issues(df)
        
        if save_path:
            self.save_cleaned_data(df, save_path)
            
        return df


if __name__ == "__main__":
    #usage example
    loader = DataLoader(r"C:\Users\darle\Documents\!Bryan Projects\healthcare-no-show-prediction\data\raw\KaggleV2-May-2016.csv")
    df = loader.load_and_clean(save_path="data/processed/cleaned_data.csv")
    print(df.head())
    print(f"\nNo-show rate: {df['no_show'].mean():.2%}") #mean gives proportion of no-shows