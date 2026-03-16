#engineering feature appointment no show prediction
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    #feature engineer for no show prediction
    
    def __init__(self):
        pass
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        #create time based feature
        df = df.copy()
        
        # Lead time (days between scheduling and appointment)
        df['lead_time_days'] = (df['appointment_day'] - df['scheduled_day']).dt.days
        
        # Appointment day features
        df['appointment_day_of_week'] = df['appointment_day'].dt.dayofweek
        df['appointment_month'] = df['appointment_day'].dt.month
        df['appointment_day_of_month'] = df['appointment_day'].dt.day
        df['appointment_hour'] = df['appointment_day'].dt.hour
        
        # Boolean indicators
        df['is_monday'] = (df['appointment_day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['appointment_day_of_week'] == 4).astype(int)
        df['is_weekend'] = (df['appointment_day_of_week'] >= 5).astype(int) #1 as True and 0 as False
        
        # Scheduled day features
        df['scheduled_day_of_week'] = df['scheduled_day'].dt.dayofweek #takes the day from the week
        df['scheduled_same_day'] = (df['lead_time_days'] == 0).astype(int)
        
        # Lead time categories
        df['lead_time_category'] = pd.cut(
            df['lead_time_days'],
            bins=[-1, 0, 3, 7, 14, 30, 999],
            labels=['same_day', '1-3_days', '4-7_days', '1-2_weeks', '2-4_weeks', '1month_plus']
        )
        
        logger.info("Created temporal features")
        return df
    
    def create_patient_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create patient historical behavior features."""
        df = df.copy()
        
        # Sort by patient and date
        df = df.sort_values(['patient_id', 'scheduled_day'])
        
        # Count total appointments per patient
        df['patient_total_appointments'] = df.groupby('patient_id').cumcount() + 1
        
        # Calculate historical no-show rate
        df['patient_cumulative_no_shows'] = df.groupby('patient_id')['no_show'].cumsum()
        df['patient_no_show_rate'] = (
            df['patient_cumulative_no_shows'] / df['patient_total_appointments']
        ).fillna(0)
        
        # Previous appointment no-show
        df['previous_no_show'] = df.groupby('patient_id')['no_show'].shift(1).fillna(0)
        
        # Days since last appointment
        df['days_since_last_appointment'] = (
            df.groupby('patient_id')['appointment_day']
            .diff()
            .dt.days
            .fillna(0)
        )
        
        # Consecutive shows streak
        df['show_streak'] = 0  # Simplified for now
        
        logger.info("Created patient history features")
        return df
    
    def create_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create health-related features."""
        df = df.copy()
        
        # Count chronic conditions
        condition_cols = ['hypertension', 'diabetes', 'alcoholism']
        df['chronic_condition_count'] = df[condition_cols].sum(axis=1)
        df['has_chronic_condition'] = (df['chronic_condition_count'] > 0).astype(int)
        
        # Handicap severity
        df['has_handicap'] = (df['handicap'] > 0).astype(int)
        
        logger.info("Created health features")
        return df
    
    def create_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create social determinant features."""
        df = df.copy()
        
        # Neighborhood-level aggregations
        neighborhood_stats = df.groupby('neighbourhood').agg({
            'no_show': ['mean', 'count']
        }).reset_index()
        neighborhood_stats.columns = ['neighbourhood', 'neighbourhood_no_show_rate', 'neighbourhood_appointment_count']
        
        df = df.merge(neighborhood_stats, on='neighbourhood', how='left')
        
        # Social risk score (composite)
        df['social_risk_score'] = (
            df['scholarship'] * 0.3 +
            (df['neighbourhood_no_show_rate'] * 100) * 0.7
        )
        
        logger.info("Created social features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df = df.copy()
        
        # Age-related interactions
        df['age_lead_time_interaction'] = df['age'] * df['lead_time_days']
        df['age_category'] = pd.cut(
            df['age'],
            bins=[0, 18, 35, 50, 65, 120],
            labels=['child', 'young_adult', 'middle_age', 'senior', 'elderly']
        )
        
        # SMS effectiveness by patient history
        df['sms_with_history'] = df['sms_received'] * df['patient_no_show_rate']
        
        logger.info("Created interaction features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        # One-hot encode gender
        df = pd.get_dummies(df, columns=['gender'], prefix='gender', drop_first=True)
        
        # One-hot encode lead time category
        df = pd.get_dummies(df, columns=['lead_time_category'], prefix='lead_cat', drop_first=False)
        
        # One-hot encode age category
        df = pd.get_dummies(df, columns=['age_category'], prefix='age_cat', drop_first=False)
        
        # Label encode neighborhood (too many categories for one-hot)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['neighbourhood_encoded'] = le.fit_transform(df['neighbourhood'])
        
        logger.info("Encoded categorical features")
        return df
    
    def select_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final features for modeling."""
        
        # Features to keep
        feature_columns = [
            # Temporal
            'lead_time_days', 'appointment_day_of_week', 'appointment_month',
            'is_monday', 'is_friday', 'is_weekend', 'scheduled_same_day',
            
            # Patient history
            'patient_total_appointments', 'patient_no_show_rate',
            'previous_no_show', 'days_since_last_appointment',
            
            # Demographics
            'age', 
            
            # Health
            'chronic_condition_count', 'has_chronic_condition',
            'hypertension', 'diabetes', 'alcoholism', 'has_handicap',
            
            # Social
            'scholarship', 'sms_received', 'social_risk_score',
            'neighbourhood_no_show_rate', 'neighbourhood_encoded',
            
            # Interactions
            'age_lead_time_interaction', 'sms_with_history',
            
            # Target
            'no_show'
        ]
        
        # Add one-hot encoded columns
        gender_cols = [col for col in df.columns if col.startswith('gender_')]
        lead_cat_cols = [col for col in df.columns if col.startswith('lead_cat_')]
        age_cat_cols = [col for col in df.columns if col.startswith('age_cat_')]
        
        all_features = feature_columns + gender_cols + lead_cat_cols + age_cat_cols
        
        # Filter to available columns
        available_features = [col for col in all_features if col in df.columns]
        
        df_final = df[available_features].copy()
        
        logger.info(f"Selected {len(available_features)} features for modeling")
        return df_final
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline")
        
        df = self.create_temporal_features(df)
        df = self.create_patient_history_features(df)
        df = self.create_health_features(df)
        df = self.create_social_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df)
        df = self.select_model_features(df)
        
        logger.info("Feature engineering complete")
        logger.info(f"Final shape: {df.shape}")
        
        return df


if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv("data/processed/cleaned_data.csv")
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_all_features(df)
    
    # Save
    df_features.to_csv(r"C:\Users\darle\Documents\!Bryan Projects\healthcare-no-show-prediction\data\features\engineered_features.csv", index=False)
    print(f"Engineered features saved. Shape: {df_features.shape}")
    print(f"\nNo-show rate: {df_features['no_show'].mean():.2%}")
    print(f"\nFeature list:\n{df_features.columns.tolist()}")
