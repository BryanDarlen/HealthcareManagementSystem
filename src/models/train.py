""" 
Model training pipeline for no-show prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, fbeta_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoShowModelTrainer:
    """Train & Evaluate No-Show Prediction Models."""

    #the random_state default value is set to 42, unless specified when called
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {} #empty dictionary
        self.results = {}
        self.best_model = None #nothing assigned yet
        self.feature_names = None

    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the featured engineered data."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)

        #seperate features and target
        X = df.drop('no_show', axis = 1) #drops a column called no_show
        y = df['no_show']

        self.feature_names = X.columns.tolist() #gives column names then converts to list

        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y #gives back to the line that's called

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train no-show rate: {y_train.mean():.2%}")
        logger.info(f"Test no-show rate: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """Apply SMOTE for class imbalance."""
        if method == 'smote':
            logger.info("Applying SMOTE for the class balancing")
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            logger.info(f"Original distribution: {pd.Series(y_train).value_counts().to_dict()}")
            logger.info(f"Balanced distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

            return X_train_balanced, y_train_balanced
        else:
            return X_train, y_train

    def train_lightbgm(self, X_train, y_train, params: Dict = None):
        """Train LightBHM model."""
        logger.info("Training LightBHM model")

        if params is None:
            # Optimized parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 7,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'random_state': self.random_state,
                'n_estimators': 500,
                'scale_pos_weight': 3.5  # Adjust for class imbalance
            }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        self.models['lightbgm'] = model
        logger.info("LightBGM training complete")

        return model

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model."""
        logger.info("Training Random Forest model")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        logger.info("Random Forest training complete")
        
        return model

    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression baseline."""
        logger.info("Training Logistic Regression baseline")

        #Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state,
            solver='liblinear'
        )

        model.fit(X_train_scaled, y_train)
        #the reason need to scale before logistic regression is to make sure no one side
        #has bigger value than the others

        self.models['logistic_regression'] = model
        self.models['scaler'] = scaler
        logger.info("Logistic Regression training complete")

        return model

    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Assessing Model Performance."""
        logger.info(f"Evaluating: {model_name}")

        #handle scaling for logistic regression (to make sure it executed the steps before properly)
        if model_name == 'logistic_regression' and 'scaler' in self.models:
            X_test = self.models['scaler'].transform(X_test)

        #predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        #calculate the metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2) #prioritize recall
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        #confusion matrix
        cm = confusion_matrix(y_test, y_pred) #TN,FP,FN,TP
        tn, fp, fn, tp = cm.ravel() #numpy library, ravel() converts 2d array to 1d array

        #business metrics
        cost_per_intervention = 12 #$3 SMS + $9 staff time
        revenue_per_appointment = 200

        #false negatives are missed no-shows (lost revenue)
        cost_fn = fn * revenue_per_appointment

        #false positives are unnecessary interventions (wasted cost)
        cost_fp = fp * cost_per_intervention

        #true positives are prevented no-shows (saved revenue - intervention cost)
        saved_tp = tp * (revenue_per_appointment - cost_per_intervention)

        total_cost = cost_fn  + cost_fp
        total_benefit = saved_tp
        net_benefit = total_benefit - total_cost      
        roi = (net_benefit / (tp * cost_per_intervention)) if tp > 0 else 0 #roi = return of investment  

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'cost_false_negatives': float(cost_fn),
            'cost_false_positives': float(cost_fp),
            'benefit_true_positives': float(saved_tp),
            'net_benefit': float(net_benefit),
            'roi': float(roi)            
        }

        self.results[model_name] = results

        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"  F2 Score:  {f2:.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        logger.info(f"\n  Business Metrics:")
        logger.info(f"  Net Benefit: ${net_benefit:,.2f}")
        logger.info(f"  ROI: {roi:.2f}x")  

        return results     

    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        """Plot confusion matrix."""
        cm = np.array(self.results[model_name]['confusion_matrix'])

        plt.figure(figsize=(8,6)) #create the canvas
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted: Show', 'Predicted: No-Show'],
                    yticklabels=['Actual: Show', 'Actual: No-Show'])#draws heatmap on that axes  
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(self, model_name: str = 'lightbgm', top_n: int = 20, save_path: str = None):
        """Plot feature importance."""
        model = self.models[model_name]

        if model_name == 'lightbgm' or model_name == 'random_forest':
            importance = model.feature_importances_ #feature_importances_ is built-in after using model.fit()
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n) #descend order and only keeps the first 'top_n' rows after sorting

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feature_importance)), feature_importance['importance'].values)
            plt.yticks(range(len(feature_importance)), feature_importance['feature'].values)
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances = {model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    def compare_models(self):
        """Compare all trained models."""
        comparison = pd.DataFrame(self.results).T # .T is from pandas, which to transpose attribute for a DataFrame or Series
        
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score']
        comparison_display = comparison[metrics_to_show]

        print("\n" + "=" * 78)
        print(" " * 30 + "MODEL COMPARISON")
        print("=" * 80)
        print(comparison_display.to_string())
        print("=" * 80)

        #select best model based on F2 score (prioritizes recall)
        best_model_name = comparison['f2_score'].idxmax()
        self.best_model = self.models[best_model_name]

        logger.info(f"\nBest Model: {best_model_name} (F2 Score: {comparison.loc[best_model_name, 'f2_score']:.4f})")

        return comparison_display

    def save_model(self, model_name: str, output_dir: str = 'models'):
        """Save trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model = self.models[model_name]
        model_file = output_path / f"{model_name}_model.pkl"

        joblib.dump(model, model_file)
        logger.info(f"Model saved to {model_file}")

        #save feature names
        feature_file = output_path / f"{model_name}_features.json"
        with open(feature_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        #save results
        results_file = output_path / f"{model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results[model_name], f, indent=2)

        logger.info(f"Model artifacts saved to {output_dir}/")

    def cross_validate_model(self, X, y, model_name: str = 'lightbgm', cv: int = 5):
        """Perform cross-validation."""
        logger.info(f"Performing {cv}-fold cross-validation for {model_name}")
        
        model = self.models[model_name]
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = cross_validate(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        print(f"\n{cv}-Fold Cross-Validation Results for {model_name}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            print(f"  {metric.capitalize():12} - Train: {train_scores.mean():.4f} (±{train_scores.std():.4f}) | "
                  f"Test: {test_scores.mean():.4f} (±{test_scores.std():.4f})")
        
        return cv_results


def main():
    """Main training pipeline."""
    print("🏥 Healthcare No-Show Prediction Model Training")
    print("=" * 50)

    BASE_DIR = Path(__file__).resolve().parents[2]
    feature_data_path = BASE_DIR / "data" / "features" / "engineered_features.csv"
    figures_dir = BASE_DIR / "docs" / "figures"
    models_dir = BASE_DIR / "models"

    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    trainer = NoShowModelTrainer(random_state=42)

    print("Loading data...")
    X, y = trainer.load_data(feature_data_path)
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.mean():.2%} no-shows")

    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)

    print("\n" + "=" * 80)
    print(" " * 30 + "TRAINING MODELS")
    print("=" * 80 + "\n")

    trainer.train_logistic_regression(X_train, y_train)
    trainer.evaluate_model(trainer.models["logistic_regression"], X_test, y_test, "logistic_regression")

    trainer.train_random_forest(X_train, y_train)
    trainer.evaluate_model(trainer.models["random_forest"], X_test, y_test, "random_forest")

    trainer.train_lightbgm(X_train, y_train)
    trainer.evaluate_model(trainer.models["lightbgm"], X_test, y_test, "lightbgm")

    trainer.compare_models()
    trainer.cross_validate_model(X, y, model_name="lightbgm", cv=5)

    print("\nGenerating visualizations...")
    trainer.plot_confusion_matrix("lightbgm", save_path=figures_dir / "confusion_matrix_lightbgm.png")
    trainer.plot_feature_importance("lightbgm", top_n=20, save_path=figures_dir / "feature_importance.png")

    trainer.save_model("lightbgm", output_dir=models_dir)

    print("\n✅ Training pipeline complete!")
    print("🏆 Best model: lightbgm")
    print(f"📁 Model saved to: {models_dir / 'lightbgm_model.pkl'}")
    print(f"📁 Visualizations saved to: {figures_dir}")


if __name__ == "__main__":
    main()
