import openml
import pandas as pd
from datetime import datetime

def search_fraud_datasets():
    """
    Search for fraud detection datasets on OpenML
    """
    print("SEARCHING FOR FRAUD DETECTION DATASETS ON OPENML")
    print("=" * 60)
    
    # Search for datasets with "fraud" in the name
    try:
        # Search for fraud-related datasets
        fraud_datasets = openml.datasets.list_datasets(
            output_format='dataframe',
            tag='fraud'
        )
        
        if len(fraud_datasets) > 0:
            print(f"Found {len(fraud_datasets)} fraud-related datasets:")
            print()
            
            for _, dataset in fraud_datasets.iterrows():
                print(f"Dataset ID: {dataset['did']}")
                print(f"Name: {dataset['name']}")
                print(f"Instances: {dataset['NumberOfInstances']}")
                print(f"Features: {dataset['NumberOfFeatures']}")
                print(f"Classes: {dataset['NumberOfClasses']}")
                print(f"Upload Date: {dataset['upload_date']}")
                print("-" * 40)
        else:
            print("No fraud datasets found with 'fraud' tag")
            
    except Exception as e:
        print(f"Error searching datasets: {e}")
    
    # Search for credit card fraud specifically
    print("\nSEARCHING FOR CREDIT CARD FRAUD DATASETS")
    print("=" * 50)
    
    try:
        # Common fraud dataset IDs
        fraud_dataset_ids = [
            1597,   # Credit Card Fraud Detection
            42178,  # IEEE-CIS Fraud Detection
            42179,  # IEEE-CIS Fraud Detection (smaller version)
            42180,  # IEEE-CIS Fraud Detection (balanced)
            42181,  # IEEE-CIS Fraud Detection (imbalanced)
            42182,  # IEEE-CIS Fraud Detection (subset)
            42183,  # IEEE-CIS Fraud Detection (subset 2)
            42184,  # IEEE-CIS Fraud Detection (subset 3)
            42185,  # IEEE-CIS Fraud Detection (subset 4)
            42186,  # IEEE-CIS Fraud Detection (subset 5)
        ]
        
        available_datasets = []
        
        for dataset_id in fraud_dataset_ids:
            try:
                dataset = openml.datasets.get_dataset(dataset_id)
                print(f"Dataset ID: {dataset_id}")
                print(f"Name: {dataset.name}")
                if dataset.qualities is not None:
                    print(f"Instances: {dataset.qualities.get('NumberOfInstances', 'Unknown')}")
                    print(f"Features: {dataset.qualities.get('NumberOfFeatures', 'Unknown')}")
                    print(f"Classes: {dataset.qualities.get('NumberOfClasses', 'Unknown')}")
                else:
                    print("Instances: Unknown")
                    print("Features: Unknown")
                    print("Classes: Unknown")
                print(f"Target: {dataset.default_target_attribute}")
                print("-" * 40)
                available_datasets.append(dataset_id)
            except Exception as e:
                print(f"Dataset {dataset_id} not available: {e}")
                continue
        
        return available_datasets
        
    except Exception as e:
        print(f"Error searching specific datasets: {e}")
        return []

def load_fraud_dataset(dataset_id=1597):
    """
    Load a specific fraud detection dataset
    """
    print(f"\nLOADING FRAUD DATASET {dataset_id}")
    print("=" * 40)
    
    try:
        # Get dataset
        dataset = openml.datasets.get_dataset(dataset_id)
        print(f"Dataset: {dataset.name}")
        
        # Load data
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", 
            target=dataset.default_target_attribute
        )
        
        print(f"Data shape: {X.shape}")
        print(f"Target variable: {dataset.default_target_attribute}")
        print(f"Target distribution:")
        print(pd.Series(y).value_counts())
        
        # Show sample data
        print(f"\nSample data (first 5 rows):")
        print(pd.DataFrame(X).head())
        
        # Show feature information
        print(f"\nFeature information:")
        print(f"Total features: {X.shape[1]}")
        print(f"Categorical features: {sum(categorical_indicator)}")
        print(f"Numerical features: {X.shape[1] - sum(categorical_indicator)}")
        
        # Show categorical features
        if sum(categorical_indicator) > 0:
            print(f"\nCategorical features:")
            for i, is_categorical in enumerate(categorical_indicator):
                if is_categorical:
                    print(f"  - {attribute_names[i]}")
        
        return X, y, dataset.name, categorical_indicator, attribute_names
        
    except Exception as e:
        print(f"Error loading dataset {dataset_id}: {e}")
        return None, None, None, None, None

def main():
    """
    Main function to search and load fraud datasets
    """
    # Search for available datasets
    available_datasets = search_fraud_datasets()
    
    if available_datasets:
        print(f"\nAvailable fraud datasets: {available_datasets}")
        
        # Try to load the first available dataset
        dataset_id = available_datasets[0]
        X, y, dataset_name, categorical_indicator, attribute_names = load_fraud_dataset(dataset_id)
        
        if X is not None:
            # Save dataset info
            dataset_info = {
                'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'shape': X.shape,
                'target_distribution': pd.Series(y).value_counts().to_dict(),
                'categorical_features': [attribute_names[i] for i, is_cat in enumerate(categorical_indicator) if is_cat] if categorical_indicator and attribute_names else [],
                'numerical_features': [attribute_names[i] for i, is_cat in enumerate(categorical_indicator) if not is_cat] if categorical_indicator and attribute_names else [],
                'loaded_at': datetime.now().isoformat()
            }
            
            # Save to file
            import json
            with open('fraud_dataset_info.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            print(f"\nDataset info saved to 'fraud_dataset_info.json'")
            
            # Save data to CSV
            data_with_target = pd.DataFrame(X)
            data_with_target['Class'] = y
            data_with_target.to_csv('fraud_dataset.csv', index=False)
            
            print(f"Dataset saved to 'fraud_dataset.csv'")
            print(f"Ready for model training!")
            
            return X, y, dataset_name
        else:
            print("Failed to load dataset")
            return None, None, None
    else:
        print("No fraud datasets found")
        return None, None, None

if __name__ == "__main__":
    main() 