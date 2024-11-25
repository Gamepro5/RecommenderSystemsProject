# %% [markdown]
# # Get and preprocess the data

# %%
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import os
import logging
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize data processor
        Args:
            data_dir (str): Directory to store processed data files
        """
        self.data_dir = data_dir
        self._ensure_data_dir()
        
    def _ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")

    def process_or_load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main function to either process raw data or load processed CSVs
        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
        # Check if processed files exist
        files_exist = all(
            os.path.exists(os.path.join(self.data_dir, f"{split}_electronics.csv"))
            for split in ['train', 'valid', 'test']
        )

        if files_exist:
            logger.info("Loading pre-processed data files...")
            return self._load_processed_data()
        
        logger.info("Processing raw data...")
        return self._process_raw_data()

    def _load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load pre-processed CSV files"""
        train_df = pd.read_csv(os.path.join(self.data_dir, "train_electronics.csv"))
        valid_df = pd.read_csv(os.path.join(self.data_dir, "valid_electronics.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test_electronics.csv"))
        
        logger.info(f"Loaded data shapes - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
        return train_df, valid_df, test_df

    def _process_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process raw data and save to CSV"""
        # Load dataset
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", split="full")
        
        # Process interactions
        interactions = self._load_ratings(dataset)
        
        # Split data
        train, valid, test = self._split_data(interactions)
        
        # Convert to dataframes and save
        train_df, valid_df, test_df = self._save_splits(train, valid, test)
        
        return train_df, valid_df, test_df

    def _load_ratings(self, dataset) -> List[Tuple]:
        """Process ratings from raw dataset"""
        inters = []
        error_count = 0
        total_records = 0
        
        for record in tqdm(dataset, desc="Processing records"):
            total_records += 1
            try:
                user = record["reviewerID"]
                item = record["asin"]
                rating = float(record["overall"])
                timestamp = int(record["unixReviewTime"])
                
                if not (1 <= rating <= 5):
                    error_count += 1
                    continue
                    
                inters.append((user, item, rating, timestamp))
                
            except (KeyError, ValueError, TypeError):
                error_count += 1
                continue
        
        logger.info(f"Processed {total_records} records with {error_count} errors")
        logger.info(f"Successfully loaded {len(inters)} interactions")
        
        return inters

    def _split_data(self, inters: List[Tuple]) -> Tuple[List, List, List]:
        """Split interactions into train/valid/test sets"""
        user2inters = defaultdict(list)
        for inter in inters:
            user2inters[inter[0]].append(inter)

        train, valid, test = [], [], []
        
        for user, interactions in user2inters.items():
            interactions.sort(key=lambda x: x[3])
            if len(interactions) >= 3:
                test.append(interactions[-1])
                valid.append(interactions[-2])
                train.extend(interactions[:-2])
            elif len(interactions) == 2:
                test.append(interactions[-1])
                train.append(interactions[0])
            elif len(interactions) == 1:
                train.append(interactions[0])
                
        logger.info(f"Split sizes - Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
        return train, valid, test

    def _save_splits(self, train: List, valid: List, test: List) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert splits to dataframes and save to CSV"""
        dfs = {}
        
        for name, data in [("train", train), ("valid", valid), ("test", test)]:
            df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])
            
            # Save to CSV
            file_path = os.path.join(self.data_dir, f"{name}_electronics.csv")
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {name} set to {file_path}")
            
            dfs[name] = df
            
        return dfs["train"], dfs["valid"], dfs["test"]

def main():
    # Initialize processor
    processor = DataProcessor()
    
    # Process or load data
    train_df, valid_df, test_df = processor.process_or_load_data()
    
    # Print some basic statistics
    for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        print(f"\n{name} Set Statistics:")
        print(f"Shape: {df.shape}")
        print(f"Unique users: {df['user_id'].nunique()}")
        print(f"Unique items: {df['item_id'].nunique()}")
        print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")

if __name__ == "__main__":
    main()

# %% [markdown]
# # Exploratory analysis

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Tuple, Dict, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """
    Validate a dataframe's structure and content
    """
    required_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    # Check if dataframe is empty
    if df.empty:
        raise DataValidationError(f"DataFrame {name} is empty")
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns in {name}: {missing_cols}")
    
    # Check data types
    expected_types = {
        'rating': ['float64', 'float32', 'int64', 'int32'],
        'timestamp': ['int64', 'int32', 'float64', 'float32']
    }
    
    for col, expected in expected_types.items():
        if df[col].dtype.name not in expected:
            logger.warning(f"Column {col} in {name} has unexpected type {df[col].dtype.name}")
            # Try to convert to appropriate type
            try:
                df[col] = df[col].astype('float64')
            except Exception as e:
                raise DataValidationError(f"Could not convert {col} to numeric type: {e}")

    # Validate value ranges
    if not (df['rating'].between(1, 5).all()):
        invalid_ratings = df[~df['rating'].between(1, 5)].index
        logger.warning(f"Found invalid ratings at indices: {invalid_ratings}")
        df.loc[~df['rating'].between(1, 5), 'rating'] = df['rating'].clip(1, 5)

    return df

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the datasets with extensive error handling
    """
    try:
        datasets = {}
        for name in ['train', 'valid', 'test']:
            file_path = f"{name}_electronics.csv"
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load data with error handling for corrupt files
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed, trying with latin1 encoding for {file_path}")
                df = pd.read_csv(file_path, encoding='latin1')
            
            # Validate and clean the dataframe
            df = validate_dataframe(df, name)
            datasets[name] = df
        
        # Combine datasets
        all_data = pd.concat([datasets['train'], datasets['valid'], datasets['test']], 
                           axis=0, ignore_index=True)
        
        # Convert timestamp safely
        try:
            all_data['datetime'] = pd.to_datetime(all_data['timestamp'], unit='s')
        except Exception as e:
            logger.error(f"Failed to convert timestamps: {e}")
            all_data['datetime'] = pd.to_datetime('now')  # Fallback to current time
        
        return datasets['train'], datasets['valid'], datasets['test'], all_data
        
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise

def calculate_safe_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate basic statistics with safety checks
    """
    stats = {}
    try:
        stats['Total Reviews'] = len(df)
        stats['Unique Users'] = df['user_id'].nunique()
        stats['Unique Items'] = df['item_id'].nunique()
        
        # Safe mean calculation
        stats['Average Rating'] = df['rating'].mean() if len(df) > 0 else 0
        stats['Rating Std'] = df['rating'].std() if len(df) > 0 else 0
        
        # Safe datetime range calculation
        if 'datetime' in df.columns and len(df) > 0:
            stats['Date Range'] = f"{df['datetime'].min()} to {df['datetime'].max()}"
        else:
            stats['Date Range'] = "No date range available"
        
        # Safe division for averages
        stats['Reviews per User'] = (len(df) / df['user_id'].nunique()) if df['user_id'].nunique() > 0 else 0
        stats['Reviews per Item'] = (len(df) / df['item_id'].nunique()) if df['item_id'].nunique() > 0 else 0
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {"Error": str(e)}

def plot_safe_distribution(df: pd.DataFrame, title: str, column: str) -> None:
    """
    Create plots with error handling
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=column, bins=30)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()
    except Exception as e:
        logger.error(f"Error creating plot {title}: {e}")
        plt.close()  # Clean up in case of error

def analyze_temporal_patterns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Analyze temporal patterns with error handling
    """
    try:
        if 'datetime' not in df.columns or df.empty:
            raise ValueError("No datetime column or empty dataframe")
            
        # Group by month with error handling
        monthly_stats = df.set_index('datetime').resample('M').agg({
            'rating': ['count', 'mean'],
            'user_id': 'nunique',
            'item_id': 'nunique'
        }).fillna(0)
        
        return monthly_stats
    except Exception as e:
        logger.error(f"Error in temporal analysis: {e}")
        return None

def main_analysis():
    """
    Run the complete analysis with error handling
    """
    results = {}
    try:
        # Load and prepare data
        logger.info("Loading data...")
        train_df, valid_df, test_df, all_data = load_and_prepare_data()
        results['data'] = {'train': train_df, 'valid': valid_df, 'test': test_df, 'all': all_data}
        
        # Calculate statistics
        logger.info("Calculating statistics...")
        results['statistics'] = calculate_safe_statistics(all_data)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_safe_distribution(all_data, 'Rating Distribution', 'rating')
        
        # Temporal analysis
        logger.info("Performing temporal analysis...")
        results['temporal'] = analyze_temporal_patterns(all_data)
        
        # User-Item statistics
        logger.info("Analyzing user-item patterns...")
        if len(all_data) > 0:
            results['user_stats'] = all_data.groupby('user_id')['rating'].agg(['count', 'mean', 'std']).fillna(0)
            results['item_stats'] = all_data.groupby('item_id')['rating'].agg(['count', 'mean', 'std']).fillna(0)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    try:
        results = main_analysis()
        if 'error' not in results:
            print("\nAnalysis completed successfully!")
            print("\nBasic Statistics:")
            print(pd.Series(results['statistics']))
        else:
            print(f"\nAnalysis failed: {results['error']}")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")

# %%



