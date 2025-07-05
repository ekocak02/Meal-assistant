# src/meal_assistant/data_ingestion/clean_data.py

import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Import functions from our other data ingestion script
from .load_data import load_food_com_data 

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Cleaning ---
MAX_RECIPE_MINUTES = 3 * 24 * 60 
MIN_RECIPE_MINUTES = 1
VALID_RATINGS = [1.0, 2.0, 3.0, 4.0, 5.0]

def get_project_root() -> Path:
    """Returns the project root folder. This is specific to the script's location."""
    # This needs to be adjusted based on the new location of the script
    # Assuming clean_data.py is at src/meal_assistant/data_ingestion/clean_data.py
    # parents[0] -> .../data_ingestion
    # parents[1] -> .../meal_assistant
    # parents[2] -> .../src
    # parents[3] -> .../meal (project root)
    return Path(__file__).resolve().parents[3]

def parse_iso8601_duration(iso_duration_series: pd.Series) -> pd.Series:
    """
    Parses a pandas Series of ISO 8601 duration strings and converts them to total minutes.
    """
    timedeltas = pd.to_timedelta(iso_duration_series, errors='coerce')
    return timedeltas.dt.total_seconds() / 60

def clean_recipes_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleaning on the raw recipes dataframe.
    """
    logging.info("Starting cleaning process for recipes data...")
    df = df_raw.copy()

    # 1. Parse time columns
    time_columns = ['CookTime', 'PrepTime', 'TotalTime']
    for col in time_columns:
        if col in df.columns:
            df[col] = parse_iso8601_duration(df[col])
    
    # 2. Handle outliers in time columns
    for col in time_columns:
        if col in df.columns:
            invalid_mask = (df[col] < MIN_RECIPE_MINUTES) | (df[col] > MAX_RECIPE_MINUTES)
            if invalid_mask.any():
                logging.warning(f"Found and nullified {invalid_mask.sum()} outlier/invalid entries in '{col}'.")
                df.loc[invalid_mask, col] = np.nan
            
    # 3. Convert RecipeServings to numeric
    if 'RecipeServings' in df.columns:
        df['RecipeServings'] = pd.to_numeric(df['RecipeServings'], errors='coerce')

    logging.info("Cleaning process for recipes finished.")
    return df

def clean_reviews_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleaning on the raw reviews dataframe.
    - Checks for missing values.
    - Validates the 'Rating' column.
    
    Args:
        df_raw (pd.DataFrame): The raw reviews dataframe.

    Returns:
        pd.DataFrame: The cleaned reviews dataframe.
    """
    logging.info("Starting cleaning process for reviews data...")
    df = df_raw.copy()

    # 1. Check for missing essential data
    if df['Review'].isnull().any():
        logging.warning(f"Found {df['Review'].isnull().sum()} reviews with missing text. Keeping them for now.")
    if df['Rating'].isnull().any():
        logging.warning(f"Found {df['Rating'].isnull().sum()} reviews with missing ratings. Keeping them for now.")

    # 2. Validate 'Rating' column values
    # We only want to keep ratings that are in our valid list.
    invalid_mask = ~df['Rating'].isin(VALID_RATINGS) & df['Rating'].notna()
    if invalid_mask.any():
        logging.warning(f"Found and nullified {invalid_mask.sum()} entries with invalid (out of range) ratings.")
        df.loc[invalid_mask, 'Rating'] = np.nan
    
    logging.info("Cleaning process for reviews finished.")
    return df

def main():
    """
    Main function to orchestrate the loading, cleaning, and saving of all data.
    """
    project_root = get_project_root()
    raw_data_path = project_root / "data" / "raw"
    processed_data_path = project_root / "data" / "processed"
    
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_recipes_raw, df_reviews_raw = load_food_com_data(raw_data_path)
    
    # Clean and save recipes data
    if df_recipes_raw is not None:
        df_recipes_cleaned = clean_recipes_data(df_recipes_raw)
        output_file_recipes = processed_data_path / "recipes_cleaned.parquet"
        df_recipes_cleaned.to_parquet(output_file_recipes, index=False)
        logging.info(f"Cleaned recipes data successfully saved to {output_file_recipes}")
    
    # Clean and save reviews data
    if df_reviews_raw is not None:
        df_reviews_cleaned = clean_reviews_data(df_reviews_raw)
        output_file_reviews = processed_data_path / "reviews_cleaned.parquet"
        df_reviews_cleaned.to_parquet(output_file_reviews, index=False)
        logging.info(f"Cleaned reviews data successfully saved to {output_file_reviews}")

if __name__ == "__main__":
    main()