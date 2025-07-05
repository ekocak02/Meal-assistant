# src/meal_assistant/data_ingestion/clean_data.py

import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Import only the necessary function from our library module
# The '.' means "from the current package/directory"
from .load_data import load_food_com_data

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Cleaning ---
MAX_RECIPE_MINUTES = 3 * 24 * 60 
MIN_RECIPE_MINUTES = 1

def get_project_root() -> Path:
    """Returns the project root folder. This is specific to the script's location."""
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

    # 1. Parse time columns from ISO 8601 format to minutes
    logging.info("Parsing time columns to numeric minutes...")
    time_columns = ['CookTime', 'PrepTime', 'TotalTime']
    for col in time_columns:
        if col in df.columns:
            df[col] = parse_iso8601_duration(df[col])
    
    # 2. Handle outliers and illogical values in time columns
    logging.info("Cleaning outliers from time columns...")
    for col in time_columns:
        if col in df.columns:
            invalid_mask = (df[col] < MIN_RECIPE_MINUTES) | (df[col] > MAX_RECIPE_MINUTES)
            num_invalid = invalid_mask.sum()
            
            if num_invalid > 0:
                logging.warning(f"Found and nullified {num_invalid} outlier/invalid entries in '{col}'.")
                df.loc[invalid_mask, col] = np.nan
            
    # 3. Convert RecipeServings to a numeric type, coercing errors
    if 'RecipeServings' in df.columns:
        df['RecipeServings'] = pd.to_numeric(df['RecipeServings'], errors='coerce')

    logging.info("Cleaning process finished.")
    return df

def main():
    """
    Main function to orchestrate the loading, cleaning, and saving of recipe data.
    """
    project_root = get_project_root()
    raw_data_path = project_root / "data" / "raw"
    processed_data_path = project_root / "data" / "processed"
    
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    # Load data by passing the correct path
    df_recipes_raw, _ = load_food_com_data(raw_data_path)
    
    if df_recipes_raw is not None:
        df_recipes_cleaned = clean_recipes_data(df_recipes_raw)
        
        output_file = processed_data_path / "recipes_cleaned.parquet"
        try:
            df_recipes_cleaned.to_parquet(output_file, index=False)
            logging.info(f"Cleaned recipes data successfully saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save cleaned data: {e}")

if __name__ == "__main__":
    main()