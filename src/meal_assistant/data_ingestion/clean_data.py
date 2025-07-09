# src/meal_assistant/data_ingestion/clean_data.py

import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Import functions from our other data ingestion script
from .load_data import load_food_com_data 

# Setup basic logging to provide feedback during script execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Data Cleaning ---
# These constants make the cleaning logic clear and easy to modify.
MAX_RECIPE_MINUTES = 3 * 24 * 60  # 3 days in minutes
MIN_RECIPE_MINUTES = 1
# Based on Kaggle data description, valid ratings are 0 through 5.
VALID_RATINGS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

def get_project_root() -> Path:
    """Returns the project root directory."""
    # This assumes the script is in src/meal_assistant/data_ingestion
    return Path(__file__).resolve().parents[3]

def parse_iso8601_duration(iso_duration_series: pd.Series) -> pd.Series:
    """Converts a pandas Series of ISO 8601 strings to minutes."""
    timedeltas = pd.to_timedelta(iso_duration_series, errors='coerce')
    return timedeltas.dt.total_seconds() / 60

def clean_recipes_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Performs comprehensive cleaning on the raw recipes dataframe based on EDA findings.
    """
    logging.info(f"Starting cleaning process for recipes data. Initial shape: {df_raw.shape}")
    df = df_raw.copy()

    # Action 1: Drop columns with a high percentage of missing values
    df.drop(columns=['RecipeYield'], inplace=True)
    logging.info("Dropped 'RecipeYield' column (>66% missing).")

    # Action 2: Fill missing rating/review data with 0
    # Missing values here imply no reviews, so 0 is a logical fill.
    for col in ['AggregatedRating', 'ReviewCount']:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    logging.info("Filled missing 'AggregatedRating' and 'ReviewCount' with 0.")

    # Action 3: Drop rows with NA in critical, low-missing-rate columns
    critical_cols_to_dropna = ['Description', 'Images', 'RecipeCategory']
    initial_rows = len(df)
    df.dropna(subset=critical_cols_to_dropna, inplace=True)
    rows_dropped = initial_rows - len(df)
    logging.info(f"Dropped {rows_dropped} rows with NA in critical columns like 'RecipeCategory'.")

    # Action 4: Parse time columns and clean outliers
    time_columns = ['CookTime', 'PrepTime', 'TotalTime']
    for col in time_columns:
        if col in df.columns:
            df[col] = parse_iso8601_duration(df[col])
            # Set nonsensical times (e.g., negative or too long) to NaN
            invalid_mask = (df[col] < MIN_RECIPE_MINUTES) | (df[col] > MAX_RECIPE_MINUTES)
            if invalid_mask.any():
                df.loc[invalid_mask, col] = np.nan
    logging.info("Parsed and cleaned time-related columns.")

    # Action 5: Clean outliers in nutritional columns using the IQR method
    nutritional_cols = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
    for col in nutritional_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            if outlier_mask.any():
                df.loc[outlier_mask, col] = np.nan
    logging.info("Cleaned outliers from nutritional columns using IQR method.")
    
    # Action 6: Convert NumPy array columns to standard Python lists
    list_cols = ['Keywords', 'RecipeIngredientParts', 'RecipeInstructions', 'Images']
    for col in list_cols:
        if col in df.columns:
            # Safely convert numpy.ndarray to list, handling potential other types
            df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [])
    logging.info("Converted array-like columns to standard Python lists.")

    logging.info(f"Cleaning process for recipes finished. Final shape: {df.shape}")
    return df

def clean_reviews_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Performs cleaning on the raw reviews dataframe.
    """
    logging.info(f"Starting cleaning process for reviews data. Initial shape: {df_raw.shape}")
    df = df_raw.copy()

    # Validate that ratings are within the 0-5 range.
    logging.info(f"Validating 'Rating' column against the valid set: {VALID_RATINGS}")
    invalid_mask = ~df['Rating'].isin(VALID_RATINGS) & df['Rating'].notna()
    
    if invalid_mask.any():
        num_invalid = invalid_mask.sum()
        logging.warning(f"Found and nullified {num_invalid} ratings outside the {VALID_RATINGS} range.")
        df.loc[invalid_mask, 'Rating'] = np.nan
        
    logging.info(f"Cleaning process for reviews finished. Final shape: {df.shape}")
    return df

def main():
    """
    Main function to orchestrate the loading, cleaning, and saving of all data.
    """
    project_root = get_project_root()
    raw_data_path = project_root / "data" / "raw"
    processed_data_path = project_root / "data" / "processed"

    # Ensure the output directory exists
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    df_recipes_raw, df_reviews_raw = load_food_com_data(raw_data_path)
    
    if df_recipes_raw is not None:
        df_recipes_cleaned = clean_recipes_data(df_recipes_raw)
        output_file_recipes = processed_data_path / "recipes_processed.parquet"
        df_recipes_cleaned.to_parquet(output_file_recipes, index=False)
        logging.info(f"Processed recipes data successfully saved to: {output_file_recipes}")
    
    if df_reviews_raw is not None:
        df_reviews_cleaned = clean_reviews_data(df_reviews_raw)
        output_file_reviews = processed_data_path / "reviews_cleaned.parquet"
        df_reviews_cleaned.to_parquet(output_file_reviews, index=False)
        logging.info(f"Cleaned reviews data successfully saved to: {output_file_reviews}")

if __name__ == "__main__":
    main()