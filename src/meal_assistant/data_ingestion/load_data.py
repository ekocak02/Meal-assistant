# src/meal_assistant/data_ingestion/load_data.py

import pandas as pd
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).resolve().parents[3]

def load_food_com_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the recipes and reviews data from the raw parquet files.

    Args:
        data_path (Path): The path to the 'raw' data directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the recipes and reviews dataframes.
                                           Returns (None, None) if files are not found.
    """
    recipes_file = data_path / "recipes.parquet"
    reviews_file = data_path / "reviews.parquet"

    if not recipes_file.exists() or not reviews_file.exists():
        logging.error(f"Data files not found in {data_path}. Ensure 'recipes.parquet' and 'reviews.parquet' are present.")
        return None, None

    try:
        logging.info(f"Loading recipes data from {recipes_file}...")
        df_recipes = pd.read_parquet(recipes_file)
        logging.info("Recipes data loaded successfully.")

        logging.info(f"Loading reviews data from {reviews_file}...")
        df_reviews = pd.read_parquet(reviews_file)
        logging.info("Reviews data loaded successfully.")
        
        return df_recipes, df_reviews

    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        return None, None

if __name__ == "__main__":
    # This block allows the script to be run directly from the command line
    project_root = get_project_root()
    raw_data_path = project_root / "data" / "raw"
    
    df_recipes, df_reviews = load_food_com_data(raw_data_path)

    if df_recipes is not None and df_reviews is not None:
        logging.info("--- Recipes DataFrame Info ---")
        df_recipes.info(verbose=False)
        print("\n")
        
        logging.info("--- First 5 Recipes ---")
        print(df_recipes.head())
        print("\n")
        
        logging.info("--- Reviews DataFrame Info ---")
        df_reviews.info(verbose=False)
        print("\n")

        logging.info("--- First 5 Reviews ---")
        print(df_reviews.head())