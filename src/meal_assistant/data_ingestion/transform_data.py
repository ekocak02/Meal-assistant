# src/meal_assistant/data_ingestion/transform_data.py

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parents[3]

def parse_ingredient(ingredient_str: str) -> dict:
    """
    Parses a single ingredient string into quantity, unit, and ingredient name
    using a more robust regex pattern.
    
    Args:
        ingredient_str (str): A single ingredient string.

    Returns:
        dict: A dictionary with 'quantity', 'unit', and 'ingredient' keys.
    """
    # Units list now includes singular and plural forms for better matching.
    UNITS = ['cup', 'cups', 'c', 'teaspoon', 'teaspoons', 'tsp', 'tablespoon', 'tablespoons', 'tbsp', 
             'ounce', 'ounces', 'oz', 'pound', 'pounds', 'lb', 'lbs', 'gram', 'grams', 'g', 'kg',
             'clove', 'cloves', 'pinch', 'dash', 'can', 'cans', 'slice', 'slices']
    
    # REVISED Regex Pattern:
    # - ^\s* -> Starts with optional whitespace.
    # - (?P<quantity>...)      -> Named group for quantity. Handles numbers (1), fractions (1/2), decimals (1.5). Optional.
    # - \s+                    -> Space between quantity and unit.
    # - (?P<unit>...)          -> Named group for unit. Uses word boundaries (\b) to match whole words only. Optional.
    # - \s* -> Optional space.
    # - (?P<ingredient>.+)     -> The rest of the string is the ingredient.
    pattern = re.compile(
        r"^\s*(?P<quantity>(\d+\s*/\s*\d+)|(\d+\s*-\s*\d+)|(\d*\.\d+)|\d+)?\s*"
        r"(?P<unit>(\b(" + "|".join(UNITS) + r")\b))?\s*"
        r"(?P<ingredient>.+)"
    )
    
    match = pattern.match(ingredient_str.lower())
    
    if match:
        data = match.groupdict()
        # Clean up the ingredient part if it was matched
        data['ingredient'] = data['ingredient'].strip()
        return data
    else:
        # Fallback for patterns the regex doesn't catch
        return {"quantity": None, "unit": None, "ingredient": ingredient_str.strip()}

def main():
    """
    Main function to transform recipe ingredients into a canonical schema.
    """
    project_root = get_project_root()
    processed_data_path = project_root / "data" / "processed"
    recipes_file = processed_data_path / "recipes_processed.parquet"

    if not recipes_file.exists():
        logging.error(f"Processed recipes file not found at {recipes_file}. Please run the clean_data.py script first.")
        return

    logging.info(f"Loading processed recipes from {recipes_file}...")
    df_recipes = pd.read_parquet(recipes_file)

    # We only need the ID and the ingredients list for this task
    df_ingredients = df_recipes[['RecipeId', 'RecipeIngredientParts']].copy()

    # Explode the list of ingredients into separate rows for each ingredient
    logging.info("Exploding ingredient lists into individual rows...")
    df_exploded = df_ingredients.explode('RecipeIngredientParts').dropna()
    df_exploded.rename(columns={'RecipeIngredientParts': 'ingredient_string'}, inplace=True)

    # Apply the parsing function to each ingredient string
    logging.info("Parsing ingredient strings using regex...")
    parsed_data = df_exploded['ingredient_string'].apply(parse_ingredient)
    df_parsed = pd.json_normalize(parsed_data)

    # Combine the parsed data back with the RecipeId
    df_canonical = pd.concat([df_exploded.reset_index(drop=True), df_parsed], axis=1)

    # Save the canonical ingredients table
    output_file = processed_data_path / "canonical_ingredients.parquet"
    df_canonical.to_parquet(output_file, index=False)
    logging.info(f"Canonical ingredients data successfully saved to: {output_file}")
    logging.info(f"Sample of the final canonical data:\n{df_canonical.head()}")


if __name__ == "__main__":
    main()