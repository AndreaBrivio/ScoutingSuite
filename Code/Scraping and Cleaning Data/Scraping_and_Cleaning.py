import pandas as pd
import numpy as np
import time
import io
import sys
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

"""
FBREF DATA SCRAPER & PROCESSOR
------------------------------
This script scrapes player statistics from FBref for the "Big 5" European leagues.
It handles multiple statistical categories (shooting, passing, etc.), merges them,
cleans the data, calculates advanced metrics (per 90s), and handles players 
who played for multiple teams in a single season.
"""

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# URLs for different statistical categories on FBref
URLS = {
    "standard": f"https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats",
    "shooting": f"https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats",
    "passing": f"https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats",
    "passing_types": f"https://fbref.com/en/comps/Big5/passing_types/players/Big-5-European-Leagues-Stats",
    "gca": f"https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats",
    "defense": f"https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats",
    "possession": f"https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats",
    "misc": f"https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats",
}

# Corresponding HTML table IDs in the FBref DOM
TABLE_IDS = {
    "standard": "stats_standard",
    "shooting": "stats_shooting",
    "passing": "stats_passing",
    "passing_types": "stats_passing_types",
    "gca": "stats_gca",
    "defense": "stats_defense",
    "possession": "stats_possession",
    "misc": "stats_misc",
}

# Mapping FBref's short/abbreviated headers to descriptive names
COLUMN_MAPPING = {
    'Player': 'Name', 'Nation': 'Nation', 'Pos': 'Position', 'Squad': 'Squad',
    'Comp': 'Comptetition', 'Age': 'Age', 'Born': 'Born',
    'MP': 'Matches', 'Starts': 'Starts', 'Min': 'Minutes', '90s': '90s',
    'Gls': 'Goals', 'Ast': 'Assists', 'G+A': 'G+A', 'PK': 'PK_Made',
    'PKatt': 'PK_Attempted', 'CrdY': 'Yellow_Cards', 'CrdR': 'Red_Cards',
    'xG': 'xG', 'npxG': 'npxG', 'xAG': 'xAG', 'npxG+xAG': 'npxG+xAG',
    'PrgC': 'Prog_Carries', 'PrgP': 'Prog_Passes', 'PrgR': 'Prog_Passes_Received',
    'Dist': 'Avg_Shot_Dist', 'Sh': 'Shots', 'SoT': 'SoT', 'SoT%': 'SoT%',
    'G-xG': 'G-xG', 'np:G-xG': 'npG-xG', 'G/Sh': 'G/Shots', 'G/SoT': 'G/SoT',
    'npxG/Sh': 'npxG/Shots', 'Att': 'Passes_Attempted', 'Cmp': 'Passes Completed',
    'Cmp%': 'Passes%', 'KP': 'Key_Passes', '1/3': 'Passes_Final_Third',
    'PPA': 'Passes_Pen_Area', 'CrsPA': 'Crosses_Pen_Area', 'PrgDist': 'Tot_Dist_Prg_Passes',
    'TotDist': 'Tot_Dist_Passes', 'Crs': 'Crosses', 'Sw': 'Switches',
    'TB': 'Through_Balls', 'CK': 'Corner_Kicks', 'TI': 'Throw_Ins',
    'Off': 'Offsides', 'Blocks': 'Blocks', 'SCA': 'SCA', 'GCA': 'GCA',
    'Tkl': 'Tackles', 'TklW': 'Tackles_Won', 'Def 3rd': 'Tackles_Def_3rd',
    'Mid 3rd': 'Tackles_Mid_3rd', 'Att 3rd': 'Tackles_Att_3rd', 'Tkl%': 'Tackles%',
    'Lost': 'Challenges_Lost', 'Int': 'Interceptions', 'Tkl+Int': 'Tkl+Int',
    'Clr': 'Clearances', 'Err': 'Errors_Leading_Shot', 'Touches': 'Touches',
    'Def Pen': 'Touches_Def_Pen', 'Att Pen': 'Touches_Att_Pen', 'Succ': 'Take_On_Succ',
    'Succ%': 'Take_On%', 'Tkld': 'Dispossessed', 'Mis': 'Miscontrols',
    'Dis': 'Dispossessed', 'Rec': 'Recoveries', 'Carries': 'Carries',
    'CPA': 'Carries_Pen_Area', 'Fls': 'Fouls', 'Fld': 'Fouled', 'Won': 'Aerial_Won',
    'Won%': 'Aerial_Won%', '2CrdY': '2nd_Yellow', 'OG': 'Own Goal',
    'PKcon': 'PK_conceded', 'PKwon': 'PK_won',
}

# Columns that will be normalized to "per 90 minutes"
COLS_TO_DIVIDE_PER_90 = {
    'Goals', 'Assists', 'G+A', 'npG', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'xG+xAG', 
    'Shots', 'SoT', 'Offsides', 'Fouls', 'Fouled', 'Clearances', 'Interceptions', 
    'Tackles', 'Tkl+Int', 'Tackles_Won', 'Dispossessed', 'Miscontrols', 'Blocks', 
    'Errors_Leading_Shot', 'Recoveries', 'Aerial_Won', 'Prog_Carries', 'Prog_Passes',
    'Prog_Passes_Received', 'Passes_Attempted', 'Take_On_Attempted', 'Passes Completed', 
    'Key_Passes', 'Crosses', 'Switches', 'Through_Balls', 'GCA', 'SCA', 
    'Take_On_Succ', 'Touches', 'Carries', 'Passes_Received'
}

# Final desired order for the output CSV
FINAL_COLUMNS_ORDER = [
    'Name', 'Age', 'Nation', 'Position', 'Squad', 'Comptetition',
    'Matches', 'Starts', 'Minutes', '90s',
    'Goals', 'Assists', 'G+A', 'PK_Made', 'PK_Attempted', 'Yellow_Cards', 'Red_Cards',
    'xG', 'npxG', 'xAG', 'npxG+xAG', 'Prog_Carries', 'Prog_Passes', 'Prog_Passes_Received',
    'xG+xAG', 'Avg_Shot_Dist', 'G-xG', 'G/Shots', 'G/SoT', 'Shots', 'SoT', 'SoT%', 'npG-xG',
    'npxG/Shots', 'Passes_Final_Third', 'A-xAG', 'Passes_Attempted',
    'Take_On_Attempted', 'Passes Completed', 'Passes%', 'Crosses_Pen_Area',
    'Key_Passes', 'Passes_Pen_Area', 'Tot_Dist_Prg_Passes',
    'Tot_Dist_Passes', 'Blocks', 'Corner_Kicks', 'Crosses', 'Offsides',
    'Switches', 'Through_Balls', 'Throw_Ins', 'Fouled', 'GCA', 'SCA',
    'Tackles_Att_3rd', 'Clearances', 'Tackles_Def_3rd',
    'Errors_Leading_Shot', 'Interceptions', 'Challenges_Lost',
    'Tackles_Mid_3rd', 'Tackles', 'Tackles%', 'Tkl+Int', 'Tackles_Won',
    'Touches_Att_Pen', 'Carries_Pen_Area', 'Carries', 'Touches_Def_Pen',
    'Dispossessed', 'Miscontrols', 'Passes_Received', 'Take_On_Succ',
    'Take_On%', 'Touches', '2nd_Yellow', 'Fouls', 'Own Goal', 'PK_conceded',
    'PK_won', 'Recoveries', 'Aerial_Won', 'Aerial_Won%', 'npG', 'Goals_p90', 'Assists_p90',
    'G+A_p90', 'npG_p90', 'xG_p90', 'npxG_p90', 'xAG_p90', 'npxG+xAG_p90',
    'xG+xAG_p90', 'Shots_p90', 'SoT_p90', 'Offsides_p90', 'Fouls_p90', 'Fouled_p90', 'Clearances_p90', 
    'Interceptions_p90', 'Tackles_p90', 'Tkl+Int_p90', 'Tackles_Won_p90',
    'Dispossessed_p90', 'Miscontrols_p90', 'Blocks_p90',
    'Errors_Leading_Shot_p90', 'Recoveries_p90', 'Aerial_Won_p90',
    'Prog_Carries_p90', 'Prog_Passes_p90', 'Prog_Passes_Received_p90',
    'Passes_Attempted_p90', 'Take_On_Attempted_p90', 'Passes Completed_p90',
    'Key_Passes_p90', 'Crosses_p90', 'Switches_p90', 'Through_Balls_p90',
    'GCA_p90', 'SCA_p90', 'Take_On_Succ_p90', 'Touches_p90', 'Carries_p90',
    'Passes_Received_p90'
]

# =============================================================================
# SCRAPER CLASS
# =============================================================================

class FBrefScraper:
    """Handles Selenium WebDriver initialization and table extraction."""
    
    def __init__(self):
        self.driver = self._get_driver()

    def _get_driver(self) -> webdriver.Chrome:
        """Initialize a headless Chrome driver with specific options to avoid detection."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3") 
        # User-agent helps prevent blocking by the website
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def scrape_table(self, url: str, table_id: str) -> pd.DataFrame:
        """
        Navigates to the URL and extracts the specified HTML table.
        FBref sometimes hides tables inside HTML comments to speed up page load; 
        this method includes a fallback to parse those comments.
        """
        print(f"Scraping {table_id}...")
        try:
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 5)
            # Wait until the table container is present
            wait.until(EC.presence_of_element_located((By.ID, f'div_{table_id}')))
            
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            table_container = soup.find('div', {'id': f'div_{table_id}'})
            if not table_container:
                return pd.DataFrame()

            table_html = table_container.find('table')
            if not table_html:
                # FALLBACK: FBref logic often comments out secondary tables. 
                # We search for strings containing '<table' inside comments.
                comment = table_container.find(string=lambda text: isinstance(text, str) and '<table' in text)
                if comment:
                    df = pd.read_html(io.StringIO(str(comment)))[0]
                else:
                    return pd.DataFrame()
            else:
                df = pd.read_html(io.StringIO(str(table_container)))[0]

            # Flatten MultiIndex columns (FBref uses nested headers like 'Performance' -> 'Goals')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[1] if c[1] and "Unnamed" not in c[1] else c[0] for c in df.columns]

            # Remove recurring header rows (FBref repeats headers every 20-25 rows)
            df = df[df['Rk'] != 'Rk'].reset_index(drop=True)
            return df

        except Exception as e:
            print(f"  Error scraping {url}: {e}")
            return pd.DataFrame()
        
    def close(self):
        """Shutdown the browser instance."""
        self.driver.quit()

# =============================================================================
# DATA CLEANING & MERGING LOGIC
# =============================================================================

def clean_and_merge_data(scraper: FBrefScraper) -> pd.DataFrame:
    """
    Scrapes all required tables and merges them on a unique set of keys.
    """
    # Start with the 'Standard' stats table as the base
    df_std = scraper.scrape_table(URLS['standard'], TABLE_IDS['standard'])
    if df_std.empty:
        raise ValueError("Standard table not found. Scraper might be blocked or URL changed.")
    
    # Define keys to identify a player across different tables uniquely
    merge_keys = ['Player', 'Nation', 'Born', 'Squad']
    for key in merge_keys:
        if key in df_std.columns:
            df_std[key] = df_std[key].astype(str).str.strip()

    merged_df = df_std
    other_tables = ['shooting', 'passing', 'passing_types', 'defense', 'possession', 'gca', 'misc']
    
    for tbl_key in other_tables:
        df_new = scraper.scrape_table(URLS[tbl_key], TABLE_IDS[tbl_key])
        if not df_new.empty:
            for key in merge_keys:
                if key in df_new.columns:
                    df_new[key] = df_new[key].astype(str).str.strip()

            # Identify new columns to avoid duplicating existing ones (like Squad, Nation, etc.)
            new_cols = list(set(df_new.columns) - set(merged_df.columns))
            cols_to_use = new_cols + [k for k in merge_keys if k in df_new.columns]
            
            try:
                merged_df = pd.merge(merged_df, df_new[cols_to_use], on=merge_keys, how='left')
            except Exception:
                pass
        
        # Polite delay to avoid rate limiting
        time.sleep(1)

    return merged_df.copy()

def process_cleaned_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering, data type conversion, and duplicate aggregation.
    """
    df = df.copy()

    # 1. Filter out Goalkeepers (their stats are typically managed separately)
    if 'Position' in df.columns:
        df = df[df['Position'] != 'GK']
    elif 'Pos' in df.columns:
        df = df[df['Pos'] != 'GK']
    
    # 2. Rename columns to descriptive names
    df = df.rename(columns=COLUMN_MAPPING)
    df = df.loc[:, ~df.columns.duplicated()] # Remove accidental duplicate columns

    # 3. Numeric Conversion
    # FBref Age is usually '25-123' (years-days); we only keep the years.
    cols_str = {'Name', 'Nation', 'Position', 'Squad', 'Comptetition', 'Born'}
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(str).str.split('-').str[0]

    cols_to_numeric = [c for c in df.columns if c not in cols_str]
    df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 4. Preliminary Calculations (Absolute values)
    pre_calcs = {}
    if 'Goals' in df.columns and 'PK_Made' in df.columns:
        pre_calcs['npG'] = df['Goals'] - df['PK_Made'] # Non-penalty Goals = Goals - Penalty Kiks Made
    
    if 'xG' in df.columns and 'xAG' in df.columns:
        pre_calcs['xG+xAG'] = df['xG'] + df['xAG'] # Expected Goals and Assists
        
    if 'Assists' in df.columns and 'xAG' in df.columns:
        pre_calcs['A-xAG'] = df['Assists'] - df['xAG'] # Assist performance = Assists - Expected Assists
    
    if pre_calcs:
        df = df.drop(columns=pre_calcs.keys(), errors='ignore')
        df = pd.concat([df, pd.DataFrame(pre_calcs)], axis=1)
    
    # 5. Handle Duplicate Players (Transfers)
    # If a player moved from 'Team A' to 'Team B', they have two rows. 
    # We sum their stats and join the names of the squads.
    group_cols = ['Name', 'Born']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in group_cols]

    agg_rules = {col: 'sum' for col in numeric_cols}
    for col in df.columns:
        if col not in agg_rules and col not in group_cols:
            if col == 'Squad':
                # e.g., "Roma+Milan"
                agg_rules[col] = lambda x: "+".join(x.unique()) if len(x.unique()) > 1 else x.iloc[0]
            else:
                agg_rules[col] = 'first'
    
    df_agg = df.groupby(group_cols, as_index=False).agg(agg_rules)
    
    # 6. Recalculate Percentages
    # After summing stats for transferred players, percentage columns (like Shot %) 
    # become mathematically incorrect. We must recalculate them.
    new_rates = {}
    def calc_rate_safe(num_col, den_col):
        if num_col in df_agg.columns and den_col in df_agg.columns:
            num = df_agg[num_col].to_numpy(dtype=float)
            den = df_agg[den_col].to_numpy(dtype=float)
            return np.divide(num, den, out=np.zeros_like(num), where=den!=0) * 100
        return None

    sot_pct = calc_rate_safe('SoT', 'Shots')
    if sot_pct is not None: new_rates['SoT%'] = np.round(sot_pct, 1)

    pass_pct = calc_rate_safe('Passes Completed', 'Passes_Attempted')
    if pass_pct is not None: new_rates['Passes%'] = np.round(pass_pct, 1)

    tkl_pct = calc_rate_safe('Tackles_Won', 'Tackles')
    if tkl_pct is not None: new_rates['Tackles%'] = np.round(tkl_pct, 1)

    take_on_pct = calc_rate_safe('Take_On_Succ', 'Take_On_Attempted')
    if take_on_pct is not None: new_rates['Take_On%'] = np.round(take_on_pct, 1)
    
    if new_rates:
        df_agg = df_agg.drop(columns=new_rates.keys(), errors='ignore')
        df_agg = pd.concat([df_agg, pd.DataFrame(new_rates)], axis=1)

    # 7. Calculate "Per 90" metrics
    if '90s' in df_agg.columns:
        nineties = df_agg['90s'].to_numpy(dtype=float)
        p90_data = {}
        for col_base in COLS_TO_DIVIDE_PER_90:
            target_col_name = f"{col_base}_p90"
            if col_base in df_agg.columns:
                vals = df_agg[col_base].to_numpy(dtype=float)
                # Safely divide by 90s, handling zero minutes played
                p90_data[target_col_name] = np.divide(vals, nineties, out=np.zeros_like(vals), where=nineties!=0).round(2)
            else:
                p90_data[target_col_name] = 0.0
        
        df_final = pd.concat([df_agg, pd.DataFrame(p90_data)], axis=1)
    else:
        df_final = df_agg

    # 8. Final Formatting
    # Reindex ensures the output CSV has the columns in the exact order specified in configuration
    df_final = df_final.reindex(columns=FINAL_COLUMNS_ORDER, fill_value=0)
    
    return df_final

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()
    print("=== Starting FBref Scraping & Cleaning Process ===")
    
    scraper = FBrefScraper()
    
    try:
        print("\nStep 1: Downloading and Merging tables...")
        raw_df = clean_and_merge_data(scraper)
        print(f"Raw data downloaded: {raw_df.shape[0]} rows.")
        
        if raw_df.empty:
            print("Error: No data retrieved.")
            sys.exit(1)

        print("\nStep 2: Processing, Cleaning and Metric Calculation...")
        final_df = process_cleaned_dataframe(raw_df)
        
        # --- OUTPUT PATH MANAGEMENT ---
        # Gets the directory where the script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up a directory structure: /Season 2025-26/
        output_dir = os.path.join(current_dir, "Season 2025-26")
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Complete file path
        output_file = os.path.join(output_dir, "Player_Final.csv")
        # -----------------------------------------
        
        final_df.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"\n=== SUCCESS! File saved to: '{output_file}'")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Final dataset shape: {final_df.shape}")
        
        # Print a small preview for verification
        cols_check = ['Name', 'Squad', 'Matches', 'Minutes', 'Goals']
        print(f"Sample data:\n{final_df[cols_check].head(5)}")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always close the browser even if an error occurs
        scraper.close()