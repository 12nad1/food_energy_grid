import xarray as xr
import pandas as pd
import numpy as np
import json
import os
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import to_hex, LinearSegmentedColormap
from scipy import stats

import sesame as ssm

__all__ = [
    'add_livestock_data',
    'add_crop_production',
    'process_fbs_data',
    'remove_multi_country_regions',
    'adjust_production_based_on_trade',
    'add_population_data',
    'convert_production_to_energy',
    'get_livestock_df',
    'compare_aggregate_elements',
    'process_tabular_data',
    'add_crop_production_data',
    'add_fish_catch_data',
    'build_grid_dataset',
    'downscale_animal_counts',
    'downscale_metabolism',
    'downscale_fao',
    'compute_mss_grid',
    'downscale_grid_data',
    'hist_da',
    'plot_da',
    'create_diverging_cmap',
    'make_fbsv_for_Voronoi',
    'make_metabolism_production_maps',
    'make_netflow_maps',
    'food_supply_metabolism_regression',
    'create_figures',
    'food_ds_to_csv',
    'food_ds_to_ctry_table',
    'create_metabolism_dataframe',
    'main'
]

###########################################################################
###                        Tabular Data Processing                      ###
###########################################################################

def process_fbs_data(fbs1):
    """Process and aggregate food balance sheet data
    
    Pivots the data to have food balance elements as columns, aggregates by item and area,
    and ensures proper handling of numeric and categorical variables.
    """
    fbs=fbs1.pivot(index = ['Area', 'Item','Class','Type','kcal','Processed'], columns="Element", values="Y2015").reset_index()
    fbs=fbs[['Area', 'Item', 'Class', 'Type', 'kcal', 'Processed', 'Domestic supply quantity', 'Export Quantity', 'Losses',
    'Fat supply quantity (t)', 'Feed', 'Food supply (kcal)', 'Import Quantity', 'Food',
    'Other uses (non-food)', 'Processing', 'Production','Protein supply quantity (t)',
    'Residuals', 'Seed', 'Stock Variation','Tourist consumption','Food supply quantity (kg/capita/yr)']]
    fbs = fbs.groupby(["Item","Area"]).agg({
        col: ('sum' if col not in ["Item", "Class", "Type", "kcal", "Processed"] else 'first') for col in fbs.columns
    })
    fbs = fbs.rename_axis(None, axis=1).drop(columns=['Item','Area']).reset_index() # Reset Index so Item and Area are just columns
    return fbs

def remove_multi_country_regions(fbs_df):
    """Filter out multi-country regions from the dataset
    
    Removes aggregate regions and focuses on individual countries to avoid double-counting
    and ensure accurate spatial analysis.
    """
    fbs1 = fbs_df[~fbs_df['Class'].isin(['DROP', 'Aggregate'])]

    multi_ctry_regions = ['Africa', 'Americas', 'Asia', 'Australia and New Zealand', 'China',
    'Caribbean', 'Central America', 'Central Asia', 'Eastern Africa',
    'Eastern Asia', 'Eastern Europe', 'Europe', 'European Union (27)',
    'Land Locked Developing Countries', 'Least Developed Countries',
    'Low Income Food Deficit Countries', 'Melanesia', 'Micronesia', 'Middle Africa',
    'Net Food Importing Developing Countries', 'Netherlands Antilles (former)',
    'Northern Africa', 'Northern America', 'Northern Europe', 'Oceania',
    'Polynesia', 'Small Island Developing States', 'South America',
    'South-eastern Asia', 'Southern Africa', 'Southern Asia', 'Southern Europe',
    'Western Africa', 'Western Asia', 'Western Europe', 'World']
    
    fbs1 = fbs1[~fbs1['Area'].isin(multi_ctry_regions)]
    return fbs1

def old_adjust_production_based_on_trade(fbs):
    """Adjust production values to account for global trade imbalances
    
    Calculates adjustment factors based on the difference between imports and exports
    to ensure global mass balance in the food system analysis.
    """
    # Calculate global trade imbalance
    diff = fbs['Import Quantity'].sum() - fbs['Export Quantity'].sum()
    tot = (fbs['Import Quantity'].sum() + fbs['Export Quantity'].sum()) / 2
    
    # Calculate adjustment factors by item
    grouped = fbs.groupby('Item').agg({'Production': 'sum', 'Import Quantity': 'sum', 'Export Quantity': 'sum'})
    grouped['import-export-factor'] = (grouped['Production'] + grouped['Import Quantity'] - grouped['Export Quantity']) / grouped['Production']
    
    # Apply adjustment factors
    df = pd.merge(fbs, grouped['import-export-factor'], how='left', on='Item')
    fbs['Production_adj'] = df['Production'] * df['import-export-factor']
    
    return fbs

def adjust_production_based_on_trade(fbs, verbose=False):
    """Adjust production values to account for global trade imbalances using quadratic optimization
    
    Calculates adjustment factors based on the difference between imports and exports
    to ensure global mass balance in the food system analysis.
    """
    import cvxpy as cp

    # Group by item
    for item, group in fbs.groupby("Item"):
        idx = group.index
        n = len(group)

        # Extract variables
        P_orig = group["Production"].values
        I_orig = group["Import Quantity"].values
        E_orig = group["Export Quantity"].values
        U_orig = group["Domestic supply quantity"].values

        # Define cvxpy variables
        P = cp.Variable(n)
        E = cp.Variable(n)
        U = cp.Variable(n)
        I = cp.Variable(n)

        epsilon = 1e-6

        objective = cp.Minimize(
            (
                cp.sum_squares((P - P_orig) / (P_orig + epsilon)) +
                cp.sum_squares((E - E_orig) / (E_orig + epsilon)) +
                cp.sum_squares((U - U_orig) / (U_orig + epsilon)) + 
                2*cp.sum_squares((I - I_orig) / (I_orig + epsilon))
            )
        )

        # Constraints
        constraints = [
            P + I == E + U,            # local mass balance
            cp.sum(P) == cp.sum(U),    # global P = global U
            cp.sum(E) == cp.sum(I),    # global E = global I
            P >= 0, E >= 0, U >= 0, I >= 0  # non-negativity
        ]

        # Ensure that the adjusted values are within a factor of the original values
        # This is infeasible so we don't use it
        # factor = 10
        # constraints += [
        # P >= 1/factor * P_orig,
        # P <= factor * P_orig,
        # E >= 1/factor * E_orig,
        # E <= factor * E_orig,
        # U >= 1/factor * U_orig,
        # U <= factor * U_orig
        # # I >= 1/factor * I_orig,
        # # I <= factor * I_orig
        # ]

        # Solve
        problem = cp.Problem(objective, constraints)

        print(f"Solving for item '{item}'", end=" ")
        try:
            problem.solve(solver=cp.OSQP, max_iter=int(1e7), verbose=False)  # or OSQP/ECOS/SCS
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            continue

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"⚠️ Optimization failed for item '{item}': {problem.status}")
        else:
            print(f"✅ Optimization succeeded")

        # Save adjusted results
        fbs.loc[idx, 'Production_adj'] = P.value
        fbs.loc[idx, 'Export Quantity_adj'] = E.value
        fbs.loc[idx, 'Domestic supply quantity_adj'] = U.value
        fbs.loc[idx, 'Import Quantity_adj'] = I.value

    print("✅ Optimization completed for all items.")
    fbs.to_csv('balanced_fbs.csv')

    if verbose:
        fbs['delta_production'] = (fbs['Production_adj'] - fbs['Production'])
        fbs['delta_exports'] = (fbs['Export Quantity_adj'] - fbs['Export Quantity'])
        fbs['delta_domestic'] = (fbs['Domestic supply quantity_adj'] - fbs['Domestic supply quantity'])
        fbs['delta_imports'] = (fbs['Import Quantity_adj'] - fbs['Import Quantity'])

        # fbs.replace([np.inf, -np.inf], np.nan).sort_values('production_percent_change', ascending=False)
        fbs['production_percent_change'] = (100 * fbs['delta_production'] / fbs['Production']).replace([np.inf, -np.inf], np.nan)
        fbs['domestic_percent_change'] = (100 * fbs['delta_domestic'] / fbs['Domestic supply quantity']).replace([np.inf, -np.inf], np.nan)
        fbs['imports_percent_change'] = (100 * fbs['delta_imports'] / fbs['Import Quantity']).replace([np.inf, -np.inf], np.nan)
        fbs['exports_percent_change'] = (100 * fbs['delta_exports'] / fbs['Export Quantity']).replace([np.inf, -np.inf], np.nan)

        print("Global percent change")
        for delta, col in [('delta_exports', 'Export Quantity'), ('delta_domestic', 'Domestic supply quantity'), ('delta_production', 'Production'), ('delta_imports', 'Import Quantity')]:
            print(f"{delta}: {fbs[delta].sum() / fbs[col].sum() * 100:.2f}%")

        print("Maximum percent change")
        for percent_change in ['production_percent_change', 'domestic_percent_change', 'imports_percent_change', 'exports_percent_change']:
            print(f"{percent_change}: {fbs.set_index(['Area','Item'])[percent_change].idxmax()}, {fbs.set_index(['Area','Item'])[percent_change].max():.2f}%")

    for food in fbs['Item'].unique():
        assert(abs(fbs[fbs['Item'] == food]['Production_adj'].sum() - fbs[fbs['Item'] == food]['Domestic supply quantity_adj'].sum()) < 1)
        assert(abs(fbs[fbs['Item'] == food]['Export Quantity_adj'].sum() - fbs[fbs['Item'] == food]['Import Quantity_adj'].sum()) < 1)
        for area in fbs['Area'].unique():
            P = fbs[fbs['Area'] == area][fbs['Item'] == food]['Production_adj'].sum()
            U = fbs[fbs['Area'] == area][fbs['Item'] == food]['Domestic supply quantity_adj'].sum()
            E = fbs[fbs['Area'] == area][fbs['Item'] == food]['Export Quantity_adj'].sum()
            I = fbs[fbs['Area'] == area][fbs['Item'] == food]['Import Quantity_adj'].sum()
            assert(abs(P + I - E - U) < 1)

    return fbs

def add_population_data(fbs, fbs_df):
    """Add population data to the food balance sheet
    
    Extracts population information from the FAO dataset and merges it with
    the food balance sheet for per-capita calculations.
    """
    fbs_pop = fbs_df[fbs_df['Element'] == 'Total Population - Both sexes'].copy()
    fbs_pop['pop2015'] = fbs_pop['Y2015'] * 1000
    fbs_pop['pop2021'] = fbs_pop['Y2021'] * 1000
    fbs = fbs.merge(fbs_pop[['Area', 'pop2015', 'pop2021']], on='Area', how='left')
    return fbs

def convert_production_to_energy(fbs):
    """Convert food quantities to energy units (Mcal)
    
    Calculates energy content for all food items and separates production
    by type (crop, animal, processed) for energy flow analysis.
    """
    # Calculate kcal per kg for each item
    fbs['kcalpkg'] = 10 * fbs['kcal']
    fbs['kcalpkg'] = fbs['kcalpkg'].fillna(fbs['Food supply (kcal)'] / fbs['Domestic supply quantity']).replace([np.inf, -np.inf], np.nan)
    
    # Separate production by type
    fbs['CropProdMCal'] = fbs.apply(lambda r: r['Production_adj']*r['kcalpkg'] if r['Class'] == 'Crop' and r['Processed'] != 'Yes' else 0, axis=1)
    fbs['AnimProdMCal'] = fbs.apply(lambda r: r['Production_adj']*r['kcalpkg'] if r['Class'] in ['Livestock', 'Seafood'] and r['Processed'] != 'Yes' else 0, axis=1)
    fbs['ProcProdMCal'] = fbs.apply(lambda r: r['Production_adj']*r['kcalpkg'] if r['Processed'] == 'Yes' else 0, axis=1)
    
    # Also for masses (tons)
    fbs['CropProd'] = fbs.apply(lambda r: r['Production_adj'] if r['Class'] == 'Crop' and r['Processed'] != 'Yes' else 0, axis=1)
    fbs['AnimProd'] = fbs.apply(lambda r: r['Production_adj'] if r['Class'] in ['Livestock', 'Seafood'] and r['Processed'] != 'Yes' else 0, axis=1)
    fbs['ProcProd'] = fbs.apply(lambda r: r['Production_adj'] if r['Processed'] == 'Yes' else 0, axis=1)

    # Convert utilization columns to Energy Units
    for fbs_col in ['Feed', 'Food', 'Processing', 'Losses', 'Residuals', 'Seed', 'Stock Variation']:
        print('rescaling', fbs_col)
        new_col = fbs_col[:4] + 'MCal'
        fbs[new_col] = fbs.apply(lambda r: r[fbs_col]*r['kcalpkg'], axis=1)
        # Check for division by zero and handle edge cases
        mask = (fbs['Domestic supply quantity'] != 0) & (fbs['Domestic supply quantity_adj'] != 0)
        fbs.loc[mask, new_col] = fbs.loc[mask, new_col] * fbs.loc[mask, 'Domestic supply quantity_adj'] / fbs.loc[mask, 'Domestic supply quantity']
        
        # If original is 0 and adjusted is within tolerance, keep as 0
        tolerance = 1e-6
        zero_original_mask = (fbs['Domestic supply quantity'] == 0) & (fbs['Domestic supply quantity_adj'].abs() <= tolerance)
        fbs.loc[zero_original_mask, new_col] = 0
        
        # If adjusted is 0 and original is within tolerance, keep as 0  
        zero_adjusted_mask = (fbs['Domestic supply quantity_adj'] == 0) & (fbs['Domestic supply quantity'].abs() <= tolerance)
        fbs.loc[zero_adjusted_mask, new_col] = 0
        
        # Raise error for other cases where one is 0 and the other is not
        problematic_mask = ((fbs['Domestic supply quantity'] == 0) & (fbs['Domestic supply quantity_adj'].abs() > tolerance)) | \
                          ((fbs['Domestic supply quantity_adj'] == 0) & (fbs['Domestic supply quantity'].abs() > tolerance))
        if problematic_mask.any():
            raise ValueError(f"Division by zero or infinite value would result. Check rows where Domestic supply quantity or Domestic supply quantity_adj are zero but the other is not.")
    
    # Convert Production, Imports, and Exports to Energy Units
    for fbs_col in ['Production_adj', 'Export Quantity_adj', 'Domestic supply quantity_adj', 'Import Quantity_adj']:
        new_col = fbs_col[:4] + 'MCal'
        fbs[new_col] = fbs.apply(lambda r: r[fbs_col]*r['kcalpkg'], axis=1)

    # Calculate other useful MCal columns
    fbs['FoodSupplyMCal'] = fbs['Food supply (kcal)']
    fbs['FoodMCal'] = fbs['FoodMCal'] * fbs['Domestic supply quantity_adj'] / fbs['Domestic supply quantity']
    fbs['ProdMCal'] = fbs['CropProdMCal'] + fbs['AnimProdMCal'] + fbs['ProcProdMCal']
    fbs['OtherMCal'] = fbs['ProdMCal'] - fbs['FeedMCal'] - fbs['ProcMCal'] - fbs['FoodSupplyMCal']
    fbs['ConsMCal'] = fbs['FoodSupplyMCal'] + fbs['FeedMCal']

    # Get mass
    fbs['FoodSupplyT'] = fbs.apply(lambda r: r['FoodSupplyMCal']/r['kcalpkg'] if r['kcalpkg'] != 0 else np.nan, axis=1) # divide by ratio to get Tons
    return fbs

def get_livestock_df(config):
    with open(config['fao_ctry_to_regions.json']) as json_file:
        fao_country_to_region = json.load(json_file)
    with open(config['Names']) as json_file:
        Names = json.load(json_file)
    
    # Source: https://www.fao.org/3/i2294e/i2294e00.pdf
    lsu_data = {
        'Region': ['Near East North Africa', 'North America', 'Africa South of Sahara', 'Central America', 'South America', 'South Africa', 'OeCD', 'East and South East Asia', 'South Asia', 'Transition Markets', 'Caribbean', 'Near East', 'Other'],
        'Cattle': [0.70, 1.00, 0.50, 0.70, 0.70, 0.70, 0.90, 0.65, 0.50, 0.60, 0.60, 0.55, 0.60],
        'Buffalo': [0.70, np.nan, np.nan, np.nan, np.nan, np.nan, 0.70, 0.70, 0.50, 0.70, 0.60, 0.60, 0.60],
        'Sheep': [0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        'Goats': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        'Swine / pigs': [0.20, 0.25, 0.20, 0.25, 0.25, 0.20, 0.25, 0.25, 0.20, 0.25, 0.20, 0.25, 0.20],
        'Asses': [0.50, 0.50, 0.30, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        'Horses': [0.40, 0.80, 0.50, 0.50, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.56, 0.65],
        'Mules and hinnies': [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        'Camels': [0.75, np.nan, 0.70, np.nan, np.nan, np.nan, 0.90, 0.80, np.nan, np.nan, np.nan, 0.70, np.nan],
        'Chickens': [0.01, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    }

    lsu_df = pd.DataFrame(lsu_data)
    lsu_df.set_index('Region', inplace=True)

    # Source: FAOSTAT CropsAndLivestockProducution
    livestock_df = pd.read_csv(config['fao_livestock.csv'])
    livestock_df = livestock_df.replace({'Area': Names})
    livestock_df.rename(columns={'Area': 'ISO3'}, inplace=True)
    livestock_df = livestock_df[livestock_df['Item'].isin(livestock_df[livestock_df['Element'] != 'Production'].Item.unique())]

    # Use FAO region categoriations along with regional lsu coefficients to get lsu values for each country
    livestock_df['FAO_Region'] = livestock_df['ISO3'].map(fao_country_to_region)
    livestock_df['LSU_coeff'] = np.nan
    valid_rows = livestock_df['Item'].isin(lsu_data.keys())

    def get_lsu_value(row):
        try:
            val = lsu_df.at[row['FAO_Region'], row['Item']]
            if np.isnan(val):
                return lsu_df[row['Item']].mean()
            else:
                return val
        except KeyError:
            return np.nan 
        
    # Format the livestock counts dataframe to have consistent units and be ready for dasymetric mapping
    livestock_df = livestock_df[livestock_df['Item'].isin(lsu_data.keys())]
    livestock_df['LSU_coeff'] = livestock_df.apply(get_lsu_value, axis=1)
    livestock_df = livestock_df[livestock_df['Item'].isin(livestock_df[livestock_df['Element']=='Stocks'].Item.unique())]
    livestock_df.loc[livestock_df['Unit'] == '1000 An', 'Value'] *= 1000
    livestock_df.loc[livestock_df['Unit'] == '1000 An', 'Unit'] = 'An'
    livestock_df['LSU'] = livestock_df['LSU_coeff'] *  livestock_df['Value']
    return livestock_df

def compare_aggregate_elements(fbs):
    df = fbs.groupby('Item').sum()[[c for c in fbs.columns if c.endswith('MCal')]].sum() * 1e6 / 7.5e9 / 365
    prod = df['CropProdMCal'] + df['AnimProdMCal'] + df['ProcProdMCal']
    cons = df['FeedMCal'] + df['ProcMCal'] + df['OtherMCal']
    print('diff:',prod - cons)
    print('food supply:', df['FoodSupplyMCal'])
    print('prod', prod)
    print('cons', cons)
    print(df)

def get_available_production(fbs):
    """Get available production for each country and food item
    
    Calculates available production for each country and food item based on
    domestic supply, imports, and exports.
    """
    # Load Data
    F_if = fbs.pivot(index='Area', columns='Item', values='FoodMCal').fillna(0)
    P_if = fbs.pivot(index='Area', columns='Item', values='ProdMCal').fillna(0)
    I_if = fbs.pivot(index='Area', columns='Item', values='ImpoMCal').fillna(0)
    E_if = fbs.pivot(index='Area', columns='Item', values='ExpoMCal').fillna(0)
    U_if = fbs.pivot(index='Area', columns='Item', values='DomeMCal').fillna(0)

    # Initialize result table
    AP_if = pd.DataFrame(0.0, index=F_if.index, columns=F_if.columns)

    # Compute LocalFrac = F / U
    LocalFrac = F_if / U_if
    LocalFrac = LocalFrac.replace([np.inf, -np.inf], 0).fillna(0)

    # Compute Available Food
    for i in F_if.index:
        # Local share of domestic availability
        local_prod_thats_food = (P_if.loc[i] - E_if.loc[i]) * LocalFrac.loc[i]

        # Import share: sum over j ≠ i of LocalFrac_j * E_j * share_ij
        export_prod_thats_food = pd.Series(0.0, index=F_if.columns)

        for j in F_if.index:
            if j == i:
                continue
            # Fraction of j's exports that go to i, proportional to i's imports
            share_ij = (I_if.loc[j] / I_if.sum(axis=0)).replace([np.inf, -np.inf], 0).fillna(0)
            export_prod_thats_food += E_if.loc[j] * LocalFrac.loc[j] * share_ij

        # Total available food in i for each commodity
        AP_if.loc[i] = local_prod_thats_food + export_prod_thats_food

    # If the fractions exceed production, we need to clip the result to the production
    AP_if = pd.DataFrame(np.minimum(AP_if, P_if), index=AP_if.index, columns=AP_if.columns)
    AP_long = AP_if.stack().reset_index()
    AP_long.columns = ['Area', 'Item', 'AvailableProductionMCal']
    # Also clip negative values to 0 (mostly from floating point error but dividing by small values and multiplying by calories makes it significant)
    AP_long.loc[AP_long['AvailableProductionMCal'] < 0, 'AvailableProductionMCal'] = 0

    fbs = fbs.merge(AP_long, on=['Area', 'Item'], how='left')

    return fbs

def process_tabular_data(config, verbose=False):
    """Process FAO Food Balance Sheet data into analysis-ready format
    
    Loads, cleans, and transforms the raw FAO data into a structured dataset
    with energy units and population-normalized metrics.
    """
    if verbose:
        print("Loading and processing FBS data...")
    
    fbs_path = config['fbs']
    fbs_labels_path = config['fbs_labels']
    names_path = config['Names']

    # Load and process FBS data
    with open(names_path, 'r') as f:
        Names = json.load(f)
        
    fbs_labels_df = pd.read_csv(fbs_labels_path)
    fbs_labels_df = fbs_labels_df[[col for col in fbs_labels_df.columns if col != 'Item']]
    
    fbs_df = pd.read_csv(fbs_path, encoding='ISO-8859-1')
    fbs_df = pd.merge(fbs_df, fbs_labels_df, on='Item Code', how='left')
    
    # Remove regions containing multiple countries
    if verbose:
        print("Removing multi-country regions...")
    fbs = remove_multi_country_regions(fbs_df)
    
    # Process and pivot data
    if verbose:
        print("Processing and pivoting data...")
    fbs = process_fbs_data(fbs)
    
    # Adjust production based on trade imbalance
    if verbose:
        print("Adjusting production based on trade imbalance...")
    fbs = adjust_production_based_on_trade(fbs)
    
    # Add population data
    if verbose:
        print("Adding population data...")
    fbs = add_population_data(fbs, fbs_df)
    
    # Convert production to energy units
    if verbose:
        print("Converting production to energy units and calculating available production...")
    fbs = convert_production_to_energy(fbs)

    # Get available production
    if verbose:
        print("Calculating available production...")
    fbs = get_available_production(fbs)

    if verbose:
        print("Comparing aggregate elements...")
        compare_aggregate_elements(fbs)

    # Sum all the foods over each country and food item
    cols = ['Item', 'Area', 'Processed', 'Domestic supply quantity', 'Export Quantity', 'Losses',
        'Fat supply quantity (t)', 'Feed', 'Import Quantity', 'Food', 'Other uses (non-food)',
        'Processing', 'Production', 'Protein supply quantity (t)', 'Residuals', 'Seed',
        'Stock Variation', 'Tourist consumption', 'Production_adj', 'CropProdMCal', 'AnimProdMCal',
        'ProcProdMCal', 'CropProd', 'AnimProd', 'ProcProd', 'FeedMCal', 'FoodMCal', 'ProcMCal',
        'LossMCal', 'ResiMCal', 'SeedMCal', 'StocMCal', 'ImpoMCal', 'ExpoMCal',
        'FoodSupplyMCal', 'FoodSupplyT', 'OtherMCal', 'ConsMCal', 'ProdMCal', 'AvailableProductionMCal']
    
    fbscatdf = fbs[cols].groupby(['Area', 'Item']).sum().reset_index()
    fbscatdf = fbscatdf.pivot(index='Area', columns=['Item'], values=list(fbscatdf.columns)[2:])
    fbscatdf.columns = ['{} {}'.format(col[0], col[1]) for col in fbscatdf.columns]
    fbscatdf = fbscatdf.reset_index()
    fbscatdf.rename(columns={'Area':'ISO3'}, inplace=True)
    fbscatdf.replace({'ISO3': Names}, inplace=True)

    livestock_df = get_livestock_df(config)

    return fbs, fbscatdf, livestock_df

###########################################################################
###                        Metabolism Processing                        ###
###########################################################################

### Helper functions

def convert_to_iso3(df, column_name, country_names):
    # Convert country names to ISO3 codes
    df['iso3'] = df[column_name].map(country_names)
    # Drop rows where iso3 is NaN
    df = df.dropna(subset=['iso3'])
    return df

### Main functions

def load_height_data(height_csv_path):
    """Load and process height data from NCD RisC dataset.
    
    Args:
        height_csv_path (str): Path to the height data CSV file
        
    Returns:
        pd.DataFrame: Processed height data with extended age groups
    """
    hdf = pd.read_csv(height_csv_path)
    hdf.set_index(['Country', 'Sex', 'Year', 'Age group'], inplace=True)
    
    # Get unique combinations for age 19
    age_19 = hdf.loc[(slice(None), slice(None), slice(None), 19), :]
    unique_combinations = age_19.reset_index()[['Country', 'Sex', 'Year']].drop_duplicates()
    
    new_rows = []
    total = len(unique_combinations)
    
    for i, row in unique_combinations.iterrows():
        percent = (i+1)/total*100
        if percent % 10 == 0:
            print(f'{int(percent)}% finished with height data')

        country, sex, year = row['Country'], row['Sex'], row['Year']
        base_height = hdf.loc[(country, sex, year, 19), 'Mean height']
        base_sigma = hdf.loc[(country, sex, year, 19), 'Mean height standard error']
        relative_sigma = base_sigma / base_height
        
        total_loss = 3.81 if sex in ['Boys', 'Men'] else 6.35
        
        # Ages 20-49: no height loss
        for age in range(20, 50):
            new_rows.append({
                'Country': country,
                'Sex': sex,
                'Year': year,
                'Age group': age,
                'Mean height': base_height,
                'Mean height standard error': base_sigma
            })
        
        # Ages 50-100: linear height loss
        for age in range(50, 101):
            years_after_50 = age - 50
            height_loss = (years_after_50 / 50) * total_loss
            estimated_height = base_height - height_loss
            
            new_rows.append({
                'Country': country,
                'Sex': sex,
                'Year': year,
                'Age group': age,
                'Mean height': estimated_height,
                'Mean height standard error': relative_sigma * estimated_height
            })
    
    # Create extended dataframe
    hdf_extended = pd.concat([
        hdf.reset_index(),
        pd.DataFrame(new_rows)
    ], ignore_index=True)
    
    # Sort and convert types
    hdf_extended = hdf_extended.sort_values(['Country', 'Sex', 'Year', 'Age group'])
    hdf_extended['Year'] = hdf_extended['Year'].astype('int64')
    hdf_extended['Age group'] = hdf_extended['Age group'].astype('int64')
    
    return hdf_extended

def load_population_data(population_csv_path):
    """Load and process population data.
    
    Args:
        population_csv_path (str): Path to the population data CSV file
        
    Returns:
        pd.DataFrame: Processed population data
    """
    pdf = pd.read_csv(population_csv_path)
    pdf['total'] = sum(pdf[str(i)] for i in range(101))
    pdf['avg_age'] = pdf[[str(i) for i in range(101)]].multiply(
        pdf[[str(i) for i in range(101)]].columns.astype(int), axis=1
    ).sum(axis=1) / pdf['total']
    pdf['total_population'] = pdf['total']
    pdf.drop(columns=['total'], inplace=True)
    pdf['iso3'] = pdf['country']
    return pdf

def merge_height_population(hdf_extended, pdf, config):
    """Merge height and population data.
    
    Args:
        hdf_extended (pd.DataFrame): Extended height data
        melted_pdf (pd.DataFrame): Melted population data
        
    Returns:
        pd.DataFrame: Merged height and population data with weighted statistics
    """
    # Melt population data
    melted_pdf = pdf.melt(
        id_vars=['country', 'year'],
        value_vars=[str(i) for i in range(101)],
        var_name='Age group',
        value_name='population'
    )
    
    # Convert types
    melted_pdf['year'] = melted_pdf['year'].astype('int64')
    melted_pdf['Age group'] = melted_pdf['Age group'].astype('int64')
    melted_pdf['country'] = melted_pdf['country'].astype('str')

    with open(config['Names'], 'r') as f:
        country_names = json.load(f)

    hdf_extended['iso3'] = hdf_extended['Country'].map(country_names)
    melted_pdf['iso3'] = melted_pdf['country'].map(country_names)

    # Merge height data with population data
    merged = pd.merge(
        hdf_extended, melted_pdf,
        left_on=['iso3', 'Year', 'Age group'],
        right_on=['iso3', 'year', 'Age group'],
        how='inner'
    )
    
    # Calculate weighted statistics
    def weighted_mean_and_error(group):
        weighted_mean = (group['Mean height'] * group['population']).sum() / group['population'].sum()
        weighted_error = np.sqrt(np.sum(group['population'] / group['population'].sum() * 
                                      group['Mean height standard error']**2))
        
        return pd.Series({
            'avg_height': weighted_mean,
            'height_std': weighted_error,
            'population': group['population'].sum()
        })
    
    final_hdf_grouped = merged.groupby(['iso3', 'Sex', 'Year'], group_keys=False).apply(
        weighted_mean_and_error
    ).reset_index()

    pdf['iso3'] = pdf['country'].map(country_names)

    # Merge with average age data
    final_hdf_grouped = final_hdf_grouped.merge(pdf[['iso3', 'year', 'avg_age']], 
                                            left_on=['iso3', 'Year'], 
                                            right_on=['iso3', 'year'], 
                                            how='left')
    final_hdf_grouped.drop(columns=['Year'], inplace=True)
    
    return final_hdf_grouped

def load_bmi_data(bmi_csv_path, n_samples=30):
    """Load and process BMI data.
    
    Args:
        bmi_csv_path (str): Path to the BMI data CSV file
        n_samples (int): Number of samples for Monte Carlo simulation
        
    Returns:
        pd.DataFrame: Processed BMI data with statistics
    """
    bdf = pd.read_csv(bmi_csv_path)
    
    bmi_bins = [
        (17, 'Prevalence of BMI<18.5 kg/m² (underweight)'),
        (19.25, 'Prevalence of BMI 18.5 kg/m² to <20 kg/m²'),
        (22.5, 'Prevalence of BMI 20 kg/m² to <25 kg/m²'),
        (27.5, 'Prevalence of BMI 25 kg/m² to <30 kg/m²'),
        (32.5, 'Prevalence of BMI 30 kg/m² to <35 kg/m²'),
        (37.5, 'Prevalence of BMI 35 kg/m² to <40 kg/m²'),
        (42, 'Prevalence of BMI >=40 kg/m² (morbid obesity)')
    ]
    
    countries, years, genders, bmis, bmi_stds = [], [], [], [], []
    total = len(bdf)
    
    for i, row in bdf.iterrows():
        percent = (i+1)/total*100
        if percent % 10 == 0:
            print(f'{int(percent)}% finished with BMI data')
        
        avg_bmi_samples = []
        for _ in range(n_samples):
            prevalences = []
            for midpoint, col in bmi_bins:
                mean = row[col]
                lower = row[f'{col} lower 95% uncertainty interval']
                upper = row[f'{col} upper 95% uncertainty interval']
                std = (upper - lower) / (2 * 1.96)
                sample = np.clip(np.random.normal(mean, std), lower, upper)
                prevalences.append(sample)
            
            prevalences = np.array(prevalences)
            prevalences /= prevalences.sum()
            avg_bmi_samples.append(np.sum([midpoint * p for (midpoint, _), p in zip(bmi_bins, prevalences)]))
        
        avg_bmi_mean = np.mean(avg_bmi_samples)
        avg_bmi_std = np.std(avg_bmi_samples)
        
        countries.append(row['Country/Region/World'])
        years.append(row['Year'])
        genders.append(row['Sex'])
        bmis.append(avg_bmi_mean)
        bmi_stds.append(avg_bmi_std)
    
    return pd.DataFrame({
        'country': countries,
        'year': years,
        'gender': genders,
        'bmi': bmis,
        'bmi_std': bmi_stds
    })

def compute_mass_bmr(bmi_stats_df, avg_height_df):
    """Compute mass and BMR from height and BMI data.
    
    Args:
        bmi_stats_df (pd.DataFrame): BMI statistics
        avg_height_df (pd.DataFrame): Average height data
        
    Returns:
        pd.DataFrame: Data with computed mass and BMR
    """
    avg_height_df['Sex'] = avg_height_df['Sex'].replace({'Boys': 'Men', 'Girls': 'Women'})
    
    df = pd.merge(
        bmi_stats_df, 
        avg_height_df, 
        left_on=['year', 'gender', 'iso3'], 
        right_on=['year', 'Sex', 'iso3'], 
        how='inner', 
        suffixes=('', '_y')
    )
    
    df = df.drop(columns=['Sex', 'country'])
    
    # Compute mass and its standard deviation
    df['avg_mass'] = df['bmi'] * (df['avg_height'] / 100)**2
    df['mass_std'] = np.sqrt(
        (df['bmi_std'] / df['bmi'])**2 + 
        2*(df['height_std'] / df['avg_height'])**2
    ) * df['avg_mass']
    
    # Compute BMR using Mifflin St. Jeor Equation
    df['bmr'] = 10 * df['avg_mass'] + 6.25 * df['avg_height'] - 5 * df['avg_age'] + \
                np.where(df['gender'] == 'Men', 5, -161)
    df['bmr_std'] = np.sqrt((10 * df['mass_std'])**2 + (6.25 * df['height_std'])**2)
    
    # Average over gender
    df = df.groupby(['iso3', 'year']).agg({
        'avg_mass': 'mean',
        'avg_height': 'mean',
        'avg_age': 'mean',
        'bmi': 'mean',
        'bmr': 'mean',
        'mass_std': lambda x: np.sqrt(np.mean(x**2)),
        'height_std': lambda x: np.sqrt(np.mean(x**2)),
        'bmi_std': lambda x: np.sqrt(np.mean(x**2)),
        'bmr_std': lambda x: np.sqrt(np.mean(x**2)),
        'population': 'sum'
    }).reset_index()
    
    return df

def load_food_supply_data(config):
    """Load and process food supply data.
    
    Args:
        fbs_path (str): Path to the food balance sheets data
        fbs_labels_path (str): Path to the FBS labels
        old_fbs_path (str): Path to the old FAO food supply data
        
    Returns:
        pd.DataFrame: Processed food supply data
    """
    fbs_path = config['fbs_path']
    fbs_labels_path = config['fbs_labels_path']
    old_fbs_path = config['old_fbs_path']
    with open(config['Names'], 'r') as f:
        country_names = json.load(f)

    # Load and process main FBS data
    orig_df = pd.read_csv(fbs_path, encoding='ISO-8859-1')
    fbs_df = orig_df[orig_df['Element'].isin([
        'Total Population - Both sexes', 
        'Food supply (kcal)', 
        'Food supply (kcal/capita/day)'
    ])]
    
    fbs_df.loc[fbs_df['Element'] == 'Total Population - Both sexes', 'Element'] = 'Population'
    
    # Get year columns
    year_columns = [col for col in fbs_df.columns if col.startswith('Y') and 2010 <= int(col[1:]) <= 2021]
    
    # Melt and pivot
    df_melted = pd.melt(
        fbs_df,
        id_vars=['Area', 'Element', 'Item'],
        value_vars=year_columns,
        var_name='Year',
        value_name='Value'
    )
    
    df_melted['Year'] = df_melted['Year'].str[1:].astype(int)
    
    fbs_df = df_melted.pivot_table(
        index=['Area', 'Element', 'Item'],
        columns='Year',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    # Process labels and filter
    fbs_labels_df = pd.read_csv(fbs_labels_path)
    fbs_df = pd.merge(fbs_df, fbs_labels_df, on='Item', how='left')
    fbs_df = fbs_df[~fbs_df['Class'].isin(['DROP', 'Aggregate'])]
    
    # Process food supply data
    fbs_df = fbs_df[fbs_df['Element'].isin(['Food supply (kcal/capita/day)'])].groupby(
        ['Area','Element']
    ).sum().drop(columns='Item').reset_index()
    
    fbs_df = fbs_df.melt(
        id_vars=['Area', 'Element'], 
        value_vars=[i for i in range(2010, 2022)], 
        var_name='Year', 
        value_name='Value'
    )
    
    # Load and process old FBS data
    fbs_time_series = pd.read_csv(old_fbs_path).drop(columns=[
        'Domain', 'Domain Code', 'Area Code (M49)', 'Element Code',
        'Item Code (FBS)', 'Year Code', 'Unit', 'Flag', 'Flag Description'
    ])
    
    fbs_time_series = fbs_time_series[fbs_time_series['Year'] < 2010].drop(columns=['Item'])
    
    # Merge and pivot
    merged_fbs_df = pd.concat([fbs_time_series, fbs_df], axis=0)
    pivoted_df = merged_fbs_df.pivot(
        index=['Area', 'Year'], 
        columns='Element', 
        values='Value'
    ).reset_index()

    pivoted_df['iso3'] = pivoted_df['Area'].map(country_names)
    
    return pivoted_df.replace(0, np.nan)

def process_activity_data(config, mixed_bmr_data, pdf):
   # Load all required data files
    concordance = pd.read_csv(config['mets_moogal'])
    ghd = pd.read_csv(config['global_human_day'])
    ghd_all = pd.read_csv(config['ghd_all_countries'])
    regions = pd.read_csv(config['country_regions'])

    sourcedir = '~/Downloads/metabolism 3/'
    concordance = pd.read_csv(sourcedir + 'compendium_METS_MOOGAL_.csv')
    ghd = pd.read_csv(sourcedir + 'global_human_day.csv')
    ghd_all = pd.read_csv(sourcedir + 'all_countries.csv')
    mixed_bmr_data = pd.read_csv(sourcedir + 'bmi_and_food_supply_data.csv')


    concordance = concordance.query("ignore != 'x'").drop('ignore', axis=1)
    mixed_bmr_data = mixed_bmr_data.dropna()

    # # Calculate MET values
    # met = concordance['MET Value'].values.reshape(556,1) * concordance.iloc[:,4:].values
    # global_met = pd.DataFrame(pd.DataFrame(met).mean(axis=0), columns=['MET'])
    # global_met['Subcategory'] = concordance.columns[4:]
    # global_met = global_met.merge(ghd[['Subcategory','hoursPerDay']], on='Subcategory')

    # Calculate MET values
    met = concordance['MET Value'].values.reshape(556,1) * concordance.iloc[:,4:].values
    global_met = pd.DataFrame(pd.DataFrame(met).median(axis=0), columns=['MET'])
    global_met['Subcategory'] = concordance.columns[4:]
    global_met = global_met.merge(ghd[['Subcategory','hoursPerDay']], on='Subcategory')

    # Calculate weighted average daily MET
    mean_met = np.sum(global_met.MET * (global_met.hoursPerDay / 24))
    print('Mean MET:', round(mean_met,2))
    sigma_hrs = (ghd.hoursPerDay / 24) * (ghd.uncertainty)
    sigma_met = np.sqrt(np.sum(0**2 + sigma_hrs**2))
    print('Stdev MET:', round(sigma_met,2))

    # Process country-specific data
    time_countries = ghd_all[['countryISO3','Subcategory','hoursPerDayCombined','uncertaintyCombined']]
    time_countries = time_countries.merge(global_met, on='Subcategory', how='left')
    time_countries['weightedMET'] = (time_countries.hoursPerDayCombined / 24) * time_countries.MET
    time_countries['stdev'] = ((time_countries.hoursPerDayCombined / 24) * time_countries.uncertaintyCombined) ** 2
    mean_met_countries = time_countries.groupby('countryISO3').weightedMET.sum()
    std_met_countries = np.sqrt(time_countries.groupby('countryISO3').stdev.sum())

    # Merge with GDP data
    df = pd.DataFrame(mean_met_countries).reset_index()
    df = df.merge(pd.DataFrame(std_met_countries).reset_index(), on='countryISO3', how='left')
    df = df.rename(columns={'countryISO3': 'iso3'})
    mixed_bmr_data = mixed_bmr_data.merge(df, on='iso3', how='left')

    # Process mixed data
    mixed_bmr_data['TMR'] = mixed_bmr_data['weightedMET'] * mixed_bmr_data['bmr']
    mixed_bmr_data['sigmaTMR'] = np.sqrt((mixed_bmr_data.stdev / mixed_bmr_data.weightedMET)**2 + (mixed_bmr_data.bmr_std / mixed_bmr_data.bmr)**2)
    mixed_bmr_data = mixed_bmr_data.drop(['avg_mass','avg_height','avg_age','mass_std','height_std','bmi','bmi_std'], axis=1)

    # drop population because its wrong after pipeline
    metabolism_2015 = mixed_bmr_data.query("year == 2015").drop(columns=['population'])

    # Calculate global values for 1990-2019
    global_metabolism = []
    for name, grp in mixed_bmr_data.groupby('year'):
        global_pop = mixed_bmr_data.query("year == @name").population.sum()
        # Total Metabolism mean
        mean_metabolism = np.sum((grp.TMR * grp.population) / global_pop)
        # Total Metabolism uncertainty
        sigma_metabolism = np.sqrt(np.sum(grp.sigmaTMR**2 * (grp.population / global_pop)**2)) * mean_metabolism
        # BMR mean
        bmr = np.sum(grp.bmr * grp.population / global_pop)
        # BMR uncertainty
        sigma_bmr = np.sqrt(np.sum(grp.bmr_std**2 * (grp.population / global_pop)**2))
        global_metabolism.append((name, round(float(np.float64(mean_metabolism)),2), sigma_metabolism, bmr, sigma_bmr))

    global_metabolism = pd.DataFrame(global_metabolism, columns=['year','TMR','sigmaTMR','BMR','sigmaBMR'])

    # Calculate confidence intervals
    global_metabolism['ci95'] = 1.96 * global_metabolism.sigmaTMR

    # Calculate food supply and AMR
    mixed_bmr_data['totalCal'] = mixed_bmr_data['Food supply (kcal/capita/day)'] * mixed_bmr_data['population']
    global_cals = mixed_bmr_data.groupby('year')['totalCal'].sum() / mixed_bmr_data.groupby('year')['population'].sum()

    global_metabolism['AMR'] = global_metabolism.TMR - global_metabolism.BMR
    global_metabolism['foodSupply'] = global_cals.values

    # Adjustment to metabolism_2015 to make fully globally complete - as-is, the 74 missing countries represent about 0.8% of the 2015 population
    metabolism_2015 = regions[['country_iso3']].merge(metabolism_2015, left_on='country_iso3', right_on='iso3', how='left')
    metabolism_2015 = metabolism_2015.drop('iso3', axis=1).rename(columns={'bmr':'BMR', 
                                                    'bmr_std':'sigmaBMR', 
                                                    'Food supply (kcal/capita/day)':'foodSupply', 
                                                    'country_iso3':'iso3'})
    metabolism_2015['year'] = metabolism_2015.year.fillna(2015)
    metabolism_2015['BMR'] = metabolism_2015.BMR.fillna(metabolism_2015.BMR.mean())
    metabolism_2015['sigmaBMR'] = metabolism_2015.sigmaBMR.fillna(metabolism_2015.sigmaBMR.max())
    metabolism_2015['foodSupply'] = metabolism_2015.foodSupply.fillna(metabolism_2015.foodSupply.mean())
    metabolism_2015['weightedMET'] = metabolism_2015.weightedMET.fillna(1.85)
    metabolism_2015['TMR'] = metabolism_2015.TMR.fillna(metabolism_2015.TMR.mean())
    metabolism_2015['stdev'] = metabolism_2015.stdev.fillna(metabolism_2015.stdev.max())
    metabolism_2015['sigmaTMR'] = metabolism_2015.sigmaTMR.fillna(metabolism_2015.sigmaTMR.max())

    pop15_df = pdf[['iso3','year','total_population']].query('year == 2015').drop(columns=['year'])
    metabolism_2015 = metabolism_2015.merge(pop15_df, on='iso3', how='inner')
    metabolism_2015.drop_duplicates(subset=['iso3'], inplace=True)

    return global_metabolism, metabolism_2015

def create_metabolism_dataframe(config, verbose=False):
    """Main function to run the entire metabolism pipeline."""

    # Load country names
    with open(config['Names'], 'r') as f:
        country_names = json.load(f)

    # Process height data
    if verbose:
        print("Loading height data...")
    hdf_extended = load_height_data(config['height_csv_path'])
    if verbose:
        print(f"Height data loaded for {len(hdf_extended['Country'].unique())} countries")

    # Process population data
    if verbose:
        print("Loading population data...")
    pdf = load_population_data(config['population_csv_path'])
    if verbose:
        print(f"Population data loaded for {len(pdf['countryName'].unique())} countries")

    # Merge height and melt and merge population
    if verbose:
        print("Merging height and population data...")
    final_hdf_grouped = merge_height_population(hdf_extended, pdf, config)
    if verbose:
        print(f"Height and population data merged for {len(final_hdf_grouped['iso3'].unique())} countries")

    # Process BMI data
    if verbose:
        print("Loading BMI data...")
    bmi_stats_df = load_bmi_data(config['bmi_csv_path'])
    bmi_stats_df['iso3'] = bmi_stats_df['country'].map(country_names)
    if verbose:
        print(f"BMI data loaded for {len(bmi_stats_df['iso3'].unique())} countries")

    # Compute mass and BMR
    if verbose:
        print("Computing mass and BMR...")
    df = compute_mass_bmr(bmi_stats_df, final_hdf_grouped)
    if verbose:
        print(f"Mass and BMR computed for {len(df['iso3'].unique())} countries")


    # Process food supply data
    if verbose:
        print("Loading food supply data...")
    pivoted_df = load_food_supply_data(config)
    if verbose:
        print(f"Food supply data loaded for {len(pivoted_df['iso3'].unique())} countries")

    # Merge data and save
    bmr_df = pd.merge(
        df, 
        pivoted_df, 
        left_on=['iso3', 'year'], 
        right_on=['iso3', 'Year'], 
        how='left'
    ).drop(columns=['Area', 'Year'])

    bmr_df.to_csv(os.path.join('data', 'bmi_and_food_supply_data.csv'), index=False)

    bmr_df = pd.read_csv(os.path.join('data', 'bmi_and_food_supply_data.csv'))

    if verbose:
        print("Processing activity data...")
    metabolism_df, metabolism_2015 = process_activity_data(config, bmr_df, pdf)
    if verbose:
        print(f"Activity data processed for {len(metabolism_2015['iso3'].unique())} countries")

    return metabolism_df, metabolism_2015

###########################################################################
###                        Grid Data Processing                         ###
###########################################################################
def add_crop_production_data(ds, crop_prod_dir):
    netcdf_paths = os.path.join(crop_prod_dir, '*.nc')
    path_list = sorted(glob.glob(netcdf_paths))

    gaez_crop_prod_T = np.zeros(ds.grid_area.to_numpy().shape)
    for prod_path in path_list:
        crop_name = prod_path.split('/')[-1].split('.')[0]
        crop_ds = xr.open_dataset(prod_path)
        crop_ds = crop_ds.reindex(y=crop_ds.y[::-1]) # GAEZ data is flipped in y direction
        crop_da = crop_ds.band_data.sel(band=1).drop_vars('band').rename({'y':'lat','x':'lon'})
        ds = ds.assign({crop_name: (crop_da.dims, 1000 * crop_da.values)})
        gaez_crop_prod_T += ds[crop_name].to_numpy()
    ds['total_crop_prod'] = sum([ds[crop_prod] for crop_prod in list(ds.data_vars)[4:]])

    return ds

def add_livestock_data(ds, livestock_dir):
    """Add global livestock distribution data to the dataset
    
    Incorporates gridded livestock density data from FAO for different animal types
    and creates aggregate variables for dairy producers, poultry producers, and total livestock.
    """
    netcdf_paths = os.path.join(livestock_dir, '*.nc')
    path_list = sorted(glob.glob(netcdf_paths))

    for livestock_path in path_list:
        livestock_name = livestock_path.split('/')[-1].split('.')[0]
        livestock_ds = xr.load_dataset(livestock_path)
        livestock_da = livestock_ds[livestock_name]
        livestock_da = livestock_da.where(livestock_da > 1000)
        ds = ds.assign(**{livestock_name: livestock_da})

    # Make surrogates for combinations of livestock
    ds['DairyProducers'] = sum([ds[An].fillna(0) for An in ['Bf','Ct','Sh','Gt']])
    ds['PoultryProducers'] = sum([ds[An].fillna(0) for An in ['Dk','Ch']])
    ds['total_livestock'] = ds['Bf'] + ds['Ct'] + ds['Sh'] + ds['Gt'] + ds['Dk'] + ds['Ch']+ ds['Pg']
    
    return ds

def add_fish_catch_data(ds, fish_catch_path):
    """Add global fish catch data to the dataset
    
    Incorporates gridded marine fish catch data to complement
    terrestrial food production information.
    """
    fish_ds = xr.load_dataset(fish_catch_path)
    ds['catch_pel'] = fish_ds['catch_pel']
    ds['catch_dem'] = fish_ds['catch_dem']
    ds['total_catch'] = fish_ds['catch_dem'] + fish_ds['catch_pel']
    
    # Rescale catch data to be between 0 and 1 for easy global distribution
    ds['total_catch_frac'] = ds['total_catch'] / ds['total_catch'].sum()
    ds['catch_dem_frac'] = ds['catch_dem'] / ds['catch_dem'].sum()
    ds['catch_pel_frac'] = ds['catch_pel'] / ds['catch_pel'].sum()
    return ds

def add_fish_catch_data_watson(ds, fish_catch_path):
    fish_ds = xr.load_dataset(fish_catch_path) # Watson_2011.5.nc
    fish_ds.sel(time=2011.5).to_netcdf('data/Watson_2011.5.nc')
    fish2015_ds = ssm.grid_2_grid('data/Watson_2011.5.nc', variable_name="Catches", long_name="Catches", agg_function="SUM", netcdf_variable="Catches")
    fish_da = fish2015_ds.Catches
    ds['catch_pel'] = fish_da / 2
    ds['catch_dem'] = fish_da / 2
    ds['total_catch'] = fish_da
    
    return ds

def build_grid_dataset(config, fbs, verbose=False):
    """Construct a comprehensive gridded dataset of global food production
    
    Combines population data with crop production, livestock, and fish catch
    to create a spatially-explicit dataset of the global food system.
    """
    # Load base dataset with population and grid area
    if verbose:
        print("Loading base dataset with population and grid area...")
    pop_nc_path = config['population.nc']
    ds = xr.load_dataset(pop_nc_path)
    ds['pop2015'] = ds['population_count'].sel(time="2015-01-01")

    # Add crop production data
    if verbose:
        print("Adding crop production data...")
    ds = add_crop_production_data(ds, config['crop_prod_dir'])
    
    # Add livestock data
    if verbose:
        print("Adding livestock data...")
    ds = add_livestock_data(ds, config['livestock_dir'])
    
    # Add fish catch data
    if verbose:
        print("Adding fish catch data...")
    ds = add_fish_catch_data(ds, config['fish_catch.nc'])

    return ds

###########################################################################
###                 Downscaling with Dasymetric Mapping                 ###
###########################################################################
def downscale_animal_counts(ds, livestock_df, fao_country_to_region, lsu_df):
    abbrev_to_livestock = {'Bf': 'Buffalo', 'Ch': 'Chickens', 'Ct': 'Cattle', 'Gt': 'Goats', 'Pg': 'Swine / pigs', 'Sh': 'Sheep'}

    new_lsu_df = pd.DataFrame(list(fao_country_to_region.items()), columns=['ISO3', 'Region']).merge(lsu_df, on='Region')  
    for animal_key, animal_name in abbrev_to_livestock.items():
        #ds[animal_key + 'LSU'] = ssm.table_2_grid(animal_key, 'LSU', ds, tabular_file=livestock_df[livestock_df['Item'] == animal_name])['LSU']
        ds[animal_key + 'LSU'] = ssm.table_2_grid(surrogate_variable=animal_key, tabular_column='LSU', surrogate_data=ds, tabular_file=livestock_df[livestock_df['Item'] == animal_name])['LSU'] # New method

    ds['total_livestockLSU'] = sum([ds[AnLSU].fillna(0) for AnLSU in ['BfLSU','ChLSU','CtLSU','GtLSU','PgLSU','ShLSU']]) #'DkLSU',
    ds['DairyProducersLSU'] = sum([ds[AnLSU].fillna(0) for AnLSU in ['BfLSU','CtLSU','GtLSU','ShLSU']])
    ds['NonSpatial'] = xr.full_like(ds['total_livestock'], np.nan)

    return ds

def downscale_fao(ds, config, fbs, fbscatdf):
    with open(config['food_to_surrogate.json'], 'r') as json_file:
        food_to_surrogate = json.load(json_file)

    item_to_surrogate = {item: 'NonSpatial' for item in fbs['Item'].unique()}
    item_to_surrogate.update(food_to_surrogate)
    print('\nUncategorized items in fbs', set(food_to_surrogate.keys()) ^ set(fbs['Item'].unique()))

    for i, food in enumerate(list(fbs.Item.unique())):
        print(f'{100*i/len(list(fbs.Item.unique()))}% done, {food}')
        if item_to_surrogate[food] != 'NonSpatial':
            ds['fbs_prod_' + food] = ssm.table_2_grid(surrogate_variable=item_to_surrogate[food], tabular_column='ProdMCal ' + food, surrogate_data=ds, tabular_file=fbscatdf)['ProdMCal ' + food]
            ds['avail_prod_' + food] = ssm.table_2_grid(surrogate_variable=item_to_surrogate[food], tabular_column='AvailableProductionMCal ' + food, surrogate_data=ds, tabular_file=fbscatdf)['AvailableProductionMCal ' + food]
            # ds['fbs_prodT_' + food] = ssm.table_2_grid(item_to_surrogate[food], 'Production ' + food, ds, tabular_file=fbscatdf)['Production ' + food]
        # ds['fbs_supT_' + food] = ssm.table_2_grid('pop2015', 'FoodSupplyT ' + food, ds, tabular_file=fbscatdf)['FoodSupplyT ' + food]
        ds['fbs_cons_' + food] = ssm.table_2_grid(surrogate_variable='pop2015', tabular_column='ConsMCal ' + food, surrogate_data=ds, tabular_file=fbscatdf)['ConsMCal ' + food]
        ds['fbs_supply_' + food] = ssm.table_2_grid(surrogate_variable='pop2015', tabular_column='FoodSupplyMCal ' + food, surrogate_data=ds, tabular_file=fbscatdf)['FoodSupplyMCal ' + food]
        ds['fbs_feed_' + food] = ssm.table_2_grid(surrogate_variable='total_livestockLSU', tabular_column='FeedMCal ' + food, surrogate_data=ds, tabular_file=fbscatdf)['FeedMCal ' + food]

    # Marine and fish production handled separately
    marine_surrogates = {"Aquatic Animals, Others": "total_catch_frac",
                  "Aquatic Plants": "catch_dem_frac",
                  "Cephalopods": "total_catch_frac",
                  "Crustaceans": "catch_dem_frac",
                  "Demersal Fish": "catch_dem_frac",
                  "Marine Fish, Other": "total_catch_frac",
                  "Molluscs, Other": "catch_dem_frac",
                  "Pelagic Fish": "catch_pel_frac"}

    for food in marine_surrogates.keys():
        ds['fbs_prod_' + food] = fbs[fbs['Item'] == food]['ProdMCal'].sum() * ds[marine_surrogates[food]]
        ds['avail_prod_' + food] = fbs[fbs['Item'] == food]['AvailableProductionMCal'].sum() * ds[marine_surrogates[food]]
        # ds['fbs_prodT_' + food] = fbs[fbs['Item'] == food]['Production'].sum() * ds[marine_surrogates[food]]
    
    return ds

def downscale_metabolism(ds, metabolism_2015, fbscatdf, verbose=False):
    metabolism_2015['ISO3'] = metabolism_2015['iso3']

    ds['pop2015'] = ds['population_count'].sel(time="2015-01-01")
    metabolism_2015['bmr_MCal'] = metabolism_2015['BMR'] * metabolism_2015['total_population'] * 365 / 1e6
    metabolism_2015['tmr_MCal'] = metabolism_2015['TMR'] * metabolism_2015['total_population'] * 365 / 1e6

    foods = [dv[9:] for dv in ds.data_vars if dv[:9] == 'fbs_prod_']

    df = pd.merge(metabolism_2015, fbscatdf, on='ISO3', how='inner')
    df['FoodSupplyMCal'] = sum(df[f'FoodSupplyMCal {f}'].fillna(0) for f in foods)
    # Proportional TMR allocation to food supply
    for food in foods:
        df[f"tmr_{food}"] = df['tmr_MCal'] * df[f'FoodSupplyMCal {food}'].fillna(0) / df['FoodSupplyMCal']
        ds[f"tmr_{food}"] = ssm.table_2_grid(surrogate_variable='pop2015', tabular_column=f"tmr_{food}", surrogate_data=ds.sel(time='2015'), tabular_file=df, verbose=verbose)[f"tmr_{food}"]

    return ds

def downscale_grid_data(config, fbs, ds, fbscatdf, livestock_df, metabolism_2015, verbose=False):
    """Downscale various data to the grid level using dasymetric mapping techniques"""
    with open(config['fao_ctry_to_regions.json']) as json_file:
        fao_country_to_region = json.load(json_file)

    # Source: https://www.fao.org/3/i2294e/i2294e00.pdf
    # Source: https://www.fao.org/4/i2294e/i2294e00.htm
    lsu_data = {
        'Region': ['Near East North Africa', 'North America', 'Africa South of Sahara', 'Central America', 'South America', 'South Africa', 'OeCD', 'East and South East Asia', 'South Asia', 'Transition Markets', 'Caribbean', 'Near East', 'Other'],
        'Cattle': [0.70, 1.00, 0.50, 0.70, 0.70, 0.70, 0.90, 0.65, 0.50, 0.60, 0.60, 0.55, 0.60],
        'Buffalo': [0.70, np.nan, np.nan, np.nan, np.nan, np.nan, 0.70, 0.70, 0.50, 0.70, 0.60, 0.60, 0.60],
        'Sheep': [0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        'Goats': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        'Swine / pigs': [0.20, 0.25, 0.20, 0.25, 0.25, 0.20, 0.25, 0.25, 0.20, 0.25, 0.20, 0.25, 0.20],
        'Asses': [0.50, 0.50, 0.30, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        'Horses': [0.40, 0.80, 0.50, 0.50, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.56, 0.65],
        'Mules and hinnies': [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        'Camels': [0.75, np.nan, 0.70, np.nan, np.nan, np.nan, 0.90, 0.80, np.nan, np.nan, np.nan, 0.70, np.nan],
        'Chickens': [0.01, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    }

    lsu_df = pd.DataFrame(lsu_data)
    lsu_df.set_index('Region', inplace=True)

    if verbose:
        print("Downscaling animal counts...")
    ds = downscale_animal_counts(ds, livestock_df, fao_country_to_region, lsu_df)
    
    ds.to_netcdf(os.path.join(config['output_dir'], 'downscaled_data.nc'))
    if verbose:
        print("Downscaling FAO food balance sheet data...")
    ds = downscale_fao(ds, config, fbs, fbscatdf)

    ds.to_netcdf(os.path.join(config['output_dir'], 'downscaled_data.nc'))
    if verbose:
        print("Downscaling metabolism data...")
    ds = downscale_metabolism(ds, metabolism_2015, fbscatdf)
    
    return ds

###########################################################################
###                          Figure Generation                          ###
###########################################################################

# Helper plotting functions
def hist_da(da, title='', xlabel='', ylabel='', savefig_path='', xscale='log', color=None, color_spectrum=False, min_val=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Handle both single DataArray and list of DataArrays
    if isinstance(da, list):
        # Process multiple DataArrays
        all_data = []
        labels = []
        for d in da:
            data = d.values.flatten()
            data = data[~np.isnan(data)]
            all_data.append(data)
            labels.append(d.attrs['long_name'])

        # Find global min_val if not provided
        if min_val is None:
            min_val = min(np.log10(d[d > 0].min()) for d in all_data)
        
        # Create bins based on all data
        if xscale == 'log':
            max_val = max(d.max() for d in all_data)
            bins = np.logspace(min_val, np.log10(max_val), num=40)
        elif xscale == 'symlog':
            all_data = [d[abs(d) > 1] for d in all_data]
            max_val = max(d.max() for d in all_data)
            logbins = np.logspace(min_val, np.log10(max_val), num=20)
            bins = np.append(-logbins[::-1], logbins)
        
        # Plot each dataset
        colors = sns.color_palette("muted", len(da)) if color is None else [color] * len(da)
        for data, c, label in zip(all_data, colors, labels):
            hist_data = sns.histplot(data, bins=bins, color=c, alpha=0.7, edgecolor='black', linewidth=0.5, label=label)

            if color_spectrum:
                patches = hist_data.patches
                cmap = LinearSegmentedColormap.from_list("custom_cmap", [c, "dimgray"])
                for i, patch in enumerate(patches):
                    color_value = i / len(patches)
                    patch.set_facecolor(cmap(1 - color_value))
        
        plt.legend()
    else:
        # Process single DataArray (original behavior)
        data = da.values.flatten()
        data = data[~np.isnan(data)]

        if min_val is None:
            min_val = np.log10(data[data > 0].min())
        
        if xscale == 'log':
            bins = np.logspace(min_val, np.log10(data.max()), num=40)
        elif xscale == 'symlog':
            data = data[abs(data) > 1]
            logbins = np.logspace(min_val, np.log10(data.max()), num=20)
            bins = np.append(-logbins[::-1], logbins)

        if color is None:
            color = sns.color_palette("muted")[3]
        
        hist_data = sns.histplot(data, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        if color_spectrum:
            patches = hist_data.patches
            cmap = LinearSegmentedColormap.from_list("custom_cmap", [color, "dimgray"])
            for i, patch in enumerate(patches):
                color_value = i / len(patches)
                patch.set_facecolor(cmap(1 - color_value))

    plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)

def plot_da(da, cmap, norm, projection='Robinson', save_to_path=None):
    fig = plt.figure(figsize=(10, 5))
    if projection == 'Robinson':
        ax = plt.axes(projection=ccrs.Robinson())
    elif projection == 'PlateCarree':
        ax = plt.axes(projection=ccrs.PlateCarree())
    elif projection == 'Orthographic':
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
    ax.set_global() 
    ax.coastlines(linewidth=0.2)
    for feature in [cfeature.BORDERS]:#, cfeature.LAND, cfeature.OCEAN]:
        ax.add_feature(feature, linewidth=0.2)
        
    def custom_formatter(x, pos):
        """Custom formatter function to format ticks as 1e4, 5e5, etc."""
        if x == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / 10**exponent
        return f"{coeff:.0f}e{exponent}"
        
    pl = da.plot(transform=ccrs.PlateCarree(),
                 cmap=cmap,
                 norm=norm, 
                 cbar_kwargs={'orientation': 'horizontal',
                             #'ticks':[0, 1e11, 5e11, 1e12, 1e13, 5e13, 1e14],
                             #'format': custom_formatter,
                             'pad': 0.05,
                             'aspect': 50,
                             'shrink': 0.75,
                             'extend': 'min'})
    
    plt.tight_layout()
    if save_to_path:
        plt.savefig(save_to_path, bbox_inches='tight', pad_inches=0)

    return ax

def create_diverging_cmap(start_color, end_color, white_percentile=0.5):
    start_color = colors.to_rgb(start_color)
    end_color = colors.to_rgb(end_color)
    # Create a dictionary to define the colormap
    cdict = {
        'red':   [[0.0, start_color[0], start_color[0]],
                  [white_percentile, 1.0, 1.0],
                  [1.0, end_color[0], end_color[0]]],

        'green': [[0.0, start_color[1], start_color[1]],
                  [white_percentile, 1.0, 1.0],
                  [1.0, end_color[1], end_color[1]]],

        'blue':  [[0.0, start_color[2], start_color[2]],
                  [white_percentile, 1.0, 1.0],
                  [1.0, end_color[2], end_color[2]]]
    }
    custom_cmap = colors.LinearSegmentedColormap('CustomMap', segmentdata=cdict, N=256)
    return custom_cmap

# Main figures
# Figure 1
def make_fbsv_for_Voronoi(fbs, ds, output_dir):
    foods = [dv[9:] for dv in ds.data_vars if dv[:9] == 'fbs_prod_']

    for food in foods:
        grid_total = ds['fbs_prod_' + food].sum().item()
        fbs_total = fbs[fbs['Item'] == food]['ProdMCal'].sum().item()
        try:
            assert(round(grid_total - fbs_total) == 0)
        except:
            print(food, round(grid_total - fbs_total))

    fbsv = fbs.rename(columns={ 'Production': 'ProdkT',
                                'Processing': 'ProckT',
                                'Feed': 'FeedkT',
                                'Food': 'ConskT'})

    fbsv['h2'] = fbsv['Type'].replace({'Fruits':'Fruits, Vegetables, Nuts', 'Vegetables':'Fruits, Vegetables, Nuts', 'Nuts':'Fruits, Vegetables, Nuts',
                                    'AlcoholicBeverages':'Spices, Sweeteners, and Beverages', 'StimulantsAndSpices':'Spices, Sweeteners, and Beverages',
                                    'SugarCrop': 'Sugar', 'Pulses':'Pulses, Roots, Tubers', 'Roots':'Pulses, Roots, Tubers',
                                    'Meat': 'Animal Products', 'NonMeatAnimalProduct': 'Animal Products', 'Seafood': 'Animal Products',
                                    'Oil': 'Oilcrops'})

    #columns = ['CropProdMCal', 'AnimProdMCal', 'ProcProdMCal', 'FeedMCal', 'ProcMCal', 'FoodMCal', 'OtherMCal']
    columns = [col for col in fbsv.columns if col[-4:] == 'MCal']

    fbsv = fbsv[['Item', 'h2'] + columns]

    fbsv = fbsv.groupby(['Item', 'h2']).sum().reset_index()
    fbsv.rename(columns={'Item': 'h3'}, inplace=True)

    # Compute total metabolism per grid cell
    ds['met_Cal_p_grid_cell'] = sum([ds[dv].fillna(0) for dv in ds.data_vars if dv[:] == 'tmr_'])

    # Add Metabolism Column
    total_cons_supply_grid = sum([ds[dv].fillna(0) for dv in ds.data_vars if dv[:8] == 'fbs_cons']) #* 10**12 / ds['grid_area'] / 365  #6 for m to km, 6 for MC to C
    metabolism = {}
    for food in foods:
        metabolism[food] = (ds['fbs_cons_' + food].fillna(0) / total_cons_supply_grid.fillna(0) * ds['met_Cal_p_grid_cell'].fillna(0)).sum().item()

    def map_metabololsim(food):
        if food in metabolism.keys():
            return metabolism[food]
        else:
            return 0

    fbsv['Metabolism'] = fbsv['h3'].map(map_metabololsim)

    # Add Lancet Diet column
    lancet_diet = [
        ['Whole Grains', 'Cereals', 811],
        ['Starchy Tubers', 'Pulses, Roots, Tubers', 39],
        ['Vegetables', 'Fruits, Vegetables, Nuts', 78],
        ['Fruits','Fruits, Vegetables, Nuts', 126],
        ['Dairy', 'Animal Products', 153],
        ['Red Meat', 'Animal Products', 30],
        ['Poultry', 'Animal Products', 62],
        ['Eggs', 'Animal Products', 19],
        ['Fish', 'Animal Products', 40],
        ['Legumes', 'Pulses, Roots, Tubers', 284],
        ['Nuts', 'Fruits, Vegetables, Nuts', 291],
        ['Oils', 'Oilcrops', 450],
        ['Sugars', 'Sugar', 120]
    ]

    fbsv['Lancet'] = 0
    row = {col: 0 for col in fbsv.columns}
    for food, category, calories in lancet_diet:
        row.update({'Lancet': calories, 'h3': food, 'h2': category})
        fbsv.loc[len(fbsv)] = row
        
    # Adding colors based on category and food category
    turquoise, orange, blue, pink, green, yellow, brown = sns.color_palette("Set2", len(fbsv['h2'].unique()))
    cat_color_dict = {'Fruits, Vegetables, Nuts': green,
                    'Animal Products': pink,
                    'Cereals': yellow,
                    'Pulses, Roots, Tubers': brown,
                    'Spices, Sweeteners, and Beverages': blue,
                    'Oilcrops': orange,
                    'Sugar': turquoise}
    # category_colors = sns.color_palette("Set2", len(fbsv['h2'].unique()))
    #category_color_dict = dict(zip(fbsv['h2'].unique(), category_colors))
    fbsv['color'] = fbsv['h2'].map(cat_color_dict)

    # Convert RGB colors to hexadecimal
    fbsv['color'] = fbsv['color'].apply(lambda x: to_hex(x, keep_alpha=False))
    fbsv = pd.melt(fbsv, id_vars=['h2', 'h3', 'color'], var_name='h1', value_name='weight')
    fbsv = fbsv[['h1', 'h2', 'h3', 'color', 'weight']]

    # Vary colors
    x = 0.2  # 0 means no change; 0.5 means large change

    np.random.seed(42)
    def adjust_color(color, x):
        import colorsys
        hslcol = np.array(colorsys.rgb_to_hls(*tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))))
        levs = np.random.uniform(1-x, 1+x) * hslcol[2]
        hslcol[2] = levs
        rgbcol = tuple(min(max(round(i * 255), 0), 255) for i in colorsys.hls_to_rgb(*hslcol))
        return f'#{rgbcol[0]:02X}{rgbcol[1]:02X}{rgbcol[2]:02X}'

    fbsv['color'] = fbsv.groupby('h3')['color'].transform(lambda color: adjust_color(color.iloc[0], x))

    # Normalize weights to be percentages
    voronoi_weights = {}
    for h1val in columns:
        selected_rows = fbsv[fbsv['h1'] == h1val]
        total_weight = selected_rows['weight'].sum()
        if h1val[-4:] == 'MCal':
            C = total_weight * 10**6 / (7.4 * 10**9) / 365
            print(h1val[:-4] + ':')
            print('\t\t', round(C), 'Cal/person/day')
            voronoi_weights[h1val[:-4]] = C
        fbsv.loc[selected_rows.index, 'weight'] = selected_rows['weight'] / total_weight * 100

    prod = voronoi_weights['CropProd'] + voronoi_weights['AnimProd'] + voronoi_weights['ProcProd']
    cons = voronoi_weights['FoodSupply'] + voronoi_weights['Feed'] + voronoi_weights['Proc'] + voronoi_weights['Other']
    print(voronoi_weights['FoodSupply'], voronoi_weights['Feed'], voronoi_weights['Proc'], voronoi_weights['Other'])
    assert(round(prod) == round(cons))

    fbsv['weight'] = fbsv['weight'].round(5) 
    fbsv.to_csv(os.path.join(output_dir,'fbsv.csv'))

# Figure 2
def make_metabolism_production_maps(ds, image_dir, map_projection='Robinson', thresh=5000):
    my_cmap = sns.color_palette("flare", as_cmap=True)
    my_norm = colors.LogNorm(vmin=thresh, vmax=1e7)
    my_norm = colors.Normalize(vmin=thresh, vmax=1e6)

    ##################
    ### PRODUCTION ###
    ##################
    prod_mcal_p_year = sum([ds[dv].fillna(0) for dv in ds.data_vars if dv[:8] == 'fbs_prod'])
    prod_p_sqkm_p_day_da = prod_mcal_p_year * 10**12 / ds['grid_area'] / 365  #6 for m to km, 6 for MC to C
    prodda = prod_p_sqkm_p_day_da.where(prod_p_sqkm_p_day_da.to_numpy() > 1e-10).copy() # save pre-masked data for histogram

    has_no_data_mask = xr.full_like(ds['grid_area'], fill_value=True, dtype=bool)
    for da in [ds[dv] for dv in ds.data_vars if dv[:8] == 'fbs_prod']:
        has_no_data_mask &= np.isnan(da)
    has_data_mask = ~has_no_data_mask

    min_thresh_mask = prod_p_sqkm_p_day_da.to_numpy() > thresh
    mask = has_data_mask & min_thresh_mask
    prod_p_sqkm_p_day_da = prod_p_sqkm_p_day_da.where(mask, other=np.nan)

    #prod_p_sqm_p_day_da /= 1e3 # convert to units of Millions
    prod_p_sqkm_p_day_da.attrs = {'long_name': 'Total Production', 'units': 'Calories / sqkm / day'}
    plot_da(prod_p_sqkm_p_day_da, cmap=my_cmap, norm=my_norm, projection=map_projection, save_to_path=os.path.join(image_dir,'total_prod_map.png'))

    ##################
    ### METABOLISM ###
    ##################
    tmr = sum([ds[dv].fillna(0) for dv in ds.data_vars if dv[:3] == 'tmr'])
    tmrda = tmr * 10**12 / ds['grid_area'] / 365 #6 for m to km, 6 for MC to C
    tmrda_full = tmrda.copy()
    min_thresh_mask = tmrda.to_numpy() > thresh
    tmrda = tmrda.where(min_thresh_mask)

    #tmrda /= 1e3 # convert to units of Millions
    tmrda.attrs = {'long_name': 'Total Metabolic Rate', 'units': 'Calories / sqkm / day'}

    plot_da(tmrda, cmap=my_cmap, norm=my_norm, projection=map_projection, save_to_path=os.path.join(image_dir,'metabolism_map.png'))
    
    # Histograms
    # hist_da(prodda, title='Distribution of Caloric Production', xlabel='Calories Production per sqkm per day', ylabel='Number of Grid Cells', savefig_path=os.path.join(image_dir,'prod_dist_hist.png'), min_val=-2)
    # hist_da(tmrda_full, title='Distribution of Human Metabolism', xlabel='Calories Metabolized per sqkm per day', ylabel='Number of Grid Cells', savefig_path=os.path.join(image_dir,'cons_dist_hist.png'), min_val=-2)
    prodda.attrs = {'long_name': 'Production Density'}
    tmrda_full.attrs = {'long_name': 'Metabolism Density'}
    # title='Distribution of Human Metabolism and Production',
    hist_da([prodda, tmrda_full], xlabel='Calories per sqkm per day', ylabel='Number of Grid Cells', savefig_path=os.path.join(image_dir,'prod_cons_dist_hist.png'), min_val=-2)

# Figure 3
def make_netflow_maps(ds, fbs, image_dir):
    foods = [dv[9:] for dv in ds.data_vars if dv[:8] == 'fbs_prod']

    fbs['Category'] = fbs['Type'].replace({'Fruits':'Fruits, Vegetables, Nuts', 'Vegetables':'Fruits, Vegetables, Nuts', 'Nuts':'Fruits, Vegetables, Nuts',
                                    'AlcoholicBeverages':'Spices, Sweeteners, and Beverages', 'StimulantsAndSpices':'Spices, Sweeteners, and Beverages',
                                    'SugarCrop': 'Sugar', 'Pulses':'Pulses, Roots, Tubers', 'Roots':'Pulses, Roots, Tubers',
                                    'Meat': 'Animal Products', 'NonMeatAnimalProduct': 'Animal Products', 'Seafood': 'Animal Products',
                                    'Oil': 'Oilcrops'})
    turquoise, orange, blue, pink, green, yellow, brown = sns.color_palette("Set2", len(fbs['Category'].unique()))

    cat_color_dict = {'Fruits, Vegetables, Nuts': green,
                    'Animal Products': pink,
                    'Cereals': yellow,
                    'Pulses, Roots, Tubers': brown,
                    'Spices, Sweeteners, and Beverages': blue,
                    'Oilcrops': orange,
                    'Sugar': turquoise}

    for type in fbs.Category.unique():
        ds['netf_'+type] = xr.zeros_like(ds['grid_area'])
        ds['tot_prod_'+type] = xr.zeros_like(ds['grid_area'])
        ds['tot_cons_'+type] = xr.zeros_like(ds['grid_area'])
        foods_of_this_type = [food for food in foods if food in fbs[fbs.Category == type].Item.unique()]
        for food in foods_of_this_type:
            ds['tot_prod_'+type] += ds['fbs_prod_' + food].fillna(0)
            ds['tot_cons_'+type] += ds['fbs_cons_' + food].fillna(0)
            ds['netf_'+type] += (ds['fbs_prod_' + food].fillna(0) - ds['fbs_cons_' + food].fillna(0)) / ds['grid_area'] / 365 * 10**12  #+ ds['fbs_feed_' + food].fillna(0))

        ds['netf_'+type].attrs = {
            'long_name': 'Net Caloric Surplus of ' + type,
            'units': 'Calories / sqkm / day',
            'description': 'This variable contains total caloric surplus/deficit for each grid cell'
        }
        
        my_cmap = create_diverging_cmap('black', cat_color_dict[type])
        my_norm = colors.Normalize(vmin=-1e5, vmax=1e5)
        plot_da(ds['netf_'+type], cmap=my_cmap, norm=my_norm, save_to_path=os.path.join(image_dir,'nf_'+type+'_map.png'))

        #ds['netf'] = sum([ds['netf_'+type] for type in fbs.Category.unique()])

# Old Figure 4
def food_supply_metabolism_regression_w_residuals(fbs, config):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.5, 1]})


    ###########################################################################
    #                                                                         #
    #                         Read in Demographic Data                        #
    #                                                                         #
    ###########################################################################

    with open(config['Names']) as json_file:
        Names = json.load(json_file)

    bmr = pd.read_csv('/Users/maxwellkaye/Documents/PhD Research/data/food_data/BMI_BMR.csv')
    fbs['ISO'] = fbs['Area'].apply(lambda s: Names[s])
    fbs[['ISO','FoodSupplyMCal']].groupby('ISO').sum()
    df = fbs[['ISO','FoodSupplyMCal']].groupby('ISO').sum().reset_index()

    df = df.merge(bmr[['ISO','approx_bmr']], on='ISO')
    pop_df = fbs[['Area','pop2015']].groupby('Area').mean().reset_index()
    pop_df['ISO'] = pop_df['Area'].apply(lambda s: Names[s])
    df = df.merge(pop_df, on='ISO')

    df['total_metabolic_rate'] = df['approx_bmr'] * 5 / 3
    df['CalpPersonpDay'] = df['FoodSupplyMCal'] / df['pop2015'] * 10**6 / 365

    # Drop Duplicates form Region Classification
    df = df[~df['Area'].isin(['Micronesia (Federated States of)', 'China'])]

    # Include GDP from World Bank 2015
    gdp = pd.read_csv('/Users/maxwellkaye/Documents/PhD Research/data/world_bank_gdp.csv')
    df = df.merge(gdp[['Country Code', '2015']], left_on='ISO', right_on='Country Code', how='left').drop(columns=['Country Code'])
    df = df.rename(columns={'2015':'2015 GDP'})

    # Make a column that is empty strings for small countires, but keeps ISO for others
    def kill_small_ctrys(iso):
        if df[df['ISO'] == iso]['pop2015'].item() < 1e8:
            return ''
        return iso
    df['ISO_Large_Pop_Only'] = df['ISO'].map(kill_small_ctrys)

    ###########################################################################
    #                                                                         #
    #                             Set up the Plot                             #
    #                                                                         #
    ###########################################################################

    # Parameters
    header1 = 'total_metabolic_rate'
    header2 = 'CalpPersonpDay'
    xlabel = 'Average National Total Metabolic Rate cal/person/day'
    ylabel = 'Average National Food Supply (cal/person/day)'
    title = 'Food Supply Vs. BMI Derived Average Metabolic Rate'

    x = df[header1]
    y = df[header2]
    mask = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    gdp = (df['2015 GDP'] / df['pop2015'])[mask]
    pop = df['pop2015'][mask]

    my_norm = colors.LogNorm()# colors.TwoSlopeNorm(vmin=gdp.min()*.01, vcenter=gdp.mean(), vmax=gdp.max())
    my_cmap = sns.color_palette("viridis", as_cmap=True)
    scatter = ax1.scatter(x, y, c=gdp, cmap=my_cmap, norm=my_norm, s=pop/1e5, alpha=.8)

    ###########################################################################
    #                                                                         #
    #               Plot the line of best fit and 1 to 1 line                 #
    #                                                                         #
    ###########################################################################

    slope, intercept, r_value, best_fit_p_value, std_err = stats.linregress(x, y)
    best_line = slope * x + intercept
    ax1.plot(x, best_line, color='black', linestyle='--', linewidth=.8,label=f'Unweighted Best Fit (r={r_value:.2f}, m={slope:.2f})')
    ax1.plot(x, x, color='black', label=f'One to One Line')

    ###########################################################################
    #                                                                         #
    #               Plot the population weighted best fit line                #
    #                                                                         #
    ###########################################################################
    w = df['pop2015'][mask]
    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)

    # Weighted slope and intercept
    slope = np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w * (x - w_mean_x)**2)
    intercept = w_mean_y - slope * w_mean_x

    # Calculate the regression line
    line = slope * x + intercept

    # Calculate the R-squared value for the weighted regression
    y_pred = slope * x + intercept
    ss_res = np.sum(w * (y - y_pred)**2)
    ss_tot = np.sum(w * (y - w_mean_y)**2)
    r_value = np.sqrt(1 - ss_res / ss_tot)

    # Plot the weighted regression line
    ax1.plot(x, line, color='black',  linestyle=':', label=f'Population Weighted Fit (R={r_value:.2f}, m={slope:.2f})')

    ###########################################################################
    #                                                                         #
    #               Annotate, Label axes, Title, Save, and Show               #
    #                                                                         #
    ###########################################################################
    for i, cntry_name in df['ISO_Large_Pop_Only'][mask].items():
        ax1.annotate(cntry_name, (x[i], y[i]))

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()

    ###########################################################################
    #                                                                         #
    #               Plot GDP vs Population Weighted Fit                       #
    #                                                                         #
    ###########################################################################

    loggdp = np.log10(gdp)

    x = best_line
    y = df['CalpPersonpDay']

    maskp = ~np.isnan(y-x) & ~np.isnan(loggdp) & np.isfinite(y-x) & np.isfinite(loggdp)
    xp = (loggdp)[maskp] # gdp
    yp = (y-x)[maskp]    # residuals
    pop = df['pop2015'][maskp]

    for i, cntry_name in df['ISO_Large_Pop_Only'][maskp].items():
        ax2.annotate(cntry_name, (xp[i], yp[i]))

    slope, intercept, r_value, residual_p_value, std_err = stats.linregress(xp, yp)
    line = slope * xp + intercept
    cbar = plt.colorbar(scatter, label='GDP per Capita')
    scatter = ax2.scatter(xp,yp, s=pop/1e5, alpha=.8) #,cmap=my_cmap, norm=my_norm, c=gdp[maskp])


    #ax2.plot(xp, line, color='black', label=f'Best Fit (R={r_value:.2f}, m={slope:.2f})')
    ax2.plot(xp, np.zeros(len(xp)), color='black', linestyle=':', label='Zero Line')
    ax2.set_ylabel('Residual Calories to the Best Fit (kcal/person/day)')
    ax2.set_xlabel('Log GDP Per Capita')
    ax2.set_title(' GDP per Capita vs Best Fit Residual Food Supply Calories')

    #plt.legend()
    plt.legend(loc='lower left')#, bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(os.path.join(config['image_dir'],'metabolism_food_supply_with_residual_vs_gdp.png'))
    #plt.show()
    # print('Best fit p value: ', best_fit_p_value)
    # print('Residual correlatipn r^2 vlaue: ', r_value**2)    

# New Figure 4 is made using the sesame python package plot_country function

# Figure 5a
def food_supply_metabolism_regression(fbs, metabolism_2015, config):
    #fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.5, 1]})
    fig, ax1 = plt.subplots(figsize=(10, 8))

    with open(config['Names']) as json_file:
        Names = json.load(json_file)

    fbs['ISO'] = fbs['Area'].apply(lambda s: Names[s])
    fbs[['ISO','FoodSupplyMCal']].groupby('ISO').sum()
    df = fbs[['ISO','FoodSupplyMCal']].groupby('ISO').sum().reset_index()

    df = df.merge(metabolism_2015[['iso3','TMR']], left_on='ISO', right_on='iso3')

    pop_df = fbs[['Area','pop2015']].groupby('Area').mean().reset_index()
    pop_df['ISO'] = pop_df['Area'].apply(lambda s: Names[s])
    df = df.merge(pop_df, on='ISO')

    #df['total_metabolic_rate'] = df['BMR'] * 5 / 3
    df['total_metabolic_rate'] = df['TMR']
    df['CalpPersonpDay'] = df['FoodSupplyMCal'] / df['pop2015'] * 10**6 / 365

    # Drop Duplicates form Region Classification
    df = df[~df['Area'].isin(['Micronesia (Federated States of)', 'China'])]

    # Include GDP from World Bank 2015
    gdp = pd.read_csv('/Users/maxwellkaye/Documents/PhD Research/data/world_bank_gdp.csv')
    df = df.merge(gdp[['Country Code', '2015']], left_on='ISO', right_on='Country Code', how='left').drop(columns=['Country Code'])
    df = df.rename(columns={'2015':'2015 GDP'})

    # Make a column that is empty strings for small countires, but keeps ISO for others
    def kill_small_ctrys(iso):
        if df[df['ISO'] == iso]['pop2015'].item() < 1e8:
            return ''
        return iso
    df['ISO_Large_Pop_Only'] = df['ISO'].map(kill_small_ctrys)

    # Parameters
    header1 = 'total_metabolic_rate'
    header2 = 'CalpPersonpDay'
    xlabel = 'Average National Total Metabolic Rate cal/person/day'
    ylabel = 'Average National Food Supply (cal/person/day)'
    title = 'Food Supply Vs. Average Metabolic Rate'

    x = df[header1]
    y = df[header2]
    mask = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    gdp = (df['2015 GDP'] / df['pop2015'])[mask]
    pop = df['pop2015'][mask]

    from matplotlib.colors import LogNorm
    my_norm = LogNorm()# colors.TwoSlopeNorm(vmin=gdp.min()*.01, vcenter=gdp.mean(), vmax=gdp.max())
    my_cmap = sns.color_palette("viridis", as_cmap=True)
    scatter = ax1.scatter(x, y, c=gdp, cmap=my_cmap, norm=my_norm, s=pop/1e5, alpha=.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('GDP per capita (USD)')

    # slope, intercept, r_value, best_fit_p_value, std_err = stats.linregress(x, y)
    # best_line = slope * x + intercept
    w = df['pop2015'][mask]
    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)

    # Weighted slope and intercept
    slope = np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w * (x - w_mean_x)**2)
    intercept = w_mean_y - slope * w_mean_x

    # Calculate the regression line
    xp = np.linspace(2000, 3000, 100)
    line = slope * xp + intercept

    # Calculate the R-squared value for the weighted regression
    y_pred = slope * x + intercept
    ss_res = np.sum(w * (y - y_pred)**2)
    ss_tot = np.sum(w * (y - w_mean_y)**2)
    r_value = np.sqrt(1 - ss_res / ss_tot)

    # Plot the weighted regression line
    ax1.plot(xp, line, color='black',  alpha=0.5, linestyle='-', label=f'Weighted Best Fit (R={r_value:.2f}, m={slope:.2f})')

    for i, cntry_name in df['ISO_Large_Pop_Only'][mask].items():
        if cntry_name:  # Only annotate non-empty labels
            # Calculate offset based on point size (pop/1e5)
            offset = (pop[i]/1e5)**.5 / np.pi # Adjust the multiplier (50) to control offset distance
            ax1.annotate(cntry_name, 
                        (x[i], y[i]),
                        xytext=(-13, -offset),  # Move text left by offset amount
                        textcoords='offset points',
                        va='center')  # Vertically center the text

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()
    #ax1.set_aspect('equal')
    xmin = 1950
    xmax = 3000
    ymin = 1850
    ymax = 3900
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_xticks(np.arange(2000, xmax, 250))
    ax1.set_yticks(np.arange(2000, ymax, 250))

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(config['image_dir'],'metabolism_food_supply_gdp.png'))

# Figure 5b
def metabolism_time_series(global_metabolism, config):
    import scipy.stats as stats
    fig, ax = plt.subplots(figsize=(10, 8))
    # Calculate linear fits
    tmr_slope, tmr_intercept, _, _, _ = stats.linregress(global_metabolism.year, global_metabolism.TMR)
    food_slope, food_intercept, _, _, _ = stats.linregress(global_metabolism.year, global_metabolism.foodSupply)
    
    # Plot lines
    ax.plot(global_metabolism.year, global_metabolism.TMR, 'k', linewidth=3, 
             label=f'Total Metabolism (95% CI)')
    ax.plot(global_metabolism.year, tmr_slope * global_metabolism.year + tmr_intercept, 
             'k', alpha=0.3, linewidth=2, label=f'best fit ({tmr_slope:.1f} Cal/cap/day/yr)')
    
    ax.plot(global_metabolism.year, global_metabolism.foodSupply, 'darkgreen', linewidth=3, 
             label=f'FAO Food Supply')
    ax.plot(global_metabolism.year, food_slope * global_metabolism.year + food_intercept, 
             'darkgreen', alpha=0.3, linewidth=2, label=f'best fit ({food_slope:.1f} Cal/cap/day/yr)')
    
    # Plot confidence interval
    lower = global_metabolism.TMR - global_metabolism.ci95
    upper = global_metabolism.TMR + global_metabolism.ci95
    ax.fill_between(global_metabolism.year, lower.values, upper.values, color='grey', alpha=0.3)
    #plt.plot(global_metabolism.year, global_metabolism.BMR, 'k--', linewidth=3, label='Basal metabolism')
    ax.set_ylim([2000,3100])
    ax.set_xlim([1992, 2019])
    ax.set_yticks(np.arange(2000,3100,200))
    ax.set_ylabel('Calories per capita per day')
    ax.set_xlabel('Year')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(config['image_dir'],'metabolism_time_series.png'))

# All figures
def create_figures(ds, fbs, metabolism_2015, metabolism_df, config, verbose=False):
    """Generate all visualizations for the food system analysis"""
    # Create output directory if it doesn't exist
    image_dir = config['image_dir']
    os.makedirs(image_dir, exist_ok=True)

    plt.rcParams.update({'font.size': 14}) # Change the font size globally
    my_cmap = sns.color_palette("flare", as_cmap=True)
    my_norm = colors.Normalize(vmin=1e2, vmax=1e8)

    # Create individual visualizations
    if verbose:
        print("Creating Figure 1: Food flow Sankey diagram...")
    make_fbsv_for_Voronoi(fbs, ds, config['output_dir'])
    
    if verbose:
        print("Creating Figure 2: Metabolism and production maps...")
    make_metabolism_production_maps(ds, image_dir)
    
    if verbose:
        print("Creating Figure 3: Net flow maps...")
    make_netflow_maps(ds, fbs, image_dir)
    
    if verbose:
        print("Creating Figure 4: Food supply metabolism regression...")
    food_supply_metabolism_regression(fbs, metabolism_2015, config)
    
    if verbose:
        print("Creating Figure 5: Metabolism time series...")
    metabolism_time_series(metabolism_df, config)


def add_ctry_dv(ds_with_coords):
    # Load country fraction dataset
    ctry_frac_ds = xr.open_dataset(os.path.join(ssm.__path__[0], 'data', 'country_fraction.1deg.2000-2023.a.nc'))
    ctry_frac_ds = ctry_frac_ds.sel(time='2015').squeeze('time')
    # Reindex latitude to ascending order if needed
    if np.any(np.diff(ctry_frac_ds.lat.values) < 0):
        ctry_frac_ds = ctry_frac_ds.reindex(lat=ctry_frac_ds.lat[::-1])
    ctrys = list(ctry_frac_ds.data_vars)
    index_2_ctry = {str(i): ctry for i, ctry in enumerate(ctrys)}

    # Round and binarize country fractions
    for ctry in ctrys:
        ctry_frac_ds[ctry] = (ctry_frac_ds[ctry].round() > 0).astype('bool')

    # Build mask and find first nonzero country index at each (lat, lon)
    fracs = [ctry_frac_ds[ctry].values for ctry in ctrys]
    nonzero_mask = np.stack(fracs, axis=0)
    any_nonzero = np.any(nonzero_mask, axis=0)
    first_nonzero_idx = np.argmax(nonzero_mask, axis=0)
    # Assign country index or np.nan if no country present
    ctry_index_arr = np.where(any_nonzero, first_nonzero_idx, np.nan)

    # Add as a data variable to ds_with_coords
    ds_with_coords = ds_with_coords.assign(
        ctry=(('lat', 'lon'), ctry_index_arr)
    )
    ds_with_coords['ctry'].attrs = index_2_ctry
    return ds_with_coords

def ds_to_food_cube(ds, config, vars = ['tmr', 'avail_prod', 'fbs_prod', 'fbs_supply'], out_path='food_cube.nc'):
    '''
    Convert the ds to a cube with dimensions lat, lon, food
    '''
    # Create a list of food items from the existing dataset
    food_items = []
    for var_name in ds.data_vars:
        if var_name.startswith('tmr_'):
            food_items.append(var_name[4:])  # Remove 'tmr_' prefix

    # Create the new dataset with dimensions lat, lon, food
    new_ds = xr.Dataset(
        coords={
            'lat': ds.lat,
            'lon': ds.lon,
            'food': food_items
        }
    )

    for var in vars:
        data = []
        for food in food_items:
            data.append(ds[f'{var}_{food}'].values)
        
        data = np.stack(data, axis=-1)  # Stack along new food dimension
        data = np.maximum(data, 0)  # Round negative values up to 0 (this is the only way to get rid of the negative values from floating point error)
        data = data * 1e6 / 365 # Convert to Cal/day
        # Create DataArray with proper attributes and metadata
        new_ds[var] = xr.DataArray(
            data=data,
            dims=['lat', 'lon', 'food'],
            coords={
                'lat': ds.lat, 
                'lon': ds.lon, 
                'food': food_items
            },
            attrs={
                'long_name': f'{var} by food type',
                'units': 'kcal/day/grid cell' if var in ['tmr', 'avail_prod', 'fbs_prod', 'fbs_supply'] else 'dimensionless',
                'description': f'{var} values by food type across spatial dimensions'
            }
        )

    new_ds = new_ds.assign({
        # 'mss': xr.where(new_ds['tmr'] < new_ds['avail_prod'], new_ds['tmr'], new_ds['avail_prod']),
        'population_count': ds['population_count'].sel(time='2015').squeeze()
    })

    # Round all values to nearest integer
    for var_name in new_ds.data_vars:
        new_ds[var_name] = new_ds[var_name].fillna(0).round()

    new_ds = add_ctry_dv(new_ds)
    new_ds.to_netcdf(os.path.join(config['output_dir'], out_path))

    return new_ds

###########################################################################
###                            Main Function                            ###
###########################################################################

def main(config_path, verbose=False, generate_data=False):
    """Main execution function for the food system analysis pipeline
    
    Orchestrates the entire data processing and visualization workflow
    from raw data to final outputs and figures.
    """
    # Load configuration
    if verbose:
        print("Loading configuration from:", config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    if generate_data:
        # Process tabular data
        if verbose:
            print("\nProcessing tabular data...")
        fbs, fbscatdf, livestock_df = process_tabular_data(config, verbose)

        # Process metabolism data
        if verbose:
            print("\nProcessing metabolism data...")
        metabolism_df, metabolism_2015 = create_metabolism_dataframe(config, verbose)

        if verbose:
            print(f"\nSaving csv files to {output_dir}")
        fbs.to_csv(os.path.join(output_dir, 'fbs.csv'))
        fbscatdf.to_csv(os.path.join(output_dir, 'fbscat.csv'))
        livestock_df.to_csv(os.path.join(output_dir, 'livestock.csv'))
        metabolism_df.to_csv(os.path.join(output_dir, 'metabolism_time_series.csv'))
        metabolism_2015.to_csv(os.path.join(output_dir, 'metabolism_2015.csv'))
        
        # Build grid dataset
        if verbose:
            print("\nBuilding grid dataset...")
        ds = build_grid_dataset(config, fbs, verbose)

        # Downscale data
        if verbose:
            print("\nDownscaling grid data...")
        ds = downscale_grid_data(config, fbs, ds, fbscatdf, livestock_df, metabolism_2015, verbose=False)

        # Save outputs
        if verbose:
            print(f"\nSaving food_ds.nc to {output_dir}...")
        ds.to_netcdf(os.path.join(output_dir, 'food_ds.nc'))

    fbs = pd.read_csv(os.path.join(output_dir, 'fbs.csv'))
    ds  = xr.load_dataset(os.path.join(output_dir, 'food_ds.nc'))
    metabolism_df = pd.read_csv(os.path.join(output_dir, 'metabolism_time_series.csv'))
    metabolism_2015 = pd.read_csv(os.path.join(output_dir, 'metabolism_2015.csv'))
    
    # Create final data table
    # if verbose:
        # print("creating food cube")
    # ds_to_food_cube(ds, config, out_path='food_cube.nc')

    # Create visualizations
    if verbose:
        print("\nCreating visualizations...")
    create_figures(ds, fbs, metabolism_2015, metabolism_df, config, verbose)
    
    if verbose:
        print("Processing complete!")

if __name__ == "__main__":
    # To run this script: python main.py --config path/to/your/config.json
    # If no config path is provided, it will default to 'config.json' in the current directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode with detailed output')
    parser.add_argument('--generate_data', action='store_true', help='Generate the output data (without this argumnet, just visualizes existing data)')
    args = parser.parse_args()
    main(args.config, args.verbose, args.generate_data)
