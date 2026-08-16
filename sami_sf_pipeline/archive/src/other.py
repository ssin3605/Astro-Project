# from astropy.io import fits

# file_path = "data/fits/gama_pairs.fits"

# with fits.open(file_path) as hdul:
#     hdul.info()  # shows structure of the FITS file

#     # Assuming the table is in the first extension (common for catalogues)
#     data = hdul[1].data
#     print("\nColumn headings:\n")
#     print(data.names)

import pandas as pd
df = pd.read_csv('master/master_results.csv')

# Check dtypes and NaN counts for key columns
print(df[['gini_all_halpha', 'CIndex', 'A_min', 'S', 
          'log_SFR', 'log_sSFR', 'Mstar', 'environment',
          'gama_in_pairs', 'gama_close_pair_100kpc', 
          'gama_close_pair_50kpc']].info())

# Confirm boolean flags are actual booleans not strings
print(df['gama_in_pairs'].dtype)
print(df['gama_in_pairs'].value_counts())