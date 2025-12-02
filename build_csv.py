from simple_connection import get_db
import pandas as pd

db = get_db()

data = db.query(
    db.Sources.c.source,
    db.Sources.c.ra,
    db.Sources.c.dec,
    db.Sources.c.reference,
    db.Photometry.c.band,
    db.Photometry.c.magnitude,
    db.Photometry.c.magnitude_error,
    db.SpectralTypes.c.spectral_type_string,
    db.SpectralTypes.c.spectral_type_code,
    db.SpectralTypes.c.reference.label("spectral_type_reference"),
).join (db.Photometry, db.Sources.c.source == db.Photometry.c.source).join (db.SpectralTypes, db.Sources.c.source == db.SpectralTypes.c.source)

df = data.pandas()

# Keep only L/T/Y spectral types
df_lty = df[df["spectral_type_string"].str.startswith(("L","T","Y"))]

df_lty.to_csv("lty_photometry_raw.csv", index=False)

#data cleaning... dropping unnecessary/insufficient bands and sources
bands = ["2MASS.J",
    "2MASS.H",
    "2MASS.Ks",
    "WISE.W1",
    "WISE.W2"]
#Drop unnecessary bands
df_lty = df_lty[df_lty["band"].isin(bands)]

# Count number of available bands
band_counts = (
    df_lty.groupby(["source", "spectral_type_string", "spectral_type_code"])["band"]
          .nunique()
          .reset_index(name="n_available")
)
df_lty = df_lty.merge(
    band_counts,
    on=["source", "spectral_type_string", "spectral_type_code"],
    how="left"
)

#Keep sources with at least 2 available bands
df_lty = df_lty[df_lty["n_available"] >= 2]

# Pivot to wide format
wide = df_lty.pivot_table(
    index=["source", "spectral_type_string", "spectral_type_code"],
    columns="band",
    values="magnitude",
    aggfunc="min"  # if multiple measurements exist, pick one
).reset_index()

# Flatten column names
wide.columns.name = None

#for higher accuracy, but smaller sample size, keep sources with at least 3 bands
df_lty_highacc = df_lty[df_lty["n_available"] >= 3]
wide_highacc = df_lty_highacc.pivot_table(
    index=["source", "spectral_type_string", "spectral_type_code"],
    columns="band",
    values="magnitude",
    aggfunc="min"  # if multiple measurements exist, pick one
).reset_index()

# Flatten column names
wide_highacc.columns.name = None

#add color features
def add_features(wide_df: pd.DataFrame) -> pd.DataFrame:
    new_df = wide_df.copy()

    # Rename bands to simpler names
    rename_map = {
        "2MASS.J": "J",
        "2MASS.H": "H",
        "2MASS.Ks": "Ks",
        "WISE.W1": "W1",
        "WISE.W2": "W2",
    }
    new_df = new_df.rename(columns=rename_map)

    #create color features
    if "J" in new_df.columns and "H" in new_df.columns:
        new_df["JH"] = new_df["J"] - new_df["H"]
    else:
        new_df["JH"] = pd.NA

    if "H" in new_df.columns and "Ks" in new_df.columns:
        new_df["HK"] = new_df["H"] - new_df["Ks"]
    else:
        new_df["HK"] = pd.NA

    if "J" in new_df.columns and "Ks" in new_df.columns:
        new_df["JK"] = new_df["J"] - new_df["Ks"]
    else:
        new_df["JK"] = pd.NA

    if "W1" in new_df.columns and "W2" in new_df.columns:
        new_df["W1W2"] = new_df["W1"] - new_df["W2"]
    else:
        new_df["W1W2"] = pd.NA

    if "Ks" in new_df.columns and "W1" in new_df.columns:
        new_df["KW1"] = new_df["Ks"] - new_df["W1"]
    else:
        new_df["KW1"] = pd.NA

    if "Ks" in new_df.columns and "W2" in new_df.columns:
        new_df["KW2"] = new_df["Ks"] - new_df["W2"]
    else:
        new_df["KW2"] = pd.NA

    # reorder columns
    cols_order = [
        "source",
        "spectral_type_string",
        "spectral_type_code",
        "J", "H", "Ks", "W1", "W2",
        "JH", "HK", "JK", "W1W2", "KW1", "KW2",
    ]
    # Keep only columns that actually exist
    cols_order = [c for c in cols_order if c in new_df.columns]
    new_df = new_df[cols_order]

    return new_df

# Apply to both tables
final_base = add_features(wide)
final_highacc = add_features(wide_highacc)

wide.to_csv("lty_photometry_wide.csv", index=False)
wide_highacc.to_csv("lty_photometry_wide_highacc.csv", index=False)

final_base.to_csv("lty_final.csv", index=False)
final_highacc.to_csv("lty_final_highacc.csv", index=False)