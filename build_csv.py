from simple_connection import get_db


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

print(df_lty.head())
df_lty.to_csv("lty_photometry_raw.csv", index=False)

# Pivot to wide format
wide = df_lty.pivot_table(
    index=["source", "spectral_type_string"],
    columns="band",
    values="magnitude",
    aggfunc="min"  # if multiple measurements exist, pick one
).reset_index()

# Flatten column names
wide.columns.name = None

wide.to_csv("lty_photometry_wide.csv", index=False)