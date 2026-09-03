import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

CSV_PATH = "outputs/traffic_history.csv"


# ==========================================
# LOAD DATASET
# ==========================================

print("AI Traffic Intelligence - Dataset Analysis")
print()
print("Loading traffic dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully!")


# ==========================================
# BASIC INFORMATION
# ==========================================

print()
print("================================")
print("       DATASET INFORMATION")
print("================================")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print()
print("Column names:")

for column in df.columns:
    print(f"- {column}")


# ==========================================
# FIRST 10 ROWS
# ==========================================

print()
print("================================")
print("        FIRST 10 ROWS")
print("================================")

print(df.head(10).to_string(index=False))


# ==========================================
# DATA TYPES
# ==========================================

print()
print("================================")
print("          DATA TYPES")
print("================================")

print(df.dtypes)


# ==========================================
# MISSING VALUES
# ==========================================

print()
print("================================")
print("        MISSING VALUES")
print("================================")

missing_values = df.isnull().sum()

print(missing_values)


# ==========================================
# NUMERICAL STATISTICS
# ==========================================

print()
print("================================")
print("    NUMERICAL STATISTICS")
print("================================")

print(
    df.describe().to_string()
)


# ==========================================
# VEHICLE COUNT ANALYSIS
# ==========================================

print()
print("================================")
print("      VEHICLE COUNT")
print("================================")

print(
    f"Minimum: {df['vehicle_count'].min()}"
)

print(
    f"Maximum: {df['vehicle_count'].max()}"
)

print(
    f"Average: {df['vehicle_count'].mean():.2f}"
)


# ==========================================
# SPEED ANALYSIS
# ==========================================

print()
print("================================")
print("       SPEED ANALYSIS")
print("================================")

print(
    f"Minimum: "
    f"{df['average_speed_kmh'].min():.2f} km/h"
)

print(
    f"Maximum: "
    f"{df['average_speed_kmh'].max():.2f} km/h"
)

print(
    f"Average: "
    f"{df['average_speed_kmh'].mean():.2f} km/h"
)


# ==========================================
# DENSITY DISTRIBUTION
# ==========================================

print()
print("================================")
print("     DENSITY DISTRIBUTION")
print("================================")

print(
    df["density"].value_counts()
)


# ==========================================
# CONGESTION DISTRIBUTION
# ==========================================

print()
print("================================")
print("   CONGESTION DISTRIBUTION")
print("================================")

print(
    df["congestion_level"].value_counts()
)


# ==========================================
# CONGESTION SCORE
# ==========================================

print()
print("================================")
print("     CONGESTION SCORE")
print("================================")

print(
    f"Minimum: "
    f"{df['congestion_score'].min()}"
)

print(
    f"Maximum: "
    f"{df['congestion_score'].max()}"
)

print(
    f"Average: "
    f"{df['congestion_score'].mean():.2f}"
)


# ==========================================
# DUPLICATE ROWS
# ==========================================

print()
print("================================")
print("      DUPLICATE ROWS")
print("================================")

print(
    f"Duplicates: {df.duplicated().sum()}"
)


# ==========================================
# FINAL
# ==========================================

print()
print("================================")
print("     DATASET ANALYSIS DONE")
print("================================")

print("Your traffic dataset is ready")
print("for the next ML preparation stage.")

print("================================")