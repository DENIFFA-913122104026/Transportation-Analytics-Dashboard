
# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The provided table name 'Trip_Anomaly_Results'.*"
)

from sqlalchemy import create_engine, URL
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ------------------------------------------------------------
# 2. MYSQL CONNECTION
# ------------------------------------------------------------

connection_url = URL.create(
    "mysql+pymysql",
    username="root",
    password="Rebu@deni1128",
    host="127.0.0.1",
    port=3306,
    database="smart_transportation"
)

engine = create_engine(connection_url)

print("MySQL connection successful!")


# ------------------------------------------------------------
# 3. LOAD TRANSPORTATION DATA
# ------------------------------------------------------------

df = pd.read_sql(
    "SELECT * FROM Transportation_Analytics",
    engine
)

print("\nDataset loaded successfully")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 4. BASIC DATA CHECK
# ------------------------------------------------------------

print("\n========== DATA CHECK ==========")

print("Number of rows:", len(df))

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate Trip IDs:",
    df["Trip_ID"].duplicated().sum()
)


# ------------------------------------------------------------
# 5. BASIC TRANSPORTATION KPIs
# ------------------------------------------------------------

print("\n========== TRANSPORTATION KPIs ==========")

print("Total Trips:", len(df))

print(
    "Total Revenue:",
    round(df["Revenue"].sum(), 2)
)

print(
    "Total Passengers:",
    df["Passenger_Count"].sum()
)

print(
    "Average Delay:",
    round(df["Delay_Min"].mean(), 2),
    "minutes"
)

print(
    "Average Occupancy:",
    round(df["Occupancy_Rate"].mean(), 2),
    "%"
)


# ------------------------------------------------------------
# 6. TRIP STATUS
# ------------------------------------------------------------

print("\n========== TRIP STATUS ==========")

print(
    df["Trip_Status"].value_counts()
)


# ------------------------------------------------------------
# 7. VEHICLE TYPE ANALYSIS
# ------------------------------------------------------------

print("\n========== REVENUE BY VEHICLE TYPE ==========")

vehicle_revenue = (
    df.groupby("Vehicle_Type")
      .agg(
          Trips=("Trip_ID", "count"),
          Revenue=("Revenue", "sum")
      )
      .reset_index()
)

print(vehicle_revenue)


# ------------------------------------------------------------
# 8. PEAK VS NON-PEAK ANALYSIS
# ------------------------------------------------------------

print("\n========== PEAK VS NON-PEAK ==========")

peak_analysis = (
    df.groupby("Peak_Hour")
      .agg(
          Trips=("Trip_ID", "count"),
          Average_Delay=("Delay_Min", "mean")
      )
      .reset_index()
)

peak_analysis["Average_Delay"] = (
    peak_analysis["Average_Delay"].round(2)
)

print(peak_analysis)


# ============================================================
# 9. MACHINE LEARNING
#    ISOLATION FOREST
# ============================================================

print("\n==========================================")
print("MACHINE LEARNING - ISOLATION FOREST")
print("==========================================")


# ------------------------------------------------------------
# 10. SELECT ML FEATURES
# ------------------------------------------------------------

ml_features = [
    "Delay_Min",
    "Actual_Duration_Min",
    "Duration_Variance",
    "Passenger_Count",
    "Occupancy_Rate",
    "Revenue",
    "Revenue_Per_KM",
    "Distance_KM"
]

print("\nML Features:")
for feature in ml_features:
    print("-", feature)


# ------------------------------------------------------------
# 11. CREATE ML DATASET
# ------------------------------------------------------------

X = df[ml_features].copy()

print("\nML input shape:", X.shape)


# ------------------------------------------------------------
# 12. HANDLE INFINITE VALUES
# ------------------------------------------------------------

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ------------------------------------------------------------
# 13. HANDLE MISSING VALUES
# ------------------------------------------------------------

X = X.fillna(
    X.median(numeric_only=True)
)


# ------------------------------------------------------------
# 14. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed")


# ------------------------------------------------------------
# 15. CREATE ISOLATION FOREST MODEL
# ------------------------------------------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# 16. TRAIN MODEL AND PREDICT ANOMALIES
# ------------------------------------------------------------

df["Anomaly_Flag"] = model.fit_predict(X_scaled)


# ------------------------------------------------------------
# 17. CALCULATE ANOMALY SCORE
# ------------------------------------------------------------

df["Anomaly_Score"] = (
    model.decision_function(X_scaled)
)


# ------------------------------------------------------------
# 18. CREATE READABLE ANOMALY STATUS
# ------------------------------------------------------------

df["Anomaly_Status"] = (
    df["Anomaly_Flag"]
    .map({
        1: "Normal",
        -1: "Anomaly"
    })
)


# ------------------------------------------------------------
# 19. ML RESULTS
# ------------------------------------------------------------

print("\n========== ISOLATION FOREST RESULTS ==========")

normal_count = (
    (df["Anomaly_Flag"] == 1).sum()
)

anomaly_count = (
    (df["Anomaly_Flag"] == -1).sum()
)

anomaly_percentage = (
    anomaly_count / len(df) * 100
)

print("Total Trips:", len(df))
print("Normal Trips:", normal_count)
print("Anomalous Trips:", anomaly_count)
print(
    "Anomaly Percentage:",
    round(anomaly_percentage, 2),
    "%"
)


# ------------------------------------------------------------
# 20. SHOW TOP 10 ANOMALOUS TRIPS
# ------------------------------------------------------------

print("\n========== TOP 10 ANOMALOUS TRIPS ==========")

top_anomalies = (
    df[df["Anomaly_Flag"] == -1]
    .sort_values("Anomaly_Score")
    .head(10)
)

columns_to_show = [
    "Trip_ID",
    "Route_ID",
    "Vehicle_ID",
    "Delay_Min",
    "Actual_Duration_Min",
    "Duration_Variance",
    "Occupancy_Rate",
    "Revenue",
    "Anomaly_Score"
]

print(
    top_anomalies[columns_to_show]
)


# ============================================================
# 21. ANOMALY RATE BY ROUTE
# ============================================================

print("\n========== ANOMALY RATE BY ROUTE ==========")

route_anomaly = (
    df.groupby("Route_ID")
      .agg(
          Total_Trips=("Trip_ID", "count"),
          Anomalies=("Anomaly_Flag",
                     lambda x: (x == -1).sum())
      )
      .reset_index()
)

route_anomaly["Anomaly_Rate"] = (
    route_anomaly["Anomalies"]
    / route_anomaly["Total_Trips"]
    * 100
)

route_anomaly = route_anomaly.sort_values(
    "Anomaly_Rate",
    ascending=False
)

print(
    route_anomaly.head(10)
)


# ============================================================
# 22. ANOMALY RATE BY VEHICLE
# ============================================================

print("\n========== ANOMALY RATE BY VEHICLE ==========")

vehicle_anomaly = (
    df.groupby("Vehicle_ID")
      .agg(
          Total_Trips=("Trip_ID", "count"),
          Anomalies=("Anomaly_Flag",
                     lambda x: (x == -1).sum())
      )
      .reset_index()
)

vehicle_anomaly["Anomaly_Rate"] = (
    vehicle_anomaly["Anomalies"]
    / vehicle_anomaly["Total_Trips"]
    * 100
)

vehicle_anomaly = vehicle_anomaly.sort_values(
    "Anomaly_Rate",
    ascending=False
)

print(
    vehicle_anomaly.head(10)
)


# ============================================================
# 23. ANOMALY RATE BY WEATHER
# ============================================================

print("\n========== ANOMALY RATE BY WEATHER ==========")

weather_anomaly = (
    df.groupby("Weather")
      .agg(
          Total_Trips=("Trip_ID", "count"),
          Anomalies=("Anomaly_Flag",
                     lambda x: (x == -1).sum())
      )
      .reset_index()
)

weather_anomaly["Anomaly_Rate"] = (
    weather_anomaly["Anomalies"]
    / weather_anomaly["Total_Trips"]
    * 100
)

print(weather_anomaly)


# ============================================================
# 24. ANOMALY RATE BY PEAK HOUR
# ============================================================

print("\n========== ANOMALY RATE BY PEAK HOUR ==========")

peak_anomaly = (
    df.groupby("Peak_Hour")
      .agg(
          Total_Trips=("Trip_ID", "count"),
          Anomalies=("Anomaly_Flag",
                     lambda x: (x == -1).sum())
      )
      .reset_index()
)

peak_anomaly["Anomaly_Rate"] = (
    peak_anomaly["Anomalies"]
    / peak_anomaly["Total_Trips"]
    * 100
)

print(peak_anomaly)


# ============================================================
# 25. CHART 1
#    NORMAL VS ANOMALOUS TRIPS - PIE CHART
# ============================================================

status_counts = (
    df["Anomaly_Status"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

plt.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Normal vs Anomalous Transportation Trips"
)

plt.tight_layout()

plt.show()


# ============================================================
# 26. CHART 2
#    TOP 10 ROUTES BY ANOMALY RATE
# ============================================================

top_routes = route_anomaly.head(10)

plt.figure(figsize=(9, 5))

plt.barh(
    top_routes["Route_ID"],
    top_routes["Anomaly_Rate"]
)

plt.title(
    "Top 10 Routes by Anomaly Rate"
)

plt.xlabel("Anomaly Rate (%)")

plt.ylabel("Route ID")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 27. CHART 3
#    TOP 10 VEHICLES BY ANOMALY RATE
# ============================================================

top_vehicles = vehicle_anomaly.head(10)

plt.figure(figsize=(9, 5))

plt.bar(
    top_vehicles["Vehicle_ID"],
    top_vehicles["Anomaly_Rate"]
)

plt.title(
    "Top 10 Vehicles by Anomaly Rate"
)

plt.xlabel("Vehicle ID")

plt.ylabel("Anomaly Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 28. CHART 4
#    DELAY VS ANOMALY SCORE
# ============================================================

plt.figure(figsize=(9, 5))

plt.scatter(
    df["Delay_Min"],
    df["Anomaly_Score"],
    alpha=0.5
)

plt.title(
    "Delay vs Isolation Forest Anomaly Score"
)

plt.xlabel("Delay (Minutes)")

plt.ylabel("Anomaly Score")

plt.tight_layout()

plt.show()


# ============================================================
# 29. SAVE COMPLETE DATASET
# ============================================================

df.to_csv(
    "Transportation_Analytics_With_Anomalies.csv",
    index=False
)

print(
    "\nSaved:"
    "\nTransportation_Analytics_With_Anomalies.csv"
)


# ============================================================
# 30. SAVE ONLY ML RESULTS
# ============================================================

ml_results = df[
    [
        "Trip_ID",
        "Anomaly_Flag",
        "Anomaly_Score",
        "Anomaly_Status"
    ]
].copy()

ml_results.to_csv(
    "Transportation_Anomaly_Results.csv",
    index=False
)

print(
    "Saved:"
    "\nTransportation_Anomaly_Results.csv"
)


# ============================================================
# 31. SAVE ML RESULTS TO MYSQL
# ============================================================

print("\n========== SAVING ML RESULTS TO MYSQL ==========")

# Drop the table first.
# This avoids the SQLAlchemy reflection problem.

with engine.begin() as connection:

    connection.exec_driver_sql(
        "DROP TABLE IF EXISTS Trip_Anomaly_Results"
    )


# Create the table again using pandas

ml_results.to_sql(
    "Trip_Anomaly_Results",
    con=engine,
    if_exists="append",
    index=False
)

print(
    "Trip_Anomaly_Results table created successfully!"
)


# ============================================================
# 32. VERIFY MYSQL TABLE
# ============================================================

check_result = pd.read_sql(
    "SELECT COUNT(*) AS Total_Rows "
    "FROM Trip_Anomaly_Results",
    engine
)

print(
    "\nRows saved in MySQL:",
    check_result.iloc[0]["Total_Rows"]
)


# ============================================================
# 33. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("MACHINE LEARNING PROCESS COMPLETED")
print("==========================================")

print("Algorithm      : Isolation Forest")
print("Learning Type  : Unsupervised ML")
print("Total Trips    :", len(df))
print("Normal Trips   :", normal_count)
print("Anomalies      :", anomaly_count)
print(
    "Anomaly Rate   :",
    round(anomaly_percentage, 2),
    "%"
)

print("\nOutput files:")
print("1. Transportation_Analytics_With_Anomalies.csv")
print("2. Transportation_Anomaly_Results.csv")

print("\nMySQL table:")
print("Trip_Anomaly_Results")

print("\nPower BI can now use the anomaly results.")
