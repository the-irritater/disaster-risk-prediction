import numpy as np
import pandas as pd
from scipy.stats import gamma, poisson

class DisasterDataGenerator:
    """
    Generates realistic synthetic district-month panel datasets for disaster risk modeling.

    Design Philosophy:
    ------------------
    This generator follows a structural simulation approach where physical and
    socio-economic relationships are encoded via parameterised statistical models.
    The causal structure is:

        Geography → Environmental Variables → Hazard Probabilities → Disaster Occurrence → Impacts

    Distribution Choices:
    ---------------------
    - Environmental variables: Gaussian noise around physically motivated baselines
      (seasonal patterns, elevation effects, coastal proximity), with AR(1) persistence
      (coefficient ~0.55) reflecting month-to-month climate autocorrelation.
    - Hazard probabilities: Logistic link functions with domain-specific coefficients,
      calibrated to produce ~15% overall disaster prevalence (consistent with DesInventar
      India database, 2000–2020).
    - Deaths: Poisson distribution (count data, rare events).
    - Economic losses: Gamma distribution (positive, heavy-tailed, consistent with
      EM-DAT loss distributions).
    - Demographics: Uniform distributions calibrated to Indian Census 2011 ranges.

    See reports/simulation_design.md for full documentation.
    """
    def __init__(self, num_districts=100, num_years=11, start_year=2015, seed=42):
        self.num_districts = num_districts
        self.num_years = num_years
        self.start_year = start_year
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def generate_spatial_grid(self):
        """
        Creates a 10x10 grid of districts with coordinates and geographic characteristics.
        """
        districts = []
        # Assign districts to a grid (1 to 10 for X and Y)
        for i in range(self.num_districts):
            d_id = f"D_{i+1:03d}"
            grid_x = (i % 10) + 1
            grid_y = (i // 10) + 1
            
            # Map grid to Latitude & Longitude
            lat = 20.0 + grid_y * 0.5
            lon = 75.0 + grid_x * 0.5
            
            # Geographic attributes
            dist_coast = grid_x * 45.0 + self.rng.uniform(-10, 10)  # West is coast (low grid_x)
            dist_river = abs(grid_x - grid_y) * 30.0 + 5.0 + self.rng.uniform(-2, 2)  # Diagonal river
            elevation = (11 - grid_y) * 150.0 + grid_x * 50.0 + self.rng.uniform(10, 50)  # North-East is mountainous
            slope = (elevation / 1500.0) * 25.0 + self.rng.uniform(0, 5)
            seismic_idx = 0.1 + (11 - grid_y) * 0.08 + (self.rng.uniform(0, 0.1) if grid_y >= 7 else 0)  # North is seismically active
            
            # Assign state and region
            state_idx = (i // 20) + 1
            state_name = f"State_{chr(64 + state_idx)}"
            regions = ["North", "Central", "East", "South", "West"]
            region_name = regions[state_idx - 1]
            
            # Demographics and Exposure (stable district profiles)
            pop = int(self.rng.uniform(200000, 2500000))
            area = self.rng.uniform(500, 3000)  # sq km
            pop_density = pop / area
            urban_rate = self.rng.uniform(0.1, 0.85)
            households = int(pop / self.rng.uniform(4.0, 5.5))
            infra_density = urban_rate * 80.0 + self.rng.uniform(5, 15)
            agri_pct = self.rng.uniform(0.2, 0.7)
            ind_pct = self.rng.uniform(0.05, 0.25) * urban_rate
            
            # Vulnerability (stable district profiles)
            poverty = self.rng.uniform(0.08, 0.45)
            elderly = self.rng.uniform(0.06, 0.18)
            child = self.rng.uniform(0.12, 0.25)
            literacy = self.rng.uniform(0.55, 0.95)
            housing_idx = 100 - (poverty * 100 + self.rng.uniform(10, 30))  # Lower housing quality where poor
            housing_idx = np.clip(housing_idx, 20, 95)
            healthcare_idx = self.rng.uniform(30, 90) - (poverty * 30)
            healthcare_idx = np.clip(healthcare_idx, 20, 95)
            road_idx = self.rng.uniform(30, 95) - (11 - grid_y) * 5  # Lower in mountains
            road_idx = np.clip(road_idx, 20, 95)
            comm_idx = self.rng.uniform(40, 95) - (poverty * 20)
            comm_idx = np.clip(comm_idx, 20, 95)
            
            # Raw Preparedness Resource Counts
            shelters = int((pop / 100000) * self.rng.uniform(2, 8)) + int(self.rng.uniform(1, 4))
            hospitals = int((pop / 100000) * self.rng.uniform(1, 5)) + 1
            rescue_teams = int((pop / 100000) * self.rng.uniform(0.5, 3)) + 1
            ews = 1 if (infra_density > 40 or self.rng.uniform(0, 1) > 0.4) else 0
            evac_plan = 1 if (ews == 1 and self.rng.uniform(0, 1) > 0.3) else 0
            resp_time = self.rng.uniform(20, 120) + (100 - road_idx) * 0.5
            resp_time = np.clip(resp_time, 15, 180)
            
            districts.append({
                "District": d_id,
                "State": state_name,
                "Region": region_name,
                "Latitude": round(lat, 4),
                "Longitude": round(lon, 4),
                "Distance_From_Coast_km": round(dist_coast, 2),
                "Distance_From_River_km": round(dist_river, 2),
                "Elevation_Metres": round(elevation, 1),
                "Slope_Degrees": round(slope, 2),
                "Seismic_Activity_Index": round(seismic_idx, 3),
                "Population": pop,
                "Area_SqKm": round(area, 2),
                "Population_Density": round(pop_density, 2),
                "Urbanisation_Rate": round(urban_rate, 4),
                "Number_of_Households": households,
                "Infrastructure_Density": round(infra_density, 2),
                "Agricultural_Land_Percentage": round(agri_pct, 4),
                "Industrial_Area_Percentage": round(ind_pct, 4),
                "Poverty_Rate": round(poverty, 4),
                "Elderly_Population_Percentage": round(elderly, 4),
                "Child_Population_Percentage": round(child, 4),
                "Literacy_Rate": round(literacy, 4),
                "Housing_Quality_Index": round(housing_idx, 2),
                "Healthcare_Access_Index": round(healthcare_idx, 2),
                "Road_Access_Index": round(road_idx, 2),
                "Communication_Access_Index": round(comm_idx, 2),
                "Raw_Shelter_Count": shelters,
                "Raw_Hospital_Count": hospitals,
                "Raw_Rescue_Team_Count": rescue_teams,
                "Early_Warning_System": ews,
                "Evacuation_Plan_Available": evac_plan,
                "Emergency_Response_Time_Minutes": round(resp_time, 1)
            })
            
        return pd.DataFrame(districts)
        
    def generate_panel_data(self):
        """
        Generates district-month panel data combining spatial baseline properties
        with seasonal dynamics, climate trends, and randomized physical shocks.

        Statistical Design Notes:
        -------------------------
        1. SEASONAL PATTERNS: Rainfall follows monsoon climatology (peak Jun–Sep),
           temperature follows summer/winter cycle. Reference: IMD climatological normals.

        2. CLIMATE TREND: Linear warming of +0.015 units/year, representing gradual
           climate change over the 2015–2025 period. This is a simplification of
           real non-linear climate trends.

        3. AUTOREGRESSIVE PERSISTENCE: AR(1) coefficient ~0.55 for environmental
           variables. This produces realistic month-to-month correlation without
           creating excessively long memory. Reference: empirical autocorrelation
           in monthly climate observations is typically 0.3–0.7.

        4. HAZARD PROBABILITIES: Logistic functions P(hazard) = 1/(1 + exp(-z)).
           Intercepts are calibrated so that the overall disaster prevalence is ~15%,
           consistent with event-month rates in DesInventar India (2000–2020).

        5. IMPACT GENERATION: Conditional on disaster_occurred = 1 only.
           Deaths ~ Poisson (count data), losses ~ Gamma (heavy-tailed positive).
           This ensures zero data leakage from post-event variables to pre-event features.
        """
        df_districts = self.generate_spatial_grid()
        records = []
        previous_states = {} # d_id -> dict for autoregressive persistence
        
        for year in range(self.start_year, self.start_year + self.num_years):
            climate_trend_factor = (year - self.start_year) * 0.015  # Gradual heating/anomaly trend
            
            for month in range(1, 12 + 1):
                # Monthly seasonal effects
                # Rainfall peaks in monsoon (June=6, July=7, August=8, September=9)
                monsoon_months = [6, 7, 8, 9]
                if month in monsoon_months:
                    base_seasonal_rain = 350.0 + (month == 7 or month == 8) * 150.0
                else:
                    base_seasonal_rain = 25.0 + (month == 5 or month == 10) * 30.0
                
                # Temperature peaks in summer (April=4, May=5, June=6)
                summer_months = [4, 5, 6]
                if month in summer_months:
                    base_seasonal_temp = 36.0 + (month == 5) * 3.0
                elif month in [12, 1, 2]:
                    base_seasonal_temp = 18.0 - (month == 1) * 3.0
                else:
                    base_seasonal_temp = 27.0
                
                for idx, dist in df_districts.iterrows():
                    d_id = dist["District"]
                    
                    # 1. Environmental Variables with Seasonality, Climate Trend, Shocks, and Autoregression
                    # Rainfall: higher near coast, mountains
                    coastal_boost = max(0, 300.0 - dist["Distance_From_Coast_km"] * 0.5)
                    elev_boost = dist["Elevation_Metres"] * 0.05
                    mean_rain = (base_seasonal_rain + coastal_boost + elev_boost) * (1.0 + climate_trend_factor * 0.2)
                    
                    # Autoregressive persistence
                    has_prev = d_id in previous_states
                    
                    if has_prev:
                        prev = previous_states[d_id]
                        # Persistence on anomaly:
                        rain_anomaly = 0.55 * prev["rain_anomaly"] + self.rng.normal(0, 0.5)
                        # Back-calculate rain based on rain_anomaly
                        rain = max(0, mean_rain + rain_anomaly * (mean_rain * 0.3 + 5.0))
                    else:
                        rain = max(0, mean_rain + self.rng.normal(0, mean_rain * 0.3))
                        rain_anomaly = (rain - mean_rain) / (mean_rain * 0.3 + 5.0)
                    
                    # Temperature
                    elev_cooling = dist["Elevation_Metres"] * 0.006  # Lapse rate
                    mean_temp = base_seasonal_temp - elev_cooling + climate_trend_factor
                    
                    if has_prev:
                        temp_anomaly = 0.55 * prev["temp_anomaly"] + self.rng.normal(0, 0.8)
                        temp = mean_temp + temp_anomaly
                    else:
                        temp = mean_temp + self.rng.normal(0, 1.5)
                        temp_anomaly = temp - (base_seasonal_temp - elev_cooling)
                    
                    # Wind speed: higher near coast and in storm seasons (May, Oct, Nov)
                    coastal_wind = max(0, 45.0 - dist["Distance_From_Coast_km"] * 0.1)
                    storm_factor = 1.8 if month in [5, 10, 11] else 1.0
                    base_wind = (12.0 + coastal_wind) * storm_factor
                    
                    if has_prev:
                        wind_speed = 0.35 * prev["wind_speed"] + 0.65 * (base_wind + self.rng.uniform(0, 10))
                    else:
                        wind_speed = base_wind + self.rng.uniform(0, 10)
                    
                    # River Level: rises with rainfall and rainfall anomalies
                    river_base = 2.0 + (1000 - dist["Elevation_Metres"]) * 0.001
                    target_river = river_base + max(0, rain_anomaly) * 2.5 + (rain / 200.0)
                    
                    if has_prev:
                        river_level = max(0.5, 0.55 * prev["river_level"] + 0.45 * target_river + self.rng.normal(0, 0.2))
                    else:
                        river_level = max(0.5, target_river + self.rng.normal(0, 0.3))
                    
                    # Soil moisture: low in summer, high in monsoon
                    target_soil = 0.7 * (rain / 400.0) - 0.05 * temp_anomaly + 0.3
                    if has_prev:
                        soil_moisture = np.clip(0.55 * prev["soil_moisture"] + 0.45 * target_soil + self.rng.normal(0, 0.05), 0.05, 0.95)
                    else:
                        soil_moisture = np.clip(target_soil + self.rng.normal(0, 0.08), 0.05, 0.95)
                    
                    # Vegetation Index (NDVI): lags rain, lower where dry/urban
                    target_veg = 0.4 + 0.3 * soil_moisture - 0.1 * dist["Urbanisation_Rate"]
                    if has_prev:
                        veg_idx = np.clip(0.60 * prev["veg_idx"] + 0.40 * target_veg + self.rng.normal(0, 0.03), 0.1, 0.9)
                    else:
                        veg_idx = np.clip(target_veg + self.rng.normal(0, 0.05), 0.1, 0.9)
                    
                    # Drought Index: high when soil moisture is very low
                    target_drought = (1.0 - soil_moisture) * 100.0 + max(0, temp_anomaly) * 5.0
                    if has_prev:
                        drought_idx = np.clip(0.55 * prev["drought_idx"] + 0.45 * target_drought + self.rng.normal(0, 2), 0, 100)
                    else:
                        drought_idx = np.clip(target_drought + self.rng.normal(0, 3), 0, 100)
                    
                    # Save current state for next month's persistence
                    previous_states[d_id] = {
                        "rain_anomaly": rain_anomaly,
                        "temp_anomaly": temp_anomaly,
                        "wind_speed": wind_speed,
                        "river_level": river_level,
                        "soil_moisture": soil_moisture,
                        "veg_idx": veg_idx,
                        "drought_idx": drought_idx
                    }
                    
                    # Seismic Activity: occasional random earthquake shock
                    seismic_base = dist["Seismic_Activity_Index"]
                    seismic_shock = self.rng.uniform(0, 3.5) if self.rng.uniform(0, 1) > 0.985 else 0.0
                    seismic_val = seismic_base + seismic_shock
                    
                    # Preparedness Index (pre-calculated index based on district baseline variables)
                    prep_score = (
                        dist["Early_Warning_System"] * 25.0 +
                        dist["Evacuation_Plan_Available"] * 20.0 +
                        (100.0 - dist["Emergency_Response_Time_Minutes"]) * 0.25 +
                        (dist["Healthcare_Access_Index"]) * 0.30
                    )
                    prep_score = np.clip(prep_score + self.rng.normal(0, 2), 10, 100)
                    
                    # 2. Multi-Hazard Latent Occurrence Process
                    haz_probs = {}
                    
                    # Flood probability: high rain anomaly, low elevation, close to river
                    z_flood = -5.8 + 2.8 * rain_anomaly - 0.008 * dist["Elevation_Metres"] - 0.015 * dist["Distance_From_River_km"] + 0.5 * river_level
                    haz_probs["Flood"] = 1.0 / (1.0 + np.exp(-z_flood))
                    
                    # Cyclone probability: high wind speed, coastal proximity, storm months
                    z_cyclone = -7.5 + 0.12 * wind_speed - 0.015 * dist["Distance_From_Coast_km"]
                    haz_probs["Cyclone"] = 1.0 / (1.0 + np.exp(-z_cyclone)) if month in [5, 10, 11] else 0.0
                    
                    # Landslide probability: high rain anomaly, steep slope, low vegetation
                    z_landslide = -7.2 + 2.2 * rain_anomaly + 0.14 * dist["Slope_Degrees"] - 3.5 * veg_idx
                    haz_probs["Landslide"] = 1.0 / (1.0 + np.exp(-z_landslide)) if dist["Elevation_Metres"] > 400 else 0.0
                    
                    # Drought probability: high temperature anomaly, low rain, low soil moisture
                    z_drought = -5.5 + 1.8 * temp_anomaly - 2.5 * rain_anomaly - 4.5 * soil_moisture
                    haz_probs["Drought"] = 1.0 / (1.0 + np.exp(-z_drought)) if month in [3, 4, 5, 10, 11] else 0.0
                    
                    # Earthquake probability: high seismic index (shocks)
                    z_earthquake = -9.0 + 3.2 * seismic_val
                    haz_probs["Earthquake"] = 1.0 / (1.0 + np.exp(-z_earthquake))
                    
                    # Heatwave probability: high temp anomaly, urban density, summer months
                    z_heatwave = -5.2 + 3.0 * temp_anomaly + 0.002 * dist["Population_Density"]
                    haz_probs["Heatwave"] = 1.0 / (1.0 + np.exp(-z_heatwave)) if month in [4, 5, 6] else 0.0
                    
                    # Wildfire probability: high temperature, dry soil, summer/autumn
                    z_wildfire = -6.5 + 2.2 * temp_anomaly - 3.8 * soil_moisture + 1.2 * veg_idx
                    haz_probs["Wildfire"] = 1.0 / (1.0 + np.exp(-z_wildfire)) if month in [3, 4, 5, 10, 11] and dist["Urbanisation_Rate"] < 0.6 else 0.0
                    
                    # Severe Storm probability: wind, rain anomaly
                    z_storm = -6.2 + 0.06 * wind_speed + 1.2 * rain_anomaly
                    haz_probs["Severe Storm"] = 1.0 / (1.0 + np.exp(-z_storm))
                    
                    # Determine event occurrence (independent latent processes)
                    active_hazards = []
                    for haz_type, p_val in haz_probs.items():
                        if p_val > 0.005:  # Minimum threshold
                            # Bernoulli trial with noise
                            if self.rng.uniform(0, 1) < p_val:
                                active_hazards.append((haz_type, p_val))
                    
                    # Resolve to a single dominant disaster type per month
                    if len(active_hazards) > 0:
                        # Sort by probability, highest first
                        active_hazards.sort(key=lambda x: x[1], reverse=True)
                        disaster_type = active_hazards[0][0]
                        disaster_occurred = 1
                        # Severity scale (1 to 10) based on hazard probability and environmental trigger
                        hazard_severity = np.clip(active_hazards[0][1] * 12.0 + self.rng.uniform(1.0, 4.0), 1.0, 10.0)
                        disaster_duration = max(1, int(hazard_severity * 2.0 + self.rng.uniform(-2, 3)))
                        disaster_magnitude = hazard_severity * self.rng.uniform(0.8, 1.3)
                    else:
                        disaster_type = "No Disaster"
                        disaster_occurred = 0
                        hazard_severity = 0.0
                        disaster_duration = 0
                        disaster_magnitude = 0.0
                    
                    # Compound event: two or more hazard types activated simultaneously
                    compound_event = 1 if len(active_hazards) >= 2 else 0
                    
                    # Add record
                    rec = {
                        "Record_ID": f"REC_{year}_{month:02d}_{d_id}",
                        "District": d_id,
                        "State": dist["State"],
                        "Region": dist["Region"],
                        "Latitude": dist["Latitude"],
                        "Longitude": dist["Longitude"],
                        "Year": year,
                        "Month": month,
                        "Event_Date": f"{year}-{month:02d}-01",
                        "Disaster_Type": disaster_type,
                        "Disaster_Occurred": disaster_occurred,
                        "Disaster_Duration_Days": disaster_duration,
                        "Disaster_Magnitude": round(disaster_magnitude, 2),
                        "Hazard_Severity": round(hazard_severity, 2),
                        "Monthly_Rainfall_mm": round(rain, 1),  # Monthly raw rainfall in mm
                        "Compound_Event": compound_event,
                        "Rainfall_Anomaly": round(rain_anomaly, 4),
                        "Temperature_Celsius": round(temp, 1),
                        "Temperature_Anomaly": round(temp_anomaly, 4),
                        "Wind_Speed_kmph": round(wind_speed, 2),
                        "River_Level_Metres": round(river_level, 2),
                        "Soil_Moisture": round(soil_moisture, 4),
                        "Drought_Index": round(drought_idx, 2),
                        "Vegetation_Index": round(veg_idx, 4),
                        "Elevation_Metres": dist["Elevation_Metres"],
                        "Slope_Degrees": dist["Slope_Degrees"],
                        "Distance_From_Coast_km": dist["Distance_From_Coast_km"],
                        "Distance_From_River_km": dist["Distance_From_River_km"],
                        "Seismic_Activity_Index": round(seismic_val, 3),
                        "Population": dist["Population"],
                        "Area_SqKm": dist["Area_SqKm"],
                        "Population_Density": dist["Population_Density"],
                        "Urbanisation_Rate": dist["Urbanisation_Rate"],
                        "Number_of_Households": dist["Number_of_Households"],
                        "Infrastructure_Density": dist["Infrastructure_Density"],
                        "Agricultural_Land_Percentage": dist["Agricultural_Land_Percentage"],
                        "Industrial_Area_Percentage": dist["Industrial_Area_Percentage"],
                        "Poverty_Rate": dist["Poverty_Rate"],
                        "Elderly_Population_Percentage": dist["Elderly_Population_Percentage"],
                        "Child_Population_Percentage": dist["Child_Population_Percentage"],
                        "Literacy_Rate": dist["Literacy_Rate"],
                        "Housing_Quality_Index": dist["Housing_Quality_Index"],
                        "Healthcare_Access_Index": dist["Healthcare_Access_Index"],
                        "Road_Access_Index": dist["Road_Access_Index"],
                        "Communication_Access_Index": dist["Communication_Access_Index"],
                        "Raw_Shelter_Count": dist["Raw_Shelter_Count"],
                        "Raw_Hospital_Count": dist["Raw_Hospital_Count"],
                        "Raw_Rescue_Team_Count": dist["Raw_Rescue_Team_Count"],
                        "Early_Warning_System": dist["Early_Warning_System"],
                        "Evacuation_Plan_Available": dist["Evacuation_Plan_Available"],
                        "Emergency_Response_Time_Minutes": dist["Emergency_Response_Time_Minutes"],
                        "Disaster_Preparedness_Index": round(prep_score, 2),
                        "Government_Response_Score": round(self.rng.uniform(40, 95) - dist["Poverty_Rate"] * 20, 2)
                    }
                    
                    # 3. Conditional Post-Event Impact Generation (Ensure zero leakage to pre-event variables)
                    if disaster_occurred == 1:
                        # Human impact depends on severity, density, vulnerability, and preparedness
                        pop_exposure = dist["Population"] * (dist["Urbanisation_Rate"] if disaster_type in ["Heatwave", "Earthquake"] else 1.0)
                        
                        # Deaths: Poisson rate influenced by factors
                        death_rate_log = -10.0 + 0.7 * hazard_severity + 0.25 * np.log(dist["Population_Density"]) + 2.0 * dist["Poverty_Rate"] \
                                         - 0.5 * dist["Early_Warning_System"] - 0.02 * prep_score
                        death_rate = np.exp(np.clip(death_rate_log, -15, 6))
                        deaths = int(self.rng.poisson(lam=max(0.01, death_rate * (pop_exposure / 10000.0))))
                        
                        # Injuries: usually higher than deaths
                        injuries = int(deaths * self.rng.uniform(2.0, 10.0) + self.rng.uniform(1, 15))
                        
                        # People affected
                        affected_pct = np.clip(0.01 * hazard_severity * (2.0 - prep_score/100.0) + dist["Poverty_Rate"]*0.1, 0.001, 0.95)
                        affected = int(pop_exposure * affected_pct)
                        
                        # Displacement
                        displacement = int(affected * (0.1 + 0.4 * (1.0 - dist["Housing_Quality_Index"]/100.0)))
                        
                        # Damaged houses
                        houses_dmg = int((displacement / 4.5) * self.rng.uniform(0.6, 1.2))
                        
                        # Infrastructure damage score (1 to 100)
                        infra_dmg = np.clip(hazard_severity * 8.0 * (1.2 - dist["Housing_Quality_Index"]/100.0) + self.rng.normal(0, 5), 0, 100)
                        
                        # Crop loss: high for floods, cyclone, drought on agricultural land
                        crop_loss = 0.0
                        if disaster_type in ["Flood", "Cyclone", "Drought"]:
                            crop_loss = np.clip(hazard_severity * 8.0 * dist["Agricultural_Land_Percentage"] + self.rng.normal(0, 5), 0.0, 100.0)
                            
                        # Economic loss in millions
                        econ_loss_mean = (hazard_severity * 2.5 * dist["Infrastructure_Density"] * (1.0 - prep_score/150.0))
                        econ_loss = max(0.01, self.rng.gamma(shape=2.0, scale=max(0.1, econ_loss_mean / 2.0)))
                        
                        rec.update({
                            "Number_of_Deaths": deaths,
                            "Number_of_Injuries": injuries,
                            "Number_of_People_Affected": affected,
                            "Displacement_Count": displacement,
                            "Houses_Damaged": houses_dmg,
                            "Infrastructure_Damage_Score": round(infra_dmg, 2),
                            "Crop_Loss_Percentage": round(crop_loss, 2),
                            "Economic_Loss_Million": round(econ_loss, 4)
                        })
                    else:
                        rec.update({
                            "Number_of_Deaths": 0,
                            "Number_of_Injuries": 0,
                            "Number_of_People_Affected": 0,
                            "Displacement_Count": 0,
                            "Houses_Damaged": 0,
                            "Infrastructure_Damage_Score": 0.0,
                            "Crop_Loss_Percentage": 0.0,
                            "Economic_Loss_Million": 0.0
                        })
                        
                    records.append(rec)
                    
        df = pd.DataFrame(records)
        return df

    def validate_marginal_distributions(self, df):
        """
        Validates simulated data against plausible real-world reference ranges.

        Computes summary statistics for key variables and compares them against
        reference values from public datasets (IMD, Census India, EM-DAT).

        Parameters
        ----------
        df : pd.DataFrame
            The generated panel dataset.

        Returns
        -------
        dict
            Validation results with simulated vs reference range comparisons.
        """
        validation = {}

        reference_ranges = {
            "Monthly_Rainfall_mm": {"ref_min": 0, "ref_max": 1000, "ref_mean": 150, "source": "IMD India"},
            "Temperature_Celsius": {"ref_min": 5, "ref_max": 48, "ref_mean": 27, "source": "IMD India"},
            "Wind_Speed_kmph": {"ref_min": 0, "ref_max": 200, "ref_mean": 15, "source": "IMD Cyclone Atlas"},
            "Population_Density": {"ref_min": 50, "ref_max": 11000, "ref_mean": 700, "source": "Census India 2011"},
            "Poverty_Rate": {"ref_min": 0.05, "ref_max": 0.50, "ref_mean": 0.22, "source": "NITI Aayog 2015"},
            "Elevation_Metres": {"ref_min": 0, "ref_max": 8848, "ref_mean": 500, "source": "SRTM DEM"},
        }

        for col, ref in reference_ranges.items():
            if col not in df.columns:
                continue
            sim_stats = {
                "simulated_min": round(float(df[col].min()), 2),
                "simulated_max": round(float(df[col].max()), 2),
                "simulated_mean": round(float(df[col].mean()), 2),
                "simulated_std": round(float(df[col].std()), 2),
                "simulated_median": round(float(df[col].median()), 2),
            }
            sim_stats.update(ref)
            sim_stats["within_reference_range"] = (
                sim_stats["simulated_min"] >= ref["ref_min"] * 0.8 and
                sim_stats["simulated_max"] <= ref["ref_max"] * 1.2
            )
            validation[col] = sim_stats

        # Disaster prevalence check
        if "Disaster_Occurred" in df.columns:
            prevalence = float(df["Disaster_Occurred"].mean())
            validation["disaster_prevalence"] = {
                "simulated": round(prevalence, 4),
                "reference_range": "0.10–0.20 (event-months in high-risk regions)",
                "source": "DesInventar India 2000–2020",
                "within_range": 0.10 <= prevalence <= 0.20,
            }

        return validation

def generate_data_dictionary(df):
    """
    Generates a DataFrame representing the data dictionary of the dataset.
    """
    dict_records = []
    
    # We define role and definition for each column
    col_definitions = {
        "Record_ID": ("Identification", "Unique primary key for each district-month observation"),
        "District": ("Identification", "Unique ID code of the district (D_001 to D_100)"),
        "State": ("Identification", "State to which the district belongs (State_A to State_E)"),
        "Region": ("Identification", "Geographical region (North, Central, East, South, West)"),
        "Latitude": ("Spatial Coordinate", "Latitude coordinate of the district centroid"),
        "Longitude": ("Spatial Coordinate", "Longitude coordinate of the district centroid"),
        "Year": ("Temporal Coordinate", "Calendar year of the observation (2015 to 2025)"),
        "Month": ("Temporal Coordinate", "Calendar month of the observation (1 to 12)"),
        "Event_Date": ("Temporal Coordinate", "First date of the district-month (YYYY-MM-01)"),
        "Disaster_Type": ("Disaster Characteristic", "The type of disaster that occurred (Flood, Cyclone, etc., or No Disaster)"),
        "Disaster_Occurred": ("Disaster Occurrence Flag", "Binary flag: 1 if a disaster occurred, 0 otherwise"),
        "Disaster_Duration_Days": ("Disaster Characteristic", "Duration of the disaster event in days"),
        "Disaster_Magnitude": ("Disaster Characteristic", "Physical magnitude of the hazard event"),
        "Hazard_Severity": ("Disaster Characteristic", "Relative severity score of the hazard (1 to 10)"),
        "Monthly_Rainfall_mm": ("Environmental Variable", "Monthly raw rainfall in millimetres for the observation month"),
        "Compound_Event": ("Disaster Characteristic", "Binary flag: 1 if two or more hazard types activated simultaneously in this district-month, 0 otherwise"),
        "Rainfall_Anomaly": ("Environmental Variable", "Normalized rainfall deviation from the monthly baseline"),
        "Temperature_Celsius": ("Environmental Variable", "Mean temperature in Celsius"),
        "Temperature_Anomaly": ("Environmental Variable", "Temperature deviation from the monthly seasonal baseline"),
        "Wind_Speed_kmph": ("Environmental Variable", "Mean wind speed in kilometers per hour"),
        "River_Level_Metres": ("Environmental Variable", "Mean height of local rivers in meters"),
        "Soil_Moisture": ("Environmental Variable", "Normalized soil moisture content (0 to 1)"),
        "Drought_Index": ("Environmental Variable", "Index of agricultural dryness (0 to 100)"),
        "Vegetation_Index": ("Environmental Variable", "Normalized Difference Vegetation Index (NDVI) (0.1 to 0.9)"),
        "Elevation_Metres": ("Environmental Variable", "District elevation in meters above sea level"),
        "Slope_Degrees": ("Environmental Variable", "Average slope angle of district terrain in degrees"),
        "Distance_From_Coast_km": ("Environmental Variable", "Shortest distance to the coastline in kilometers"),
        "Distance_From_River_km": ("Environmental Variable", "Shortest distance to the major river in kilometers"),
        "Seismic_Activity_Index": ("Environmental Variable", "Index representing seismological activity levels"),
        "Population": ("Exposure Variable", "Total population of the district"),
        "Population_Density": ("Exposure Variable", "Population per square kilometer"),
        "Urbanisation_Rate": ("Exposure Variable", "Percentage of population living in urban areas"),
        "Number_of_Households": ("Exposure Variable", "Estimated total households in the district"),
        "Infrastructure_Density": ("Exposure Variable", "Index of infrastructure development density"),
        "Agricultural_Land_Percentage": ("Exposure Variable", "Percentage of land area dedicated to agriculture"),
        "Industrial_Area_Percentage": ("Exposure Variable", "Percentage of land area dedicated to industry"),
        "Poverty_Rate": ("Vulnerability Variable", "Percentage of population living below the poverty line"),
        "Elderly_Population_Percentage": ("Vulnerability Variable", "Percentage of population aged 65 or above"),
        "Child_Population_Percentage": ("Vulnerability Variable", "Percentage of population aged 14 or below"),
        "Literacy_Rate": ("Vulnerability Variable", "Percentage of literate population"),
        "Housing_Quality_Index": ("Vulnerability Variable", "Composite index of housing structural integrity (0-100)"),
        "Healthcare_Access_Index": ("Vulnerability Variable", "Composite index of healthcare accessibility (0-100)"),
        "Road_Access_Index": ("Vulnerability Variable", "Composite index of road infrastructure quality (0-100)"),
        "Communication_Access_Index": ("Vulnerability Variable", "Composite index of telecommunications access (0-100)"),
        "Raw_Shelter_Count": ("Preparedness Resource", "Raw count of operational emergency shelters"),
        "Raw_Hospital_Count": ("Preparedness Resource", "Raw count of general hospitals"),
        "Raw_Rescue_Team_Count": ("Preparedness Resource", "Raw count of disaster rescue teams"),
        "Early_Warning_System": ("Preparedness Variable", "Binary flag: 1 if operational, 0 otherwise"),
        "Evacuation_Plan_Available": ("Preparedness Variable", "Binary flag: 1 if detailed plan exists, 0 otherwise"),
        "Emergency_Response_Time_Minutes": ("Preparedness Variable", "Average emergency service response time in minutes"),
        "Disaster_Preparedness_Index": ("Preparedness Variable", "Composite preparedness score (10 to 100)"),
        "Government_Response_Score": ("Preparedness Variable", "Response rating score of local authorities (30 to 100)"),
        "Number_of_Deaths": ("Post-Event Impact Variable", "Total fatalities caused by the disaster event"),
        "Number_of_Injuries": ("Post-Event Impact Variable", "Total injuries caused by the disaster event"),
        "Number_of_People_Affected": ("Post-Event Impact Variable", "Total people requiring immediate relief or shelter"),
        "Displacement_Count": ("Post-Event Impact Variable", "Total individuals displaced from their homes"),
        "Houses_Damaged": ("Post-Event Impact Variable", "Total residential houses damaged or destroyed"),
        "Infrastructure_Damage_Score": ("Post-Event Impact Variable", "Score representing damage to roads/power (0-100)"),
        "Crop_Loss_Percentage": ("Post-Event Impact Variable", "Percentage of agricultural crop loss"),
        "Economic_Loss_Million": ("Post-Event Impact Variable", "Estimated economic loss in USD Millions"),
        "Shelter_Rate_per_100k": ("Engineered Exposure/Prep", "Emergency shelters per 100,000 population"),
        "Hospital_Rate_per_100k": ("Engineered Exposure/Prep", "Hospitals per 100,000 population"),
        "Rescue_Team_Rate_per_100k": ("Engineered Exposure/Prep", "Rescue teams per 100,000 population"),
        "Hazard_Score": ("Composite Index Component", "Descriptive score representing geographical hazard exposure (0-100)"),
        "Exposure_Score": ("Composite Index Component", "Descriptive score representing human and infrastructure exposure (0-100)"),
        "Vulnerability_Score": ("Composite Index Component", "Descriptive score representing socio-economic vulnerability (0-100)"),
        "Preparedness_Score": ("Composite Index Component", "Descriptive score representing emergency preparedness capacity (0-100)"),
        "Preparedness_Deficit_Score": ("Composite Index Component", "Descriptive score representing preparedness gap (100 - Preparedness_Score)"),
        "Disaster_Risk_Score": ("Composite Index Component", "Composite descriptive disaster risk index (Weighted combination of H, E, V, P_Deficit)"),
        "Equal_Weighted_Risk": ("Composite Index Component", "Descriptive risk score calculated with uniform component weights"),
        "Risk_Category": ("Composite Index Category", "Quantile-based categorical classification of Disaster_Risk_Score (Low, Moderate, High, Critical)"),
        "Previous_Month_Disaster_Occurred": ("Lag Feature", "Disaster occurrence flag from the previous month (t-1)"),
        "Previous_Month_Hazard_Severity": ("Lag Feature", "Hazard severity score from the previous month (t-1)"),
        "Rolling_12_Month_Disaster_Count": ("Rolling Feature", "Cumulative count of disasters in the district over the last 12 months"),
        "Disaster_Next_Month": ("Lead Variable (Target)", "Binary flag: 1 if a disaster occurs in the following month (t+1), 0 otherwise")
    }
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        if col in col_definitions:
            role, desc = col_definitions[col]
        else:
            role, desc = "Feature (Derived)", "Custom feature engineered column"
        dict_records.append({
            "Column_Name": col,
            "Data_Type": dtype,
            "Variable_Role": role,
            "Description": desc
        })
        
    return pd.DataFrame(dict_records)
