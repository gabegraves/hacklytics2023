#app v3
from pulp import *
import pulp
import numpy as np
from grid_solve import solve_lp, solve_random_lp
import streamlit as st
import requests
from geopy.geocoders import Nominatim
from geopy import distance
import folium
import streamlit.components.v1 as components
import json
from folium import plugins
import pandas as pd
# Define endpoints and API keys for OpenChargeMap API and ElectricityMap API
openchargemap_endpoint = "https://api.openchargemap.io/v3/poi/"
openchargemap_api_key = "dedd0e7b-b361-4a7b-8d3e-39690d3ec2f4"
electricitymap_endpoint = "https://api.electricitymap.org/v3/"
electricitymap_api_key = "tqC9EqNZWQzwqDO0bzXdbBIwpO28nbKz"

start_latitude, start_longitude = None, None
zones = []

# Define function to get carbon intensity and electricity mix for a given latitude and longitude from ElectricityMap API
def get_electricity_info(latitude, longitude):
    url1 = "https://api-access.electricitymaps.com/2w97h07rvxvuaa1g/carbon-intensity/history"
    url2 = "https://api-access.electricitymaps.com/2w97h07rvxvuaa1g/power-breakdown/history"
    headers = {"X-BLOBR-KEY": electricitymap_api_key}
    params = {"lat": latitude, "lon": longitude}
    response1 = requests.get(url1, headers=headers, params=params)
    response_json = response1.json()
    print(response_json)
    electricity_info1 = response_json["history"]
    ci_count = 0.0
    ci_sum = 0.0
    for ci in electricity_info1:
        ci_count += 1
        ci_sum += ci['carbonIntensity']
    carbon_intensity = round(ci_sum/ci_count, 2)
    response2 = requests.get(url2, headers=headers, params=params)
    response_json2 = response2.json()
    electricity_info2 = response_json2["history"]
    ffp_count = 0
    ffp_sum = 0
    for ff in electricity_info2:
        ffp_count += 1
        ffp_sum += ff['fossilFreePercentage']
    ffp = round(ffp_sum/ffp_count, 0)
    print(electricity_info2)
    return carbon_intensity, ffp

# Define function to get latitude and longitude from an address using Geopy Nominatim
def get_lat_lon_from_address(address):
    geolocator = Nominatim(user_agent="carbon_footprint_app")
    location = geolocator.geocode(address)
    return location.latitude, location.longitude

# Define function to get zones within a given radius of a given latitude and longitude from ElectricityMap API
# def get_zones_within_radius(latitude, longitude, radius):
    zones = []
    for zoom in [1, 5, 10, 20, 30, 40, 50]:
        url = f"{electricitymap_endpoint}zones?depth={zoom}"
        response = requests.get(url)
        response_json = json.loads(response.text)  # Convert response to JSON
        raise Exception(response_json)

        for zone in response_json:

            dist = distance.distance((latitude, longitude), (zone["latitude"], zone["longitude"])).km
            if dist <= radius:
                zones.append(zone)
    return zones


def page1():
    # Define Stream
    # Define Streamlit app
    st.title("Fleet EV Planner")

    # # Prompt user for starting and ending address and number of cars
    # start_address = st.text_input("Starting Address:")
    # end_address = st.text_input("Ending Address:")
    num_cars = st.number_input("Number of Cars:", min_value=1, step=1, value=1)

    df = pd.DataFrame( {
        "Start Address": [], 
        "End Address": [],
        "Start Latitude": [],
        "Start Longitude": [],
        "End Latitude": [],
        "End Longitude": [],
        "Battery Health": [],
        "Battery Capacity": [],
    }
    )

    for car in range(1, num_cars+1):
        # Prompt user for starting and ending address and number of cars
        st.markdown("""---""")
        st.markdown(f"## Car {car}")
        start_address = st.text_input(f"Starting Address {car}:")
        end_address = st.text_input(f"Ending Address {car}:")
        battery_health = st.number_input(f"Battery Health {car}:", min_value=0, max_value=100, step=1, value=25)
        battery_capacity = st.number_input(f"Battery Capacity {car}:", min_value=0, max_value=100, step=1, value=25)


        if start_address is not "" and end_address is not "":
            # print("Aksh is awesome")
            # Convert starting and ending address to latitude and longitude
            geolocator = Nominatim(user_agent="carbon_footprint_app")
            start_location = geolocator.geocode(start_address)
            end_location = geolocator.geocode(end_address)

            # create markdown that shows the longitude and latitude
            st.markdown(f"Starting Latitude: {start_location.latitude} | Starting Longitude: {start_location.longitude}")
            st.markdown(f"Ending Latitude: {end_location.latitude} | Ending Longitude: {end_location.longitude}")

            df.loc[len(df)] = [start_address, end_address, start_location.latitude, start_location.longitude, end_location.latitude, end_location.longitude, battery_health, battery_capacity]

    st.dataframe(df)

    button1 = st.button("Save Dataframe")
    button2 = st.button("Load Dataframe")
    #streamlit button that saves the dataframe to disk
    if button1:
        df.to_csv("df.csv")

    # add another button in that row
    if button2:
        df = pd.read_csv("df.csv", index_col=0)
        st.dataframe(df)

    # Show carbon intensity and electricity mix for given location on a box on the left of the app
    if st.button("Get Electricity Info"):
        for i in range(len(df)):
            row = df.iloc[i]
            latitude = row["Start Latitude"]
            longitude = row["Start Longitude"]
            start_address = row["Start Address"]
            end_address = row["End Address"]
            carbon_intensity, ffp = get_electricity_info(latitude, longitude)
            st.sidebar.subheader(f"Charging Info for car {i+1} travelling from {start_address} to {end_address}")
            st.sidebar.write(f"Carbon Intensity: {carbon_intensity} gCO2eq/kWh")
            st.sidebar.write(f"Electricity Mix: {ffp}% free of fossil fuels")

    if len(df) > 0:
        components.iframe(f"https://map.openchargemap.io/?mode=embedded?search=SanFrancisco&latitude={df.iloc[0]['Start Latitude']}&longitude={df.iloc[0]['Start Longitude']}", width=800, height=600)
    
    st.markdown("""---""")

    button3 = st.button("Run Planner")

    # run planner button
    if button3:
        st.markdown("""---""")
        st.markdown(f"## Planner")
        page2()

def page2():
    st.write("This is the planning page")


    df = pd.read_csv('df.csv')

    df['Start Latitude'] = df['Start Latitude'].astype(float)
    df['Start Longitude'] = df['Start Longitude'].astype(float)
    df['End Latitude'] = df['End Latitude'].astype(float)
    df['End Longitude'] = df['End Longitude'].astype(float)
    df['Battery Capacity'] = df['Battery Capacity'].astype(int)

    def normalize_column(df, col):
        if df[col].max() == df[col].min():
            df[col] = 0
            return df
        
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df

    df = pd.read_csv('df.csv')

    df = normalize_column(df, 'Start Latitude')
    df = normalize_column(df, 'Start Longitude')
    df = normalize_column(df, 'End Latitude')
    df = normalize_column(df, 'End Longitude')

    starts = df[['Start Latitude', 'Start Longitude']].values
    ends = df[['End Latitude', 'End Longitude']].values
    battery_level = df['Battery Capacity'].values

    size = 20

    starts = (starts * size).astype(int)
    ends = (ends * size).astype(int)

    vehicle_start = starts
    vehicle_end = ends
    num_chargers = 10
    charger_loc = np.random.randint(0, size, (num_chargers, 2))
    charger_co2s = np.random.rand(num_chargers)

    # start_battery = [25] * len(vehicle_start)
    start_battery = battery_level
    ending_battery = [10] * len(vehicle_start)
    T = 30

    prob, objective, x_out, c_out, b_out, c_cond_out, starts, ends =  solve_lp(T, vehicle_start, vehicle_end, charger_loc, charger_co2s, start_battery, ending_battery, speed=2, num_chargers_per_station=5, max_rate_of_charge=3, battery_per_mile=1, grid_size=size)
    prob.status

    # compute the total distance traveled
    total_distance = 0
    for i in range(len(vehicle_start)):
        for t in range(T-1):
            total_distance += np.linalg.norm(x_out[i, t, :] - x_out[i, t+1, :])


    # compute the total co2
    co2_obj = 0
    for ci in range(len(charger_loc)):
        for t in range(T-1):
            for i in range(len(vehicle_start)):
                co2_obj += c_out[i, ci, t] * charger_co2s[ci]
                

    time_to_solution = 0
    # Find the first timestamp after which the x_out doesn't change
    for t in range(T-1):
        if np.linalg.norm(x_out[:, t, :] - x_out[:, t+1, :]) > 0:
            time_to_solution = t
        else:
            break

    # display the total distance traveled, total co2, and time to solution markdown

    st.markdown(f"## Results")
    st.write("Found Trajectories Saved to 'model.lp' and 'model.mp4' and Transmitted to Your Drivers")

    results_df = pd.DataFrame({
        "Total Distance Traveled": [total_distance],
        "Total CO2": [co2_obj],
        "Peak Travel Time": [time_to_solution]
    })


    st.dataframe(results_df)


    # button to go back to page 1
    button1 = st.button("Go Back")
    if button1:
        page1()

if __name__ == '__main__':
    page1()
# st.row(button1, button2, button3)



# if start_location is not None and end_location is not None:
#     start_latitude = start_location.latitude
#     start_longitude = start_location.longitude
#     end_latitude = end_location.latitude
#     end_longitude = end_location.longitude

#     print(start_latitude, start_longitude, end_latitude, end_longitude)

#     # Get zones within a 50 mile radius of the starting and ending locations
#     zones = get_zones_within_radius(start_latitude, start_longitude, 50)
#     zones += get_zones_within_radius(end_latitude, end_longitude, 50)

#     print(zones)
#     zone_names = [zone["zoneName"] for zone in zones]

#     # Add markers for each zone to a folium map
#     m = folium.Map(location=[start_latitude, start_longitude], zoom_start=10)
#     for zone in zones:
#         folium.Marker(location=[zone["latitude"], zone["longitude"]], tooltip=zone["zoneName"]).add_to(m)

#     # Get carbon intensity and fossil fuel percentage for each zone
#     zone_info = {}
#     for zone_name in zone_names:
#         zone_info[zone_name] = {"carbon_intensity": 0, "fossil_fuel_percentage": 0}
#         latitude = zones[zone_names.index(zone_name)]["latitude"]
#         longitude = zones[zone_names.index(zone_name)]["longitude"]
#         carbon_intensity, fossil_fuel_percentage = get_electricity_info(latitude, longitude)
#         zone_info[zone_name]["carbon_intensity"] = carbon_intensity
#         zone_info[zone_name]["fossil_fuel_percentage"] = fossil_fuel_percentage
#         folium.Marker(location=[latitude, longitude], tooltip=f"{zone_name}\nCarbon Intensity: {carbon_intensity} gCO2eq/kWh\nElectricity Mix: {fossil_fuel_percentage}% free of fossil fuels").add_to(m)
#     st.plotly_chart(m)

# else:
#     st.write("Invalid address. Please try again.")

# # Prompt user for starting and ending address for multiple cars
# locations = []
# for i in range(num_cars):
#     st.subheader(f"Car {i+1}")
#     start_address = st.text_input(f"Starting Address {i+1}", key=f"start_address_{i}")
#     end_address = st.text_input(f"Ending Address {i+1}", key=f"end_address_{i}")
#     start_location = geolocator.geocode(start_address)
#     end_location = geolocator.geocode(end_address)
#     if start_location is not None and end_location is not None:
#         start_latitude = start_location.latitude
#         start_longitude = start_location.longitude
#         end_latitude = end_location.latitude
#         end_longitude = end_location.longitude
#         locations.append((start_latitude, start_longitude, end_latitude, end_longitude))
#     else:
#         st.write("Invalid address provided")

# # Define function to create a TripsLayer with the given data:
# def create_trips_layer(df):
#     return pdk.Layer(
#         "TripsLayer",
#         data=df,
#         get_path="path",
#         start_longitude="start_longitude",
#         start_latitude="start_latitude",
#         end_longitude="end_longitude",
#         end_latitude="end_latitude",
#         width_scale=20,
#         width_min_pixels=2,
#         rounded=True,
#         trail_length=100,
#         current_time=0,
#         auto_highlight=True,
#     )

# # Convert locations into a DataFrame that has the structure expected by the TripsLayer.
# locations_df = pd.DataFrame(
#     locations, columns=["start_latitude", "start_longitude", "end_latitude", "end_longitude"]
# )

# locations_df['start_longitude'] = start_longitude
# locations_df['start_latitude'] = start_latitude
# locations_df['end_longitude'] = end_longitude
# locations_df['end_latitude'] = end_latitude

# raise Exception(locations_df)

# # print(locations_df.head(10))
# locations_df["path"] = locations_df.apply(
#     lambda row: [[row["start_longitude"], row["start_latitude"]], [row["end_longitude"], row["end_latitude"]]], axis=1
# )

# locations_df["time"] = locations_df.index.to_series() * 3600
# locations_df["color"] = [[240, 98, 146]] * len(locations_df)

# # Create the TripsLayer and add it to the folium map object, m.
# trips_layer = create_trips_layer(locations_df)
# trips_layer_map = pdk.Deck(
#     layers=[trips_layer],
#     initial_view_state={
#         "latitude": start_latitude,
#         "longitude": start_longitude,
#         "zoom": 10,
#         "pitch": 50,
#     },
#     map_style="mapbox://styles/mapbox/light-v9",
# )
# folium_map = trips_layer_map.to_html()
# components.html(folium_map, width=1000, height=500, scrolling=True)