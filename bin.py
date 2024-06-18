import pickle
import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Define the place name for which to download the street network
place_name = "Manhattan Island, New York, United States"

# Download the street network data from OSM
G = ox.graph_from_place(place_name, network_type='drive', simplify=True)

# Load eviction data and filter for Manhattan only
url = "https://raw.githubusercontent.com/ronmaccms/DE-Team-J-A/main/data/Evictions_20240612.csv"
eviction_data = pd.read_csv(url)
manhattan_evictions = eviction_data[eviction_data['BOROUGH'] == 'MANHATTAN']
clean_evictions = manhattan_evictions.dropna(subset=['Latitude'])

# Convert eviction data to GeoDataFrame
geometry = gpd.points_from_xy(clean_evictions['Longitude'], clean_evictions['Latitude'])
manhattan_evictions_gdf = gpd.GeoDataFrame(clean_evictions, geometry=geometry)
manhattan_evictions_gdf.set_crs(epsg=4326)

# Convert graph to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# Count the number of evictions for each edge
close_edges = ox.distance.nearest_edges(G, manhattan_evictions_gdf['geometry'].x, manhattan_evictions_gdf['geometry'].y)
eviction_count = Counter(close_edges)
evictions = [eviction_count[edge] for edge in edges.index]

# Add eviction data to edges GeoDataFrame
edges['evictions'] = evictions

# Create dummy property data
property_data = {
    'Latitude': [40.7831, 40.7128, 40.730610, 40.789623],
    'Longitude': [-73.9712, -74.0060, -73.935242, -73.959893],
    'AVTOT': [1000000, 2000000, 1500000, 2500000]
}
property_df = pd.DataFrame(property_data)

# Check if columns are correctly named
print("Property Data Columns:", property_df.columns)

# Ensure 'Longitude' and 'Latitude' columns are present
if 'Longitude' in property_df.columns and 'Latitude' in property_df.columns:
    # Convert property data to GeoDataFrame
    geometry = gpd.points_from_xy(property_df['Longitude'], property_df['Latitude'])
    property_gdf = gpd.GeoDataFrame(property_df, geometry=geometry)
    property_gdf.set_crs(epsg=4326)
else:
    print("Longitude and/or Latitude columns are missing in the property data.")

# Find the nearest edge for each property point
nearest_edges = ox.distance.nearest_edges(G, property_gdf['geometry'].x, property_gdf['geometry'].y)
edge_values = {edge: [] for edge in nearest_edges}

# Collect property values for each edge
for i, edge in enumerate(nearest_edges):
    edge_values[edge].append(property_gdf.iloc[i]['AVTOT'])

# Calculate average property value for each edge
avg_values = {edge: np.mean(values) for edge, values in edge_values.items() if values}
edges['Average Value'] = edges.index.map(avg_values).fillna(0)

# Scale property values
scaler = StandardScaler()
edges['Scaled Value'] = scaler.fit_transform(np.array(edges['Average Value']).reshape(-1, 1))

# Create dummy construction data
construction_data = {
    'Latitude': [40.7831, 40.7128, 40.730610, 40.789623],
    'Longitude': [-73.9712, -74.0060, -73.935242, -73.959893],
}
construction_df = pd.DataFrame(construction_data)

# Convert construction data to GeoDataFrame
construction_geometry = gpd.points_from_xy(construction_df['Longitude'], construction_df['Latitude'])
construction_gdf = gpd.GeoDataFrame(construction_df, geometry=construction_geometry)
construction_gdf.set_crs(epsg=4326)

# Count the number of new constructions for each edge
close_edges = ox.distance.nearest_edges(G, construction_gdf['geometry'].x, construction_gdf['geometry'].y)
construction_count = Counter(close_edges)
new_construction = [construction_count[edge] for edge in edges.index]

# Add construction data to edges GeoDataFrame
edges['New Construction'] = new_construction

# Enrich the graph with computed attributes
G_enriched = ox.graph_from_gdfs(nodes, edges)

# Compute average evictions, property value, and new constructions for each node
for node in G_enriched.nodes():
    neighbors = list(G_enriched.neighbors(node))
    if neighbors:
        neighbor_evictions = [G_enriched.get_edge_data(node, neighbor)[0]['evictions'] for neighbor in neighbors]
        neighbor_values = [G_enriched.get_edge_data(node, neighbor)[0]['Average Value'] for neighbor in neighbors]
        neighbor_constructions = [G_enriched.get_edge_data(node, neighbor)[0]['New Construction'] for neighbor in neighbors]
        
        avg_evictions = sum(neighbor_evictions) / len(neighbors)
        avg_value = sum(neighbor_values) / len(neighbors)
        avg_construction = sum(neighbor_constructions) / len(neighbors)

        G_enriched.nodes[node]['avg_evictions'] = avg_evictions
        G_enriched.nodes[node]['avg_value'] = avg_value
        G_enriched.nodes[node]['avg_construction'] = avg_construction
    else:
        G_enriched.nodes[node]['avg_evictions'] = 0
        G_enriched.nodes[node]['avg_value'] = 0
        G_enriched.nodes[node]['avg_construction'] = 0

# Prepare data to serialize
data_to_serialize = {
    "nodes": G_enriched.nodes(data=True),
    "edges": G_enriched.edges(data=True)
}

# Serialize and save to a .bin file
bin_file_path = 'Manhattan_data.bin'
with open(bin_file_path, 'wb') as f:
    pickle.dump(data_to_serialize, f)

print("Data has been serialized and saved to Manhattan_data.bin")
