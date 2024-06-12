
# .\venv\Scripts\activate
# python workingFile.py

import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import momepy as mp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import shapely.geometry as geom
from itertools import islice
from collections import Counter
import sklearn as sk
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import dgl

# Define the place name for which to download the street network
place_name = "Manhattan Island, New York, United States"

# Download the street network data from OSM
G = ox.graph_from_place(place_name, network_type='drive', simplify=True)

# Get the area boundary for the specified place
area = ox.geocode_to_gdf(place_name)

# Load eviction data and filter for Manhattan only
eviction_data = open(r"C:\Users\ARoncal\OneDrive - Thornton Tomasetti, Inc\__License Resources\Desktop\test-de\data\Evictions_20240612.csv")
eviction_df = pd.read_csv(eviction_data)
manhattan_evictions = eviction_df[eviction_df['BOROUGH'] == 'MANHATTAN']
clean_evictions = manhattan_evictions.dropna(subset=['Latitude'])

# Convert eviction data to GeoDataFrame
geometry = gpd.points_from_xy(clean_evictions['Longitude'], clean_evictions['Latitude'])
manhattan_evictions_gdf = gpd.GeoDataFrame(clean_evictions, geometry=geometry)
manhattan_evictions_gdf.set_crs(epsg=4326)

# Extract eviction points
eviction_gdf_points = manhattan_evictions_gdf[manhattan_evictions_gdf['geometry'].type == 'Point']
eviction_gdf_points.head()

# Convert graph to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# Plot the street network and eviction points
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.2, edgecolor='white', facecolor='black')
nodes.plot(ax=ax, color='white', markersize=0.1)
eviction_gdf_points.plot(ax=ax, markersize=3, color='red')
plt.show()

# Count the number of evictions for each edge
close_edges = ox.distance.nearest_edges(G, eviction_gdf_points['geometry'].x, eviction_gdf_points['geometry'].y)
eviction_count = Counter(close_edges)
evictions = []

# Assign eviction count to each edge
for i in edges.index:
    eviction = eviction_count[edges.loc[i].name]
    evictions.append(eviction)

# Add eviction data to edges GeoDataFrame
edges['evictions'] = evictions

# Plot evictions on the street network
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, column='evictions', cmap='hot', linewidth=1.5)
nodes.plot(ax=ax, color='white', markersize=0.1)
plt.show()

# Load property data
property_data = open(r'H:\03_AIA\01_GRAPHML\cleaned_property_data.csv')
property_df = pd.read_csv(property_data)

# Convert property data to GeoDataFrame
geometry = gpd.points_from_xy(property_df['Longitude'], property_df['Latitude'])
property_gdf = gpd.GeoDataFrame(property_df, geometry=geometry)
property_gdf.set_crs(epsg=4326)

# Extract property points
property_gdf_points = property_gdf[property_gdf['geometry'].type == 'Point']

# Plot property points on the street network
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.2, edgecolor='white', facecolor='black')
nodes.plot(ax=ax, color='white', markersize=0.1)
property_gdf_points.plot(ax=ax, markersize=3, color='blue')
plt.show()

# Find the nearest edge for each property point
nearest_edges = ox.distance.nearest_edges(G, property_gdf['geometry'].x, property_gdf['geometry'].y)
edge_values = {edge: [] for edge in nearest_edges}

# Collect property values for each edge
for i, edge in enumerate(nearest_edges):
    edge_values[edge].append(property_gdf.iloc[i]['AVTOT'])

# Calculate average property value for each edge
for edge, values in edge_values.items():
    avg_value = np.mean(values)
    u, v, key = edge
    edges.loc[(u, v, key), 'Average Value'] = avg_value

# Fill NaN values in the edges GeoDataFrame
edges = edges.fillna(0)

# Scale property values
from sklearn.preprocessing import StandardScaler
value_scaler = StandardScaler()
edges['Scaled Value'] = value_scaler.fit_transform(np.array(edges['Average Value']).reshape(-1, 1))

# Plot histogram of scaled property values
plt.hist(edges['Scaled Value'], bins=100)
plt.show()

# Plot scaled property values on the street network
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, column='Scaled Value', cmap='magma', linewidth=1)
nodes.plot(ax=ax, color='white', markersize=0.1)
plt.show()

# Load construction data
construction_data = open(r"C:\Users\ARoncal\OneDrive - Thornton Tomasetti, Inc\__License Resources\Desktop\test-de\data\HousingDB_post2010.csv")
construction_df = pd.read_csv(construction_data)
manhattan_construction = construction_df[construction_df['Boro'] == 1]
clean_construction = manhattan_construction.dropna(subset=['Latitude'])

# Convert construction data to GeoDataFrame
construction_geometry = gpd.points_from_xy(clean_construction['Longitude'], clean_construction['Latitude'])
construction_gdf = gpd.GeoDataFrame(clean_construction, geometry=construction_geometry)
construction_gdf.set_crs(epsg=4326)

# Extract construction points
construction_points = construction_gdf[construction_gdf['geometry'].type == 'Point']

# Plot construction points on the street network
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.2, edgecolor='white', facecolor='black')
nodes.plot(ax=ax, color='white', markersize=0.1)
construction_points.plot(ax=ax, markersize=3, color='yellow')
plt.show()

# Count the number of new constructions for each edge
close_edges = ox.distance.nearest_edges(G, construction_points['geometry'].x, construction_points['geometry'].y)
construction_count = Counter(close_edges)
new_construction = []

# Assign construction count to each edge
for i in edges.index:
    project = construction_count[edges.loc[i].name]
    new_construction.append(project)

# Add construction data to edges GeoDataFrame
edges['New Construction'] = new_construction

# Plot new constructions on the street network
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, column='New Construction', cmap='hot', linewidth=1.5)
nodes.plot(ax=ax, color='white', markersize=0.1)
plt.show()

# Import phik library for correlation analysis
import phik
from phik.report import plot_correlation_matrix
from phik import report
import seaborn as sns
import warnings

# Display first few rows of edges GeoDataFrame
edges.head()

# Analyze correlations using phik
attributes_to_check = ['Scaled Value', 'evictions', 'New Construction']
data_to_check = edges[attributes_to_check]
warnings.filterwarnings("ignore")

# Plot heatmap of correlations
plt.figure(figsize=(10, 5))
heatmap = sns.heatmap(data_to_check.phik_matrix(), annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
plt.show()

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

# Convert enriched graph back to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G_enriched)
nodes['avg_evictions']

# Load and process neighborhood data
neighborhood_data = open(r'C:\Users\jfoo\OneDrive - tvsdesign\Desktop\MACAD\AIA\Studio\AIA-GML-James-Andres\Data\Neighborhoods Boundries.geojson')
neighborhood_gdf = gpd.read_file(neighborhood_data)
neighborhood_gdf.to_crs(epsg=4326)

# Filter for Manhattan neighborhoods
manhatan_neighborhood_gdf = neighborhood_gdf[neighborhood_gdf['boroname'] == 'Manhattan']

# Create a mapping of neighborhood names to indices
unique_neighborhoods = manhatan_neighborhood_gdf['ntaname'].unique()
neighborhood_index = {neighborhood: idx for idx, neighborhood in enumerate(unique_neighborhoods)}

# Add neighborhood index to the GeoDataFrame
manhatan_neighborhood_gdf['neighborhood_index'] = manhatan_neighborhood_gdf['ntaname'].map(neighborhood_index)
manhatan_neighborhood_gdf.head()

# Assign neighborhoods to nodes
neighborhoods = []
for node, data in G_enriched.nodes(data=True):
    point = Point(data['x'], data['y'])
    for ind, shape in manhatan_neighborhood_gdf.iterrows():
        if shape['geometry'].contains(point):
            neighborhoods.append(shape['neighborhood_index'])

nodes['neighborhood'] = neighborhoods
nodes.head()

# Plot nodes with neighborhood assignments
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.5)
nodes.plot(ax=ax, column='neighborhood', cmap='hsv', markersize=10)
plt.show()

# Classify nodes based on evictions
classes = []
for node in nodes['avg_evictions']:
    if node > 0 and node <= 5:
        classes.append(1)
    elif node > 5 and node <= 10:
        classes.append(2)
    elif node > 10 and node <= 15:
        classes.append(3)
    elif node > 15 and node <= 20:
        classes.append(4)
    elif node > 20:
        classes.append(5)
    else:
        classes.append(6)

nodes['class'] = classes
nodes.head()

# Split data for training and testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

labeled_nodes = [i for i, j in zip(list(nodes.index), nodes['class']) if j != 6]
unlabeled_nodes = [i for i, j in zip(list(nodes.index), nodes['class']) if j == 6]

print(len(labeled_nodes))
print(len(unlabeled_nodes))

train_nodes, test_nodes = train_test_split(labeled_nodes, random_state=1, test_size=0.2)
train_nodes, validation_nodes = train_test_split(train_nodes, random_state=1, test_size=0.2)
prediction_nodes = unlabeled_nodes

print(len(train_nodes))
print(len(test_nodes))
print(len(validation_nodes))
print(len(prediction_nodes))

# Normalize features
scaler = StandardScaler()
train_features_std = scaler.fit_transform(nodes[['avg_evictions', 'avg_value', 'avg_construction']].loc[train_nodes])
validation_features_std = scaler.fit_transform(nodes[['avg_evictions', 'avg_value', 'avg_construction']].loc[validation_nodes])
test_features_std = scaler.fit_transform(nodes[['avg_evictions', 'avg_value', 'avg_construction']].loc[test_nodes])
prediction_features_std = scaler.fit_transform(nodes[['avg_evictions', 'avg_value', 'avg_construction']].loc[prediction_nodes])

# Assign normalized values back to the nodes
for ind, norm_train in zip(train_nodes, train_features_std):
    nodes.loc[ind, 'evictions_norm'] = norm_train[0]
    nodes.loc[ind, 'value_norm'] = norm_train[1]
    nodes.loc[ind, 'construction_norm'] = norm_train[2]

for ind, norm_validation in zip(validation_nodes, validation_features_std):
    nodes.loc[ind, 'evictions_norm'] = norm_validation[0]
    nodes.loc[ind, 'value_norm'] = norm_validation[1]
    nodes.loc[ind, 'construction_norm'] = norm_validation[2]

for ind, norm_test in zip(test_nodes, test_features_std):
    nodes.loc[ind, 'evictions_norm'] = norm_test[0]
    nodes.loc[ind, 'value_norm'] = norm_test[1]
    nodes.loc[ind, 'construction_norm'] = norm_test[2]

for ind, norm_prediction in zip(prediction_nodes, prediction_features_std):
    nodes.loc[ind, 'evictions_norm'] = norm_prediction[0]
    nodes.loc[ind, 'value_norm'] = norm_prediction[1]
    nodes.loc[ind, 'construction_norm'] = norm_prediction[2]

# Create dictionaries for node attributes
eviction_dic = {}
value_dic = {}
construction_dic = {}
neighborhood_dic = {}

classes_dic = {}
gdf_order_dic = {}

for i, j in enumerate(G_enriched.nodes()):
    eviction_dic[j] = nodes['evictions_norm'].iloc[i]
    value_dic[j] = nodes['value_norm'].iloc[i]
    construction_dic[j] = nodes['construction_norm'].iloc[i]
    neighborhood_dic = nodes['neighborhood'].iloc[i]
    classes_dic[j] = nodes['class'].iloc[i]
    gdf_order_dic[j] = i

# Assign node attributes to the graph
neighborhood_names = ["neighborhood " + str(c) for c in range(len(unique_neighborhoods))]
clusters_dictionaries = {c: {} for c in neighborhood_names}
print(clusters_dictionaries)

for i, j in enumerate(G_enriched.nodes()):
    for c in clusters_dictionaries.keys():
        clusters_dictionaries[c][j] = nodes[c].iloc[i]

for c in clusters_dictionaries.keys():
    nx.set_node_attributes(G, clusters_dictionaries[c], c)

nx.set_node_attributes(G_enriched, eviction_dic, 'evictions_norm')
nx.set_node_attributes(G_enriched, value_dic, 'value_norm')
nx.set_node_attributes(G_enriched, construction_dic, 'construction_norm')
nx.set_node_attributes(G_enriched, neighborhood_dic, 'neighborhood')
nx.set_node_attributes(G_enriched, gdf_order_dic, 'gdf_order')
nx.set_node_attributes(G_enriched, classes_dic, 'classes')

# Print the classes attribute of the nodes
print(nx.get_node_attributes(G_enriched, "classes"))
