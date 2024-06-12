import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import momepy as mp
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from itertools import islice
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
import dgl
import phik
from phik.report import plot_correlation_matrix
import seaborn as sns
import warnings
from tqdm import tqdm

# Step 1: Load Manhattan street network data using OSMnx
place_name = "Manhattan Island, New York, United States"
G = ox.graph_from_place(place_name, network_type='drive', simplify=True)
area = ox.geocode_to_gdf(place_name)

# Step 2: Load and process eviction data
eviction_data_path = r"C:\Users\ARoncal\OneDrive - Thornton Tomasetti, Inc\__License Resources\Desktop\test-de\data\Evictions_20240612.csv"
eviction_df = pd.read_csv(eviction_data_path)
manhattan_evictions = eviction_df[eviction_df['BOROUGH'] == 'MANHATTAN']
clean_evictions = manhattan_evictions.dropna(subset=['Latitude'])

geometry = gpd.points_from_xy(clean_evictions['Longitude'], clean_evictions['Latitude'])
manhattan_evictions_gdf = gpd.GeoDataFrame(clean_evictions, geometry=geometry)
manhattan_evictions_gdf.set_crs(epsg=4326)

eviction_gdf_points = manhattan_evictions_gdf[manhattan_evictions_gdf['geometry'].type == 'Point']

# Step 3: Visualize street network and eviction points
nodes, edges = ox.graph_to_gdfs(G)
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.2, edgecolor='white', facecolor='black')
nodes.plot(ax=ax, color='white', markersize=0.1)
eviction_gdf_points.plot(ax=ax, markersize=3, color='red')
plt.show()

# Step 4: Count number of evictions for each edge
close_edges = ox.distance.nearest_edges(G, eviction_gdf_points['geometry'].x, eviction_gdf_points['geometry'].y)
eviction_count = Counter(close_edges)

evictions = [eviction_count.get(edge_name, 0) for edge_name in edges.index]
edges['evictions'] = evictions

# Step 5: Visualize evictions on edges
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, column='evictions', cmap='hot', linewidth=1.5)
nodes.plot(ax=ax, color='white', markersize=0.1)
plt.show()

# Step 6: Load and process property data
property_data_path = r'H:\03_AIA\01_GRAPHML\cleaned_property_data.csv'
property_df = pd.read_csv(property_data_path)

geometry = gpd.points_from_xy(property_df['Longitude'], property_df['Latitude'])
property_gdf = gpd.GeoDataFrame(property_df, geometry=geometry)
property_gdf.set_crs(epsg=4326)

property_gdf_points = property_gdf[property_gdf['geometry'].type == 'Point']

# Step 7: Visualize property points
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.2, edgecolor='white', facecolor='black')
nodes.plot(ax=ax, color='white', markersize=0.1)
property_gdf_points.plot(ax=ax, markersize=3, color='blue')
plt.show()

# Step 8: Calculate average property value for each edge
nearest_edges = ox.distance.nearest_edges(G, property_gdf['geometry'].x, property_gdf['geometry'].y)
edge_values = {edge: [] for edge in nearest_edges}

for i, edge in enumerate(nearest_edges):
    edge_values[edge].append(property_gdf.iloc[i]['AVTOT'])

for edge, values in edge_values.items():
    avg_value = np.mean(values)
    u, v, key = edge
    edges.loc[(u, v, key), 'Average Value'] = avg_value

edges = edges.fillna(0)

# Step 9: Scale property values
value_scaler = StandardScaler()
edges['Scaled Value'] = value_scaler.fit_transform(np.array(edges['Average Value']).reshape(-1, 1))

plt.hist(edges['Scaled Value'], bins=100)
plt.show()

# Step 10: Visualize scaled property values on edges
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, column='Scaled Value', cmap='magma', linewidth=1)
nodes.plot(ax=ax, color='white', markersize=0.1)
plt.show()

# Step 11: Load and process construction data
construction_data_path = r"C:\Users\ARoncal\OneDrive - Thornton Tomasetti, Inc\__License Resources\Desktop\test-de\data\HousingDB_post2010.csv"
construction_df = pd.read_csv(construction_data_path)
manhattan_construction = construction_df[construction_df['Boro'] == 1]
clean_construction = manhattan_construction.dropna(subset=['Latitude'])

construction_geometry = gpd.points_from_xy(clean_construction['Longitude'], clean_construction['Latitude'])
construction_gdf = gpd.GeoDataFrame(clean_construction, geometry=construction_geometry)
construction_gdf.set_crs(epsg=4326)

construction_points = construction_gdf[construction_gdf['geometry'].type == 'Point']

# Step 12: Visualize construction points
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.2, edgecolor='white', facecolor='black')
nodes.plot(ax=ax, color='white', markersize=0.1)
construction_points.plot(ax=ax, markersize=3, color='yellow')
plt.show()

# Step 13: Count number of new constructions for each edge
close_edges = ox.distance.nearest_edges(G, construction_points['geometry'].x, construction_points['geometry'].y)
construction_count = Counter(close_edges)
new_construction = [construction_count.get(edge_name, 0) for edge_name in edges.index]
edges['New Construction'] = new_construction

# Step 14: Visualize new constructions on edges
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, column='New Construction', cmap='hot', linewidth=1.5)
nodes.plot(ax=ax, color='white', markersize=0.1)
plt.show()

# Step 15: Analyze correlations using phik
warnings.filterwarnings("ignore")
attributes_to_check = ['Scaled Value', 'evictions', 'New Construction']
data_to_check = edges[attributes_to_check]

plt.figure(figsize=(10, 5))
heatmap = sns.heatmap(data_to_check.phik_matrix(), annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
plt.show()

# Step 16: Enrich the graph with computed attributes
G_enriched = ox.graph_from_gdfs(nodes, edges)

# Compute average evictions, property value, and new constructions for each node
for node in G_enriched.nodes():
    neighbors = list(G_enriched.neighbors(node))
    
    if neighbors:
        neighbor_evictions = [G_enriched.get_edge_data(node, neighbor)[0]['evictions'] for neighbor in neighbors]
        neighbor_values = [G_enriched.get_edge_data(node, neighbor)[0]['Average Value'] for neighbor in neighbors]
        neighbor_constructions = [G_enriched.get_edge_data(node, neighbor)[0]['New Construction'] for neighbor in neighbors]
        
        G_enriched.nodes[node]['avg_evictions'] = np.mean(neighbor_evictions)
        G_enriched.nodes[node]['avg_value'] = np.mean(neighbor_values)
        G_enriched.nodes[node]['avg_construction'] = np.mean(neighbor_constructions)
    else:
        G_enriched.nodes[node]['avg_evictions'] = 0
        G_enriched.nodes[node]['avg_value'] = 0
        G_enriched.nodes[node]['avg_construction'] = 0

nodes, edges = ox.graph_to_gdfs(G_enriched)

# Step 17: Load and process neighborhood data
neighborhood_data_path = r'C:\Users\jfoo\OneDrive - tvsdesign\Desktop\MACAD\AIA\Studio\AIA-GML-James-Andres\Data\Neighborhoods Boundries.geojson'
neighborhood_gdf = gpd.read_file(neighborhood_data_path)
neighborhood_gdf.to_crs(epsg=4326)

manhattan_neighborhood_gdf = neighborhood_gdf[neighborhood_gdf['boroname'] == 'Manhattan']
unique_neighborhoods = manhattan_neighborhood_gdf['ntaname'].unique()
neighborhood_index = {neighborhood: idx for idx, neighborhood in enumerate(unique_neighborhoods)}

manhattan_neighborhood_gdf['neighborhood_index'] = manhattan_neighborhood_gdf['ntaname'].map(neighborhood_index)

# Step 18: Assign neighborhoods to nodes
neighborhoods = []
for node, data in G_enriched.nodes(data=True):
    point = Point(data['x'], data['y'])
    for ind, shape in manhattan_neighborhood_gdf.iterrows():
        if shape['geometry'].contains(point):
            neighborhoods.append(shape['neighborhood_index'])

nodes['neighborhood'] = neighborhoods

# Step 19: Visualize nodes with neighborhood assignments
fig, ax = plt.subplots(figsize=(30, 50))
area.plot(ax=ax, facecolor='black')
edges.plot(ax=ax, linewidth=0.5)
nodes.plot(ax=ax, column='neighborhood', cmap='hsv', markersize=10)
plt.show()

# Step 20: Classify nodes based on evictions
classes = []
for eviction in nodes['avg_evictions']:
    if eviction > 0 and eviction <= 5:
        classes.append(1)
    elif eviction > 5 and eviction <= 10:
        classes.append(2)
    elif eviction > 10 and eviction <= 15:
        classes.append(3)
    elif eviction > 15 and eviction <= 20:
        classes.append(4)
    elif eviction > 20:
        classes.append(5)
    else:
        classes.append(6)

nodes['class'] = classes

# Step 21: Split data for training and testing
labeled_nodes = [i for i, j in zip(list(nodes.index), nodes['class']) if j != 6]
unlabeled_nodes = [i for i, j in zip(list(nodes.index), nodes['class']) if j == 6]

train_nodes, test_nodes = train_test_split(labeled_nodes, random_state=1, test_size=0.2)
train_nodes, validation_nodes = train_test_split(train_nodes, random_state=1, test_size=0.2)
prediction_nodes = unlabeled_nodes

# Step 22: Normalize features
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

# Step 23: Create dictionaries for node attributes
eviction_dic = {node: nodes.loc[node, 'evictions_norm'] for node in G_enriched.nodes()}
value_dic = {node: nodes.loc[node, 'value_norm'] for node in G_enriched.nodes()}
construction_dic = {node: nodes.loc[node, 'construction_norm'] for node in G_enriched.nodes()}
neighborhood_dic = {node: nodes.loc[node, 'neighborhood'] for node in G_enriched.nodes()}
classes_dic = {node: nodes.loc[node, 'class'] for node in G_enriched.nodes()}
gdf_order_dic = {node: i for i, node in enumerate(G_enriched.nodes())}

# Step 24: Assign node attributes to the graph
nx.set_node_attributes(G_enriched, eviction_dic, 'evictions_norm')
nx.set_node_attributes(G_enriched, value_dic, 'value_norm')
nx.set_node_attributes(G_enriched, construction_dic, 'construction_norm')
nx.set_node_attributes(G_enriched, neighborhood_dic, 'neighborhood')
nx.set_node_attributes(G_enriched, gdf_order_dic, 'gdf_order')
nx.set_node_attributes(G_enriched, classes_dic, 'classes')

print(nx.get_node_attributes(G_enriched, "classes"))
