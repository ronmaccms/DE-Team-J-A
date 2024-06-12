<!-- PROJECT STATUS -->
<!-- <div align="center">
  <h3>🚧 This project is currently under development 🚧</h3>
</div> -->

<!-- PROJECT HEADER -->
<br />
<div align="center">
  <h3 align="center">Manhattan Eviction and Property Analysis</h3>
  <p align="center" style="font-weight: bold;">
    Analyzing eviction and property data in Manhattan<br>
    <!-- <a href="ADD_DEMO_LINK_HERE">View Demo</a>
    ·
    <a href="mailto:ADD_EMAIL_HERE">Report Bug</a>
    ·
    <a href="mailto:ADD_EMAIL_HERE">Request Feature</a> -->
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This project involves analyzing eviction and property data in Manhattan using various Python libraries and visualizing the data on street networks. The analysis aims to understand the patterns of evictions and property values and how new constructions affect these patterns.

### Project Breakdown

This code processes and visualizes eviction, property, and construction data for Manhattan. It uses the OSMnx library to download street network data and geopandas for spatial data analysis.

## Getting Started

To get a local copy up and running, follow these steps:

1. **Clone the repository**
    ```sh
    git clone https://github.com/your_username/your_project_name.git
    ```

2. **Navigate to the project directory**
    ```sh
    cd your_project_name
    ```

3. **Create and activate the virtual environment**
    - On Windows, use:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

4. **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```

5. **Prepare the data files**
    - Add links to your data files or provide instructions on where to obtain them.
    - Example: Download the eviction data from [add link here].

6. **Run the script**
    ```sh
    python workingFile.py
    ```

### Project Structure

- `workingFile.py`: The main script that processes and visualizes the data.
- `data/`: Directory where you should place your data files.
- `venv/`: Virtual environment directory.

### Script Overview

- **Street Network Data**: Downloading and processing street network data using OSMnx.
- **Eviction Data**: Loading, cleaning, and visualizing eviction data.
- **Property Data**: Loading, processing, and visualizing property data.
- **Construction Data**: Loading, processing, and visualizing construction data.
- **Correlation Analysis**: Analyzing correlations between evictions, property values, and new constructions using Phik.

### Deactivation

When you're done working, you can deactivate the virtual environment by running:
```sh
deactivate
