# Usecase: SewerMapper

Colab between
- Wim Christiaansen
- Santosh Ahirrao
- Maarten Fish

**Challenge**: Automate manhole (sewer) geometry from pointcloud data and generate scientific drawing

*Update planned near the end of November*

# Researched technology
- [RANSAC](http://www.hinkali.com/Education/PointCloud.pdf): Detect basic shapes in unorganized point cloud data.
- [SVD](https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/):  Fit a circle to the cluster of points in 3D space.

During our initial research we came across some promising algorithms, the use case however provided only 4 days of intense development.  
This lead to the decision to focus our efforts on mapping the shaft of the manhole, creating our own algorithm, inspired by the researched methods.

# CloudSchematic

Our first task was to map a [single cylinder](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/cylinder.ipynb) for it's dimension:
- Slice cylinder into 10 segments (z-axis) and find the least noisy slice.
- Center point was calculated on the mean centroids by splitting the circle into 6 clusters (kmeans on x&y).
- Radius of the cirle was easily calculated through the mean distance from center point to all points on the circle.

Next we applied this to all [cylinders in the shaft](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/cloud_schematics.ipynb):
- Slice shaft in n-segments (every 1cm), min_max difference hinted at radius but very noisy.
- Applying a sequence of clustering techniques (on radius and z-axis) generated an accurate, cleaned-up cross section.
- Based on the clustered bounding boxes, we applied the same cylinder technique to retrieve the dimensions of each segment.
- Using matplotlib and numpy we achieved a simplified technical drawing with the diameter and height.

# Try it yourself

Module currently only reads raw vertices from a json file.  
(We used Blender, open source 3d-modeling software, to manually pull the vertices from the shaft)

Read [this](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/presentation.ipynb) notebook for a more visual tutorial.

```py
import json
from cloudtool import CloudSchematic

with open("path-to-json", "r") as file:
    cs = CloudSchematic(json.load(file))
    
    cs.generate(True)  # Mapping single cylinder (True returns plot) 
    cs.cross_section() # Clustering method for bounding boxes
    cs.project()       # Simplified orthographic representation
    cs.shaft_circles() # Calculates all shaft circles
    cs.project_shaft() # Generates front-view with measurements
```
# Contents
- [cloudtool.py](https://github.com/FishMaarten/SewerMapper/blob/master/cloudtool.py) The main module developed for the use case
- [plyfile.ipynb](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/plyfile.ipynb) Retrieving vertex data from .ply files
- [spectral_clustering.ipynb](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/spectral_clustering.ipynb) Attempt at automating segmentation of manhole components
- [cluster_cube.ipynb](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/cluster_cube.ipynb) Ransac algorithm attempt
- [cylinder.ipynb](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/cylinder.ipynb) Development of the single cylinder mapping
- [cloud_schematics.ipynb](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/cloud_schematics.ipynb) Development of CloudSchematic module
- [presentation.ipynb](https://github.com/FishMaarten/SewerMapper/blob/master/Notebooks/presentation.ipynb) Visual presentation of the module
