# Perception3D
Implementation of the 3D Perception Project Udacity Robotics Nanodegree
Thanks to everybody!
In order to complete the project first complete <a href="https://github.com/udacity/RoboND-Perception-Exercises">Perception exercises</a>
<h1>Different types of filtering used:</h1>
<ul>
    <li>Voxel Grid downsampling(downsampling of 3d to elementary cubes; the normal leaf_size is 0.01)</li>
    <li>Pass through filter(filter that leaves only the region of interest... takes filter axis as a parameter)</li>
    <li>RANSAC algorithm</li>
    <li>Statistical Outlier Removal Filter</li>
</ul>
<h1>Different types of CLUSTERING used:</h1>
<ul>
    <li>K-means clustering(unsupervised learning algorithm)</li>
    <li>DBSCAN algorithm</li>
    <li>Euclidiean clustering</li>
    
</ul>
<h1>Usage of SVM(support vector machine to determine and classify objects, by means of:</h1>
<ul>
    <li>Color</li>
    <li>Shape</li>
    <li>confusion matrices</li>
</ul>
All the code for the filtering uses useful methods of <a href="https://github.com/strawlab/python-pcl">PCL python</a>

At any point in time, the point cloud can be observed with pcl_viewer.
