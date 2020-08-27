# Obtaining data files

## Description of the data

This project uses geographical data in order to construct the 3D model of the buildings and the ground.

In the examples, the ground is considered to be flat but it is still possible to build a custom ground surface using polygons.

The files contain a list of 2D polygons describing the rooftops and a height attribute for each of them.
Each building will be built by extruding its polygon from z=0 to z=height.

## Getting data with [overpass turbo](https://overpass-turbo.eu/)

A way to get data for free is to use OpenStreetMap's data.

After arriving on the website, you will be able to select the area of your interest using the window as the delimiting box.

<img src = "../images/overpass-turbo/overpass_choosing_location.png" />