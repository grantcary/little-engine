<div align="center">
<h1>Little Engine</h1>
<img src='https://github.com/grantcary/little-engine/blob/main/rendered_images/skyscapeexample.PNG' alt='monkey+cube' width='960'>
</div>

As far as I know, this is the fastest CPU based python ray tracer, aiming to become even faster.

Every object has to be a triangle mesh or polygon mesh (which gets converted to a triangle mesh). Little Engine does not support simple object primatives such as cube or sphere. 
The goal of this project is to optimize and accelerate triangle mesh CPU ray tracing to its absolute limits with python.

Supported Features:
 - Triangle Meshes
 - Shadows
 - Reflections
 - Bounding Volume Hierarchy
 - Meshlets
 - Skybox (equirectangular image)

# Examples:
All examples were run on an AMD Ryzen 7 3700x (8 core) CPU

The title image contained 980 triangles and was rendered at 1920 by 1080. Render Time: 4m 13s

