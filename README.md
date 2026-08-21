# Understanding affective responses to indoor environments using machine learning

This code used to quantify low-, mid and high-level visual eeatires in the IDEAS dataset. See Deniz et al. (2025), for more information about these visual features and how they are quantified. Figure below taken from Deniz et al. (2025) shows examples of visual features quantified.

![Examples of scene properties](/info/Picture1.svg)
*Examples of scene properties that might have an influence on the perception of environments and affective responses towards them. Low-level visual features consist of elementary features such as color, edges and spatial frequencies. Color plots represent hue, saturation and brightness distribution. Edges plots represent edge maps created by using Canny-edge detection (Canny, 1986), and spatial frequency plots show the power spectrum of selected images (Redies et al., 2020; Redies et al., 2007). Mid-level visual features combine these low-level visual features into meaningful characteristics that representing objects and scenes’ shapes. Object plots represent objects in the environment using a segmentation algorithm (Jain et al., 2023). Contour plots show orientation of contours in the images (Walther et al., 2023). Spatial layout plots show the depth map of selected images (Bhat et al., 2023). Symmetry plots show the mirror symmetry scores of selected images (Walther et al., 2023). High-level visual features combine both low- and mid-level visual features to represent more semantic and abstract information about the scene (Epstein & Baker, 2019). Possible functions (e.g., working, eating) and the type of scene (e.g., living room, office room) represent high-level visual features.* Figure 1 from Deniz, F. C., Chamilothori, K., Schoenmakers, S., & de Kort, Y. (2025). Do (not) enter? Objective visual features of indoor scenes predict approach-avoidance responses and core affect. Journal of Environmental Psychology, 102686.


## Repository Structure
`Extract Visual Features` include code to quantify various visual features:

**`extract_low_and_symmetry.py`** script processes input images and computes following features:
  - **Color features (Mean and Standard Deviation)**
    - Hue
    - Saturation
    - Brightness

  - **Edge features**
    - Edge density
    - First-order edge entropy
    - Second-order edge entropy

  - **Spatial Frequency features**
    - Fourier slope
    - Fourier sigma

  - **Symmetry features**
    - Left-right symmetry
    - Up-down symmetry

We used code from [AestheticToolbox](https://github.com/RBartho/Aesthetics-Toolbox/tree/main) to calculate edge, spatial frequency and symmetry features.

**`extract_objects.py`** uses [Maskformer](https://github.com/facebookresearch/maskformer) model to segment objects from images. `extract_objects.py`  quantifies binary presence of 150 objects detected by the model. List of all object can be found [here](https://github.com/CSAILVision/ADE20K)

**`extract_functions_and_spatial_layout`** uses [places365 model](https://github.com/csailvision/places365) to extract functions and spatial layout (returns a probability score between 0 an 1). 
  - **List of all functions extracted**
    - boating, driving, biking, transporting, sunbathing, touring, hiking, climbing, camping,
reading, studying, training, research, diving, swimming, bathing, eating, cleaning,
socializing, congregating, waiting in line, competing, sports, exercise, playing, gaming,
spectating, farming, constructing, shopping, medical activity, working, using tools, digging,
conducting business, praying


  - **List of all spatial layout features extracted**
    - open area, semi-enclosed area, enclosed area, far-away horizon, no horizon

`extract_contours.m` uses [Mid-level vision toolbox in MATLAB](https://github.com/bwlabToronto/MLV_toolbox) to extract contour features. Please download this toolbox and put it inside `Extract Visual Features\utils`.
  - **Contour features**
    - Contour Orientation
    - Contour Length
    - Contour Angularity


Results from all these models are saved in `Extracted Visual Features`. `process_visual_features.ipynb` includes how we processed the visual features, remove highly correlated visual features and combined them with the mean ratings obtained in IDEAS. All extracted features are saved in the `Extracted Visual Features` directory, making them available for downstream analysis or modeling. 

### Software requirements and data processing

We use `python` to extract all visual features except contour features, since there is no alternative of Mid-level vision toolbox in python. 

**We modified code to quantify a,b,c. Check compare results**

## Citation

If you use (part of) this dataset or code for your research, please cite our paper and other papers we used to generate(place holder for now):

Visual features:
```
Deniz, Fatih Celalettin, Kynthia Chamilothori, Sanne Schoenmakers, and Yvonne De Kort. “Do (Not) Enter? Objective Visual Features of Indoor Scenes Predict Approach-Avoidance Responses and Core Affect.” Journal of Environmental Psychology, July 2025, 102686. https://doi.org/10.1016/j.jenvp.2025.102686.
```
IDEAS dataset:
```

```
Mid-level vision toolbox:
```
Walther, D. B., Farzanfar, D., Han, S., & Rezanejad, M. (2023). The mid-level vision toolbox for computing structural properties of real-world images. Frontiers in Computer Science, 5, 1140723.
```
Aesthetics Toolbox:
```
Redies, C., Bartho, R., Koßmann, L., Spehar, B., Hübner, R., Wagemans, J., & Hayn-Leichsenring, G. U. (2025). A toolbox for calculating quantitative image properties in aesthetics research. Behavior Research Methods, 57(4), 117.
```
Maskformer:
```
Cheng, B., Schwing, A., & Kirillov, A. (2021). Per-pixel classification is not all you need for semantic segmentation. Advances in neural information processing systems, 34, 17864-17875.
```
Places365:
```
Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2017). Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence, 40(6), 1452-1464.
``` 
## References

Deniz, F. C., Chamilothori, K., Schoenmakers, S., & de Kort, Y. (2025). Do (not) enter? Objective visual features of indoor scenes predict approach-avoidance responses and core affect. *Journal of Environmental Psychology*, 102686.