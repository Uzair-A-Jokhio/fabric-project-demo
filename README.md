# Fabric-Pattern-Color Classifier Model API 
A FastAPI-based image classification API that predicts:

- Fabric type (12 classes like Cotton, Wool, Denim, etc.)
- Pattern type (12 classes like Floral, Geometric, Stripes, etc.)
- Dominant colors from an image

This project uses TensorFlow (MobileNetV2) for the fabric & pattern classifiers and KMeans clustering for dominant color extraction.

# Fabric Model 
- **Architecture:** MobileNetV2 (transfer learning)
- **Classes:** 12 types of fabrics


<table>
  <tr>
    <td align="center"><img src="misc/acrylic.png" width="120"/><br/>Acrylic</td>
    <td align="center"><img src="misc/blended.png" width="120"/><br/>Blended</td>
    <td align="center"><img src="misc/cotton.png" width="120"/><br/>Cotton</td>
    <td align="center"><img src="misc/denim.png" width="120"/><br/>Denim</td>
  </tr>
  <tr>
    <td align="center"><img src="misc/fleece.png" width="120"/><br/>Fleece</td>
    <td align="center"><img src="misc/fur.jpeg" width="120"/><br/>Fur</td>
    <td align="center"><img src="misc/leather.png" width="120"/><br/>Leather</td>
    <td align="center"><img src="misc/polyester.png" width="120"/><br/>Polyester</td>
  </tr>
  <tr>
    <td align="center"><img src="misc/silk.png" width="120"/><br/>Silk</td>
    <td align="center"><img src="misc/velvet.png" width="120"/><br/>Velvet</td>
    <td align="center"><img src="misc/wool.png" width="120"/><br/>Wool</td>
    <td align="center"><img src="path/to/image12.png" width="120"/><br/>other</td>
  </tr>
</table>


# Pattern Model 
- **Architecture:** MobileNetV2 (transfer learning)
- **Classes:** 12 types of patterns

<table>
  <tr>
    <td align="center"><img src="misc/1abstract_graph.jpg" width="120"/><br>abstract_graph</td>
    <td align="center"><img src="misc/2animals.jpg" width="120"/><br/>Animals</td>
    <td align="center"><img src="misc/3birds.jpg" width="120"/><br/>Birds</td>
    <td align="center"><img src="misc/4checks.jpg" width="120"/><br/>Checks</td>
  </tr>
  <tr>
    <td align="center"><img src="misc/5damasks.jpg" width="120"/><br/>Damaks</td>
    <td align="center"><img src="misc/6floral.jpg" width="120"/><br/>Floral</td>
    <td align="center"><img src="misc/7geometric.jpg" width="120"/><br/>Geometric</td>
    <td align="center"><img src="misc/8kids.jpg" width="120"/><br/>Kids</td>
  </tr>
  <tr>
    <td align="center"><img src="misc/9leaf_trees.jpg" width="120"/><br/>Leafs Trees</td>
    <td align="center"><img src="misc/10plains_texture.jpg" width="120"/><br/>Plain texture</td>
    <td align="center"><img src="misc/11spots.jpg" width="120"/><br/>Spots</td>
    <td align="center"><img src="misc/12stripes.jpg" width="120"/><br/>Stripes</td>
  </tr>
</table>

# Color Model

- **Method:** KMeans clustering on image pixels (OpenCV + scikit-learn)
- **Goal:** Extract dominant color and map to the nearest known color name