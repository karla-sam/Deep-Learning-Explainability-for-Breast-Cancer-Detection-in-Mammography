# Deep-Learning-Explainability-for-Breast-Cancer-Detection-in-Mammography
This code was developed as a master thesis project for the Erasmus Mundus Joint Master's Degree in MedicAl Imaging and Applications (MAIA) by the University of Girona, the University of Cassino and Souther Lazio & the University of Burgundy. The work was developed in collaboration with the ViCOROB research group in the University of Girona.

![Highest TP part 1](https://github.com/karla-sam/Deep-Learning-Explainability-for-Breast-Cancer-Detection-in-Mammography/assets/101817439/816f5570-873f-4638-a54a-e814d01009a5)



## Main contributors
- Karla Sam Millan
- Robert Martí Marly

## Description
This repository offers the code for a qualitative and quantitative analysis of different XAI methods trained for breast cancer detection in mammography. The chosen explainability methods are:
- Saliency maps
- Occlusion
- Integrated Gradients
- Guided GradCAM
- DeepLIFT
- SHAP
- LIME

## Repository contents
- main: contains the main code for training the network and subsequently obtaining the attribution maps for each of the methods, as well as the IOU scores.
- aux_functions: contains the auxiliary functions used in main, mainly for the visualization of the model's performance, as well as the visualization of the attribution maps and the generation of the bounding boxes.
- models: contains the code for loading models from pytorch.
- best_models: contain the trained EfficientNetB0 and ResNet-50 models for whole-mammogram classification and patch-based classification, respectively.
- iou_scores: a csv file containing the IOU scores computed with the ground truth bounding boxes and the bounding boxes from the attribution maps for every image in the validation set for each of the explainability methods.
- documentation: contains the thesis paper, presentation and poster.
