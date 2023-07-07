import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from shapely.geometry import box
import torch.nn.functional as F
import cv2 as cv
from skimage.segmentation import slic
from skimage.color import label2rgb
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_curve, auc, roc_auc_score, plot_roc_curve


## IMSHOW
def imshow(inp, title=None):
    """
    Imshow for Tensor: a Function that uses Matplotlib's PyPlot to show an image stored in a pytorch tensor.
    inputs: 
        inp   --> image stored in tensor.
        title --> title to be displayed with the image.
    returns: 
        None.
    """
    inp = inp[0,:,:]

    plt.imshow(inp, cmap='gray')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


## VISUALIZE MODEL
def visualize_model(model, loader, device, num_images=6):
    """
    Imshow for Tensor: a function to visualize some images and their predictions with the corresponding ground truth labels.
    inputs: 
        model      --> model  to predict the classes.
        loader     --> PyTorch dataloader to get the images.
        device     --> CUDA or CPU.
        num_images --> the number of images to show.
    returns: 
        None.
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            inputs = torch.cat((inputs, inputs, inputs), dim=1)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                #ax = plt.subplot(num_images//2, 2, images_so_far)
                #ax.axis('off')
                #ax.set_title(f'predicted: {preds[j]}')
                imshow(inputs.cpu().data[j])
                print(f'Actual: {labels[j]}, predicted: {preds[j]}')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


## PLOT CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


## PLOT ROC
def plot_roc(labels, preds, flag, model_name, n_classes):
    '''param : labels  --> True classification label
                preds  --> Predicted classification label
                flag   --> Boolean variable to determine if the ROC curves would be plotted (flag==True) or not (flag==False)
                clf    --> String containing the model's name
      return : roc_auc --> Dictionary of AUC values for each class'''  

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = ['#f06e93', '#327ba8']
    for i in range(n_classes):
        actuals = (np.array(labels) == i+1).astype(np.uint8)  # Creating a one-vs-rest array for each class
        print(actuals.shape)
        print(preds.shape)
        fpr[i], tpr[i], _ = roc_curve(actuals, preds[:, i])  # Computing the FPR and TPR
        roc_auc[i] = auc(fpr[i], tpr[i])  # Computing the AUC values for class i

    if(flag == True):  # If flag is True, then display the ROC curves
      for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve for {0} (area = {1:0.2f})'.format('Cancer/No Cancer', roc_auc[i]))  # roc_auc_score

      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic [ROC] for {0}'.format(model_name))
      plt.legend(loc="lower right")
      plt.show()
    return roc_auc


## SHOW ATTRIBUTIONS
def show_attributions(attributions_list, original_imgs, labels, predictions, cmap='coolwarm', alpha_map=0.8, alpha_img=0.4, title=None):
    """
    Plots 10 images of the attribution maps overlayed on the original images.
    Params:
        attributions_list: list with the attribution maps obtained from an explainability method.
        original_imgs: list containing the arrays of the original images
        labels: array with the label for each image.
        predictions: array with the prediction for reach image.
        cmap: color map to use. Default 'coolwarm'.
        alpha_map: alpha value for the attribution maps.
        alpha_img: alpha value for the original images.
        title: title for the figure. Default None.
    """
    # Create a 2x5 grid of subplots for the images
    fig, axs = plt.subplots(2,5, figsize=(12,6))
    axs = axs.ravel()

    for i in range(10):
        #attributions_channels_merged = np.maximum(attributions_list[i][0], attributions_list[i][1], attributions_list[i][2])
        axs[i].imshow(attributions_list[i][0], cmap=cmap, alpha=alpha_map, interpolation='nearest')
        axs[i].imshow(original_imgs[i], cmap='gray', alpha=alpha_img)
        axs[i].axis('off')
        axs[i].set_title(f'Lbl: {labels[i]}, Pred: {predictions[i]}')
    fig.suptitle(title)


## SHOW BOUNDING BOXES
def show_bbox(img_bbox_list, original_imgs, labels, predictions, alpha_map=0.9, alpha_img=0.6, title=None):
    """
    Plots 10 images of the bounding boxes obtained from the attribution maps overlayed on the original images.
    Params:
        img_bbox_list: list of the black images with the bounding boxes drawn on.
        original_imgs: list containing the arrays of the original images.
        labels: array with the label for each image.
        predictions: array with the prediction for each image.
        alpha_map: alpha for the image with the bounding boxes.
        alpha_img: alpha for the original images.
        title: title for the figure.
    """
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig, axs = plt.subplots(2,5, figsize=(12,6))
    axs = axs.ravel()
    for i in range(10):
        axs[i].imshow(img_bbox_list[i], alpha=alpha_map)
        axs[i].imshow(original_imgs[i], cmap='gray', alpha=alpha_img)
        axs[i].axis('off')
        axs[i].set_title(f'Lbl: {labels[i]}, Pred: {predictions[i]}')
    fig.suptitle(title)
    logger.setLevel(old_level)


## DILATE ATTRIBUTIONS 
def dilate_attributions(attributions_list, kernel_size=9, iterations=1):
    """
    Dilates the attribution maps according obtained from an explainability method.
    Params:
        attributions_list: list with the attribution maps obtained from an explainability method.
        kernel_size: the size of the square kernel used for dilating the attribution maps.
        iterations: number of iterations for the dilation.
    """
    attributions_list_dilated = []
    for i in range(len(attributions_list)):
        attributions_np = attributions_list[i]
        attributions_np = attributions_np.transpose(1,2,0)
        attributions_dilated = cv.dilate(attributions_np, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        attributions_dilated = attributions_dilated.transpose(2,0,1)
        attributions_list_dilated.append(attributions_dilated)
    return attributions_list_dilated


## BINARIZE ATTRIBUTIONS
def binarize_attributions(attributions_list, quantile=0.95):
    """
    Binarizes the attribution maps according to a threshold set by the quantile.
    Params:
        attributions_list: list with the attribution maps obtained from an explainability method.
        quantile: threshold for the binarization. Default is 0.95
    """
    binarized_attr_list = []
    kernel = np.ones((11,11), np.uint8)
    for i in range(len(attributions_list)):
        mask_img = attributions_list[i][0,:,:]
        mask = np.zeros(mask_img.shape[:2], np.uint8)
        fquantile = np.quantile(mask_img.ravel(), quantile)
        
        mask[mask_img>=fquantile] = 1
        mask[mask_img<=fquantile] = 0

        mask = cv.erode(mask, kernel, iterations=2)
        mask = cv.dilate(mask, kernel, iterations=2)
        mask = np.expand_dims(mask, axis=0)
        
        binarized_attr_list.append(mask)
    return binarized_attr_list


## IOU FUNCTION
def calculate_iou(box1, box2):
    """
    Computes the IOU of two bounding boxes.
    Params:
        box1: First list of shape [min_x, min_y, max_x, max_y]
        box2: Second list of shape [min_x, min_y, max_x, max_y]
    Returns:
        iou: The Intersection Over Union value of the two boxes
    """
    if math.isinf(box1[0]):
        return 0
    else:
        box1 = box(*box1)
        box2 = box(*box2)
        iou = box1.intersection(box2).area / box1.union(box2).area
        return iou


# COMBINE BOUNDING BOXES INTO A BIG ONE
def combine_bbox(bbox_list):
    """
    This function receives a list of sublists, where each sublist corresponds to an image from which bounding
    boxes were computed. The sublists contain tuples corresponding to each bounding box found in the image. If
    there are multiple bounding boxes, they are combined into a single big bounding box.
    Params:
        bbox_list: A list with a sublist for each image with tuples for each bbox found
    Returns:
        bbox_list_combined: A list list with one sublist for each image of shape [minx, miny, maxx, maxy]
    """
    combined_rects = []
    for sublist in bbox_list:
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        for rect in sublist:
            x, y, w, h = rect
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        width = max_x - min_x
        height = max_y - min_y
        combined_rect = [min_x, min_y, width, height]
        combined_rects.append(combined_rect)

    bbox_list_combined = [[x, y, x + w, y + h] for x, y, w, h in combined_rects]
    return bbox_list_combined


# OBTAIN BOUNDING BOXES FROM BINARIZED ATTRIBUTION MAPS
def attr_bbox(binarized_attr_list):
    """
    This function finds the contours from the binarized attribution map, and then
    finds the bounding boxes from those contours.
    Params:
        binarized_attr_list: A list of arrays of the binarized attribution maps
    Returns:
        bbox_list: A list with a sublist for each binarized attribution map containing
                   a of (x,y,w,h) for each bounding box found in the image
    """
    bbox_list = []
    # FIND THE CONTOURS OF THE BINARIZED ATTRIBUTIONS AND DRAW BOUNDING BOX
    for i in range(len(binarized_attr_list)):
        contours, hierarchy = cv.findContours(binarized_attr_list[i].squeeze(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # GET THE BOUNDING BOX COORDINATES FOR EACH CONTOUR
        bounding_boxes = [cv.boundingRect(c) for c in contours]
        bbox_list.append(bounding_boxes)
    return bbox_list


# DRAW THE COMBINED BOUNDING BOXES IN A BLACK IMAGE
def draw_bbox(bbox_list_combined, original_imgs, color=(0,255,0)):
    """
    This function creates black images to draw the bounding boxes on, to overlay it later on the original images.
    Params:
        bbox_list_combined: The list with a sublist for each image containing the previously
                            combined bounding box coordinates (only one bounding box per
                            image) of shape [minx, miny, maxx, maxy]
        original_imgs: The list containing the arrays of the original images
        color: RGB tuple
    Returns:
        img_bbox_list_combined: A list of black images with the corresponding bbox drawn.
    """
    img_bbox_list_combined = []
    for i in range(len(bbox_list_combined)):
        img_bbox = np.zeros_like(original_imgs[i])
        img_bbox = cv.cvtColor(img_bbox, cv.COLOR_GRAY2BGR)
        cv.rectangle(img_bbox, (bbox_list_combined[i][0], bbox_list_combined[i][1]), (bbox_list_combined[i][2], bbox_list_combined[i][3]), color, 5)
        img_bbox_list_combined.append(img_bbox)
    return img_bbox_list_combined

def plot_iou(iou_scores):
    """
    This function plots the IOU vs TPR graph.
    Params:s
        iou_scores -> a list with all the iou_scores
    Returns:
        none
    """
    iou_thresholds = np.arange(0,1,0.01)
    true_positives = []

    # Compute number of true positives for each threshold
    for threshold in iou_thresholds:
        tp = sum(iou >= threshold for iou in iou_scores)
        true_positives.append(tp)
    
    # Compute TPR
    total_positives = len(iou_scores)
    tpr = [tp/total_positives for tp in true_positives]

    # Plotting the graph
    plt.plot(iou_thresholds, tpr)
    plt.xlabel('IOU Threshold')
    plt.ylabel('True Positive Rate')
    plt.title('IOU vs True Positive Rate')
    plt.grid(True)
    plt.show()

def get_tpr(iou_scores):
    """
    This function plots the IOU vs TPR graph.
    Params:s
        iou_scores -> a list with all the iou_scores for an explainability method
    Returns:
        none
    """
    iou_thresholds = np.arange(0,1,0.01)
    true_positives = []

    # Compute number of true positives for each threshold
    for threshold in iou_thresholds:
        tp = sum(iou >= threshold for iou in iou_scores)
        true_positives.append(tp)
    
    # Compute TPR
    total_positives = len(iou_scores)
    tpr = [tp/total_positives for tp in true_positives]
    return tpr