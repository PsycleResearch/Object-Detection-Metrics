import glob
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

from Object_Detection_Metrics.pascalvoc import ValidateFormats, ValidateCoordinatesTypes, getBoundingBoxes, \
    MethodAveragePrecision, CoordinatesType, \
    ValidateImageSize, Evaluator


def calculate_metrics(dict_name, gtFolder, detFolder, iouThreshold, gtFormat='xywh', detFormat='xywh', showPlot=False,
                      gtCoordinates='rel', detCoordinates='rel',
                      imgSize='540, 540', savePath='results'):
    # Get current path to set default folders
    currentPath = os.path.dirname(os.path.abspath(__file__))
    args = {}
    args['gtFolder'] = gtFolder
    args['detFolder'] = detFolder
    args['iouThreshold'] = iouThreshold
    args['gtCoordinates'] = gtCoordinates
    args['detCoordinates'] = detCoordinates
    args['imgSize'] = imgSize
    args['savePath'] = savePath
    args['gtFormat'] = gtFormat
    args['detFormat'] = detFormat

    # Arguments validation
    errors = []
    # Validate formats
    gtFormat = ValidateFormats(args['gtFormat'], '-gtformat', errors)
    detFormat = ValidateFormats(args['detFormat'], '-detformat', errors)
    # Groundtruth folder
    if os.path.isdir(gtFolder) is False:
        errors.append('folder %s not found' % gtFolder)
    # Coordinates types
    gtCoordType = ValidateCoordinatesTypes(args['gtCoordinates'], '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(args['detCoordinates'], '-detCoordinates', errors)
    imgSize = (0, 0)
    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args['imgSize'], '-imgsize', '-gtCoordinates', errors)
    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args['imgSize'], '-imgsize', '-detCoordinates', errors)

    # Detection folder
    if os.path.isdir(detFolder) is False:
        errors.append('folder %s not found' % detFolder)
    # If error, show error messages
    if len(errors) != 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()
    # Check if path to save results already exists and is not empty
    if os.path.isdir(savePath) and os.listdir(savePath):
        # Clear folder and save results
        shutil.rmtree(savePath, ignore_errors=True)
        os.makedirs(savePath)
    elif not os.path.isdir(savePath):
        os.makedirs(savePath)
    # Show plot during execution
    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(dict_name,
                                                    allBoundingBoxes,
                                                    # Object containing all bounding boxes (ground truths and detections)
                                                    IOUThreshold=iouThreshold,  # IOU threshold
                                                    method=MethodAveragePrecision.EveryPointInterpolation,
                                                    showAP=True,  # Show Average Precision in the title of the plot
                                                    showInterpolatedPrecision=False,
                                                    # Don't plot the interpolated precision curve
                                                    savePath=savePath,
                                                    showGraphic=showPlot)

    f = open(os.path.join(savePath, 'results.txt'), 'w')
    f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % dict_name[int(cl)])
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)


def convertToAbsoluteValues(size, box):
    # w_box = round(size[0] * box[2])
    # h_box = round(size[1] * box[3])
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)


def add_bb_into_image(image, bb, color=(255, 0, 0), thickness=2, label=None):
    size = (image.shape[1], image.shape[0])
    x1, y1, x2, y2 = convertToAbsoluteValues(size, bb)
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1
    cv2.rectangle(image, (x1, y1), (x2, y2), (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image


def draw_boxes(dict_name, annotations, img_folder, results, isGT=True):
    files = glob.glob(f"{annotations}/*.txt")
    # Extract bboxes
    for f in files:
        nameOfImage = f.split('/')[-1].replace(".txt", "")
        if isGT:
            image = np.array(Image.open(f'{img_folder}/{nameOfImage}.jpg'))
        else:
            image = np.array(Image.open(f'{results}/{nameOfImage}.jpg'))
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                img = add_bb_into_image(image, [x, y, w, h],
                                        color=(255, 0, 0), thickness=2,
                                        label=dict_name[int(idClass)])
            else:
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                img = add_bb_into_image(image, [x, y, w, h], color=(0, 0, 255), thickness=2,
                                        label=dict_name[int(idClass)])
        Image.fromarray(img).save(f'{results}/{nameOfImage}.jpg')
        fh1.close()


def add_legend(results):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color_pred = (255, 0, 0)
    color_truth = (0, 0, 255)
    thickness = 2
    images = glob.glob(f"{results}/*.jpg")
    for image_path in images:
        image = np.array(Image.open(image_path))
        org_truth = (image.shape[1] - 200, 100)
        org_pred = (image.shape[1] - 200, 150)
        cv2.putText(image, 'Pred', org_pred, font,
                    fontScale, color_pred, thickness, cv2.LINE_AA)
        cv2.putText(image, 'Verite', org_truth, font,
                    fontScale, color_truth, thickness, cv2.LINE_AA)
        Image.fromarray(image).save(image_path)


def draw_bounding_boxes_truth_pred(dict_name, gt_annotations, img_folder, results, prediction_annotations):
    """
    dict_name : dictionary of classes, index starting at 0
    gt_annotations : folder with ground truth annotations for test/val images
    prediction_annotations : folder with ground prediction annotations
    img_folder : folder which contains test images
    results : folder where results are to be saved
    """
    draw_boxes(dict_name, gt_annotations, img_folder, results, isGT=True)
    draw_boxes(dict_name, prediction_annotations, img_folder, results, isGT=False)
    add_legend(results)
