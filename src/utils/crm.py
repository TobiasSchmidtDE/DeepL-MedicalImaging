import keras.backend as K
import cv2
from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input

import numpy as np

from scipy import ndimage
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def generate_ensemble_crm(crms, image_path, thresh):
    num_models = len(crms)
    outputs = np.zeros_like(crms)
    crm_outputs = np.zeros_like(crms)

    for i in range(num_models):
        original, output, resized_crm = crms[i].generate_crm_combined(
            image_path)
        crm_outputs[i] = resized_crm
        outputs[i]  = output

    output = outputs.sum(axis=0) / num_models
    resized_crm = crm_outputs.sum(axis=0) / num_models

    heatmap = cv2.applyColorMap(
        np.uint8(255 * resized_crm), cv2.COLORMAP_JET)
    heatmap = (255 - heatmap)
    heatmap[np.where(resized_crm < thresh)] = 0
    aImg = np.float32(heatmap) * 0.8 + np.float32(original)
    img = 255 * aImg / np.max(aImg)

    bbox, fig = crms[0].plot_crm(original, img, resized_crm, thresh)

    return img, output, fig, bbox


def generate_ensemble_crm_class(crms, image_path, thresh, class_idx=0):
    num_models = len(crms)
    outputs = np.zeros_like(crms)
    crm_outputs = np.zeros_like(crms)

    for i in range(num_models):
        original, output, resized_crm = crms[i].generate_crm_class(
            image_path, class_idx)
        crm_outputs[i] = resized_crm
        outputs[i] = output

    output = outputs.sum(axis=0) / num_models
    resized_crm = crm_outputs.sum(axis=0) / num_models

    heatmap = cv2.applyColorMap(
        np.uint8(255 * resized_crm), cv2.COLORMAP_JET)
    heatmap = (255 - heatmap)
    heatmap[np.where(resized_crm < thresh)] = 0
    aImg = np.float32(heatmap) * 0.8 + np.float32(original)
    img = 255 * aImg / np.max(aImg)

    bbox, fig = crms[0].plot_crm(original, img, resized_crm, thresh)

    return original, img, output, fig, bbox

class CRM:
    def __init__(self, model, classes, dims=(256, 256), preprocess_fn=None):
        self.model = model
        self.num_classes = len(classes)
        self.dims = dims
        self.classes = classes
        self.preprocess_fn = preprocess_fn

    def generate_crm_combined_plot(self, image_path, thresh):
        original, resized_crm, img, output = self.single_image_crm(
            image_path, thresh)

        top = decode_predictions(self.classes, output[0], self.num_classes)[:7]
        print('Model prediction:')
        for c, i, p in top:
            print('\t{:15s}\t({})\twith probability \t{}'.format(c, i, p))

        return self.plot_crm(original, img, resized_crm, thresh)

    def generate_crm_class_plot(self, image_path, thresh, top_num=3, figsize=(6, 3)):
        original, resized_crm, img, output = self.single_image_crm(
            image_path, thresh)

        top = decode_predictions(
            self.classes, output[0], self.num_classes)[:top_num]

        boxes = []
        for c, i, p in top:
            original, resized_crm, img, output = self.single_image_crm(
                image_path, thresh, class_idx=i)
            print('{:15s}({})with probability {}'.format(c, i, p))
            bbox, fig = self.plot_crm(
                original, img, resized_crm, thresh, figsize=figsize)
            boxes = boxes + bbox.tolist()

        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        plt.imshow(original)

        for bbox in boxes:
            ys = bbox[1]
            xs = bbox[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle((xs, ys), w, h, linewidth=1,
                                     edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()

    def plot_crm(self, original, img, resized_crm, thresh, figsize=(6, 3)):
        aBBox_coord = self.generate_bBox(resized_crm, thresh + 0.3)

        fig = plt.figure(figsize=figsize)

        plt.subplot(131)
        plt.title('Original')
        plt.axis('off')
        plt.imshow(original)

        plt.subplot(132)
        plt.title('CRM')
        plt.axis('off')
        plt.imshow(img.astype(np.uint8))

        ax = plt.subplot(133)
        plt.title('Bboxes')
        plt.axis('off')
        plt.imshow(original)

        # Create a Rectangle patch
        for bbox in aBBox_coord:
            ys = bbox[1]
            xs = bbox[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle((xs, ys), w, h, linewidth=1,
                                     edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        return aBBox_coord, fig

    def get_predictions(self, image):
        image = np.expand_dims(image, axis=0)
        print(self.model.predict(image))

    def single_image_crm(self, image_path, thresh, class_idx=None):
        if class_idx:
            original, output, resized_crm = self.generate_crm_class(
                image_path, class_idx)
        else:
            original, output, resized_crm = self.generate_crm_combined(
                image_path)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * resized_crm), cv2.COLORMAP_JET)
        heatmap = (255 - heatmap)
        heatmap[np.where(resized_crm < thresh)] = 0
        aImg = np.float32(heatmap) * 0.8 + np.float32(original)
        aImg = 255 * aImg / np.max(aImg)

        return original, resized_crm, aImg, output

    def generate_crm_combined(self, image_path):
        width, height = self.dims

        original_img = Image.open(image_path)
        original_img = original_img.convert(mode="RGB")
        resized_original_image = original_img.resize(self.dims)
        input_image = np.asarray(resized_original_image)

        input_image = np.expand_dims(input_image, axis=0)
        input_image = self.preprocess_fn(input_image)

        class_weights = self.model.layers[-1].get_weights()[0]

        get_output = K.function([self.model.layers[0].input], [
            self.model.layers[-4].output, self.model.layers[-1].output])
        [conv_outputs, predictions] = get_output([input_image])
        conv_outputs = conv_outputs[0, :, :, :]

        final_output = predictions

        iMSE = []
        for j in range(self.num_classes):
            wf_j = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

            for i, w in enumerate(class_weights[:, j]):
                wf_j += w * conv_outputs[:, :, i]
            S_j = np.sum(wf_j)

            iMSE_j = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

            row, col = wf_j.shape

            for x in range(row):
                for y in range(col):
                    tmp = np.array(wf_j)
                    # remove activation at the spatial location (x,y)
                    tmp[x, y] = 0.
                    iMSE_j[x, y] = (S_j - np.sum(tmp)) ** 2
            iMSE.append(iMSE_j)

        crm = iMSE[0]
        for i in range(1, self.num_classes):
            crm += iMSE[i]

        # normalize
        crm /= np.max(crm)
        # upscaling to original image size
        resized_crm = cv2.resize(crm, (height, width))

        return [resized_original_image, final_output, resized_crm]

    def generate_crm_class(self, image_path, idx):
        original_img = cv2.imread(image_path)
        width, height = self.dims
        resized_original_image = cv2.resize(original_img, (width, height))

        input_image = img_to_array(resized_original_image)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = preprocess_input(input_image)

        class_weights = self.model.layers[-1].get_weights()[0]

        get_output = K.function([self.model.layers[0].input], [
            self.model.layers[-4].output, self.model.layers[-1].output])
        [conv_outputs, predictions] = get_output([input_image])
        conv_outputs = conv_outputs[0, :, :, :]

        final_output = predictions

        iMSE = []
        wf = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

        for i, w in enumerate(class_weights[:, idx]):
            wf += w * conv_outputs[:, :, i]
        S = np.sum(wf)

        iMSE = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

        row, col = wf.shape

        for x in range(row):
            for y in range(col):
                tmp = np.array(wf)
                # remove activation at the spatial location (x,y)
                tmp[x, y] = 0.
                iMSE[x, y] = (S - np.sum(tmp)) ** 2

        crm = iMSE

        # normalize
        crm /= np.max(crm)
        # upscaling to original image size
        resized_crm = cv2.resize(crm, (height, width))

        return [resized_original_image, final_output, resized_crm]

    def generate_bBox(self, crm, threshold):
        bboxes = []
        TheProps = self.generate_BoundingBox(crm, threshold)
        for b in TheProps:
            bbox = b.bbox
            bboxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])

        CRM_bboxes = np.vstack(bboxes)

        return CRM_bboxes

    def generate_BoundingBox(self, aCRM, threshold):
        labeled_CRM, nr_objects = ndimage.label(aCRM > threshold)
        props = regionprops(labeled_CRM)
        return props

    def Calculate_Confidence_Score(self, crm, bboxes, outScores):
        c_scores = []
        for a_b in bboxes:
            a_bbox = crm[a_b[1]:a_b[3], a_b[0]:a_b[2]]
            # a_score = np.max(a_bbox)
            # b_score = outScores[0][0]
            c_scores.append(np.max(a_bbox) * outScores[0][0])

        return np.array(c_scores)


def decode_predictions(classes, predictions, num_classes):
    decoded_predictions = []
    for i in range(num_classes):
        decoded_predictions.append((classes[i], i, predictions[i]))

    sorted_preds = sorted(decoded_predictions, key=lambda tup: -tup[2])
    return sorted_preds
