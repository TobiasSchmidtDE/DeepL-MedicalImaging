from src.utils.load_model_crm import build_crm
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from src.utils.crm import CRM, decode_predictions
import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt

basepath = Path(os.getcwd())
# make sure your working directory is the repository root.
if basepath.name != "idp-radio-1":
    os.chdir(basepath.parent.parent)

load_dotenv(find_dotenv())


st.title('CRM Visualization')

models = ['DenseNet121_Chexpert_CWBCE_L1Normed_E3_B32_C0_N12_AugAffineColor_sharp21_U75_D256_DS9505_5LR1_LF1_SGD_Upsampled']

model_name = st.selectbox('Select the model', models)


crm = build_crm(model_name)

image = st.file_uploader("Upload image")

thresh = st.slider('Threshold', 0.0, 1.0, 0.3)

if image:
    with open("app/temp.png", "wb") as f:
        f.write(image.getbuffer())

    original, resized_crm, img, output = crm.single_image_crm(
        'app/temp.png', thresh)

    visualization = st.selectbox('Select the visualization mode', [
        'combined', 'class based'])

    top = decode_predictions(crm.classes, output[0], crm.num_classes)[:7]

    bbox, plt = crm.plot_crm(original, img, resized_crm, thresh)

    if visualization == 'combined':
        for c, i, p in top:
            st.write('{:15s}({}) {:f}'.format(c, i, p))
        bbox, plt = crm.plot_crm(original, img, resized_crm, thresh)

        st.subheader("Combined CRM Plot")
        st.write(plt)

    if visualization == 'class based':
        st.subheader("Class based CRM Plot")
        boxes = []
        for c, i, p in top[:3]:
            if p > 0.3:
                original, resized_crm, img, output = crm.single_image_crm(
                    'app/temp.png', thresh, class_idx=i)
                st.write('{:15s}({}) {:f}'.format(c, i, p))
                bbox, fig = crm.plot_crm(original, img, resized_crm, thresh)
                st.write(fig)
                boxes = boxes + bbox.tolist()

        fig = plt.figure(figsize=(15, 10))
        ax = plt.subplot(111)
        plt.imshow(original)

        for bbox in boxes:
            xs = bbox[1]
            ys = bbox[0]
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]
            rect = patches.Rectangle((ys, xs), w, h, linewidth=1,
                                     edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        st.write(fig)
