import os
import io
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

basepath = Path(os.getcwd())
# make sure your working directory is the repository root.
if basepath.name != "idp-radio-1":
    os.chdir(basepath.parent.parent)

load_dotenv(find_dotenv())

from src.utils.load_model_crm import build_crm
from src.utils.crm import CRM, decode_predictions, generate_ensemble_crm, generate_ensemble_crm_class

colors = ['#F79F1F', '#A3CB38', '#1289A7',
          '#D980FA', '#B53471', '#EE5A24', '#009432', '#0652DD', '#9980FA', '#EA2027', '#5758BB', '#ED4C67']

model_type = st.sidebar.selectbox(
    "Model Type",
    ("Ensemble", "Single")
)


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Model Prediction Visualization')

models = os.listdir('models')
index = models.index(
    'DenseNet121_Chexpert_CWBCE_L1Normed_E3_B32_C0_N12_AugAffine_sharp21_U75_D256_DS9505_2LR4_LF5_Adam_Upsampled')
models = [model for model in models if "Chexpert" in model]

if model_type == 'Single':
    thresh = st.sidebar.slider('Threshold for bounding boxes', 0.0, 1.0, 0.3)

    model_name = st.selectbox('Select the model', models, index=index)

    with st.spinner('Loading model....'):
        try:
            crm = build_crm(model_name)
        except Exception as err:
            print(err)
            st.error("Could not load selected model: {}".format(err))

    image = st.file_uploader("Upload image")


    with st.spinner('Evaluating image....'):
        if image:
            with open("app/temp.png", "wb") as f:
                f.write(image.getbuffer())

            original, resized_crm, img, output = crm.single_image_crm(
                'app/temp.png', thresh)

            visualization = st.selectbox('Select the visualization mode', [
                'combined', 'class based'])

            top = decode_predictions(crm.classes, output[0], crm.num_classes)[:7]

            if visualization == 'combined':
                for c, i, p in top:
                    st.write('{:15s}({}) {:f}'.format(c, i, p))
                bbox, plot = crm.plot_crm(original, img, resized_crm, thresh)

                st.subheader("Combined CRM Plot")
                st.write(plot)

            if visualization == 'class based':
                st.subheader("Class based CRM Plot")
                pred_thresh = st.slider('Prediction threshold', 0.0, 1.0, 0.3)

                boxes = []
                for c, i, p in top:
                    if p > pred_thresh:
                        original, resized_crm, img, output = crm.single_image_crm(
                            'app/temp.png', thresh, class_idx=i)
                        st.write('{:15s}({}) {:f}'.format(c, i, p))
                        bbox, fig = crm.plot_crm(
                            original, img, resized_crm, thresh)
                        st.write(fig)
                        newboxes = bbox.tolist()
                        newboxes = [(i, box) for box in newboxes]
                        boxes = boxes + newboxes

                fig = plt.figure(figsize=(6, 4))
                ax = plt.subplot(111)
                plt.imshow(np.zeros((256, 256)))
                plt.imshow(original)
                plt.axis('off')

                for box in boxes:
                    i = box[0]
                    bbox = box[1]
                    ys = bbox[1]
                    xs = bbox[0]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]

                    if w > 25 and h > 25:
                        rect = patches.Rectangle((xs, ys), w, h, linewidth=1,
                                                edgecolor=colors[i], facecolor='none')

                        # Add the patch to the Axes
                        ax.add_patch(rect)
                        ax.annotate(crm.classes[i], (xs, ys), color=colors[i],
                                    fontsize=6, ha='left', va='bottom')
                        for c, j, p in top[:3]:
                            if i == j:
                                prob = p
                        ax.annotate("{:.2f}".format(prob), (xs + w, ys), color=colors[i],
                                    fontsize=6, ha='left', va='bottom')

                st.write(fig)

elif model_type == 'Ensemble':
    thresh = st.sidebar.slider('Threshold for bounding boxes', 0.0, 1.0, 0.25)


    default = ['DenseNet121_Chexpert_BCE_E3_B32_C0_N12_Uones_D256_DS9505_2LR1_LF5_SGD_Upsampled_1',
               'DenseNet121_Chexpert_BCE_E3_B32_C0_N12_AugAffine_Uones_D256_DS9505_2LR1_LF5_SGD_Upsampled_1',
               'DenseNet121_Chexpert_WBCE_E3_B32_C0_N12_AugAffine_U75_D256_DS9505_1LR1_LF5_SGD_Upsampled',
               'DenseNet121_Chexpert_BCE_E3_B32_C0_N12_AugAffine_U66_D256_DS9505_2LR1_LF5_SGD_Upsampled_1']

    model_names = st.multiselect('Select the models', models, default=default)

    with st.spinner('Loading models....'):
        try:
            crms = [build_crm(model) for model in model_names]
        except Exception as err:
            print(err)
            st.error("Could not load selected model: {}".format(err))

    image = st.file_uploader("Upload image")

    with st.spinner('Evaluating image....'):
        if image:
            with open("app/temp.png", "wb") as f:
                f.write(image.getbuffer())

            img, output, fig, bbox = generate_ensemble_crm(
                crms, 'app/temp.png', thresh)

            visualization = st.selectbox('Select the visualization mode', [
                'combined', 'class based'])

            top = decode_predictions(
                crms[0].classes, output[0], crms[0].num_classes)

            if visualization == 'combined':
                for c, i, p in top:
                    if p > 0.3:
                        st.write('{:15s}({}) {:f}'.format(c, i, p))

                st.subheader("Combined CRM Plot")
                st.write(fig)
            else:
                st.subheader("Class based CRM Plot")
                pred_thresh = st.slider('Prediction threshold', 0.0, 1.0, 0.4)

                boxes = []
                for c, i, p in top:
                    if p > pred_thresh:
                        original, img, output, fig, bbox = generate_ensemble_crm_class(
                        crms, 'app/temp.png', thresh, class_idx=i)
                        st.write('{:15s}({}) {:f}'.format(c, i, p))
                        st.write(fig)
                        newboxes = bbox.tolist()
                        newboxes = [(i, box) for box in newboxes]
                        boxes = boxes + newboxes

                fig = plt.figure(figsize=(6, 4))
                ax = plt.subplot(111)
                plt.imshow(original)
                plt.axis('off')

                for box in boxes:
                    i = box[0]
                    bbox = box[1]
                    ys = bbox[1]
                    xs = bbox[0]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]

                    if w > 25 and h > 25:
                        rect = patches.Rectangle((xs, ys), w, h, linewidth=1,
                                                edgecolor=colors[i], facecolor='none')

                        # Add the patch to the Axes
                        ax.add_patch(rect)
                        ax.annotate(crms[0].classes[i], (xs, ys), color='white',
                                    fontsize=6, ha='left', va='bottom')
                        for c, j, p in top[:3]:
                            if i == j:
                                prob = p
                        ax.annotate("{:.2f}".format(prob), (xs + w, ys), color='white',
                                    fontsize=6, ha='right', va='bottom')

                st.write(fig)
