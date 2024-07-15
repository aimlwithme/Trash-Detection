#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import requests
import streamlit as st
from streamlit_image_select import image_select

def create_array():
    directory =  os.path.join(os.getcwd(), "images")
    images = []
    image_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            images.append(os.path.join(directory, filename))
            image_names.append(filename.split(".")[0])
    image_names.sort()
    return images, image_names
def main():
    st.set_page_config(layout="wide")
    st.title("Environmental Waste Detection Through AI")
    image = None
    paths, image_names = create_array()
    tab1, tab2, tab3 = st.tabs(["Model", "Find your own image", "Help & Contact"])
    with tab1:
        with st.expander("Try some examples", expanded = False):
            image_index = image_select(
            label="Select an image to test with",
            images = paths,
            return_value = "index",
            captions= image_names,
            use_container_width = False,
            key = "selector")
        img_file = st.sidebar.file_uploader(label = "Upload an image to recieve output", type=["jpg", "png", "jpeg"], key = 'uploader')
        confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0, 100, 50, 1)
        overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0, 100, 50, 1)
        col1, col2 = st.columns(2)
        if st.session_state["uploader"] is not None:
            image = Image.open(img_file)
            with st.spinner("Uploading..."):
                st.success(f"{img_file.name} has been uploaded and processed successfully!")
        elif st.session_state["selector"] is not None:
            image = Image.open(paths[image_index])
        if image is not None:
            col1.caption("Input")
            col1.image(image, use_column_width=True)
            parts = []
            url_base = 'https://detect.roboflow.com/'
            endpoint = 'waste-detection-vnfx1/2'
            access_token = f'?api_key={st.secrets["api_key"]}'
            format = '&format=json'
            overlap = f'&overlap={overlap_threshold}'
            confidence = f'&confidence={confidence_threshold}'
            stroke='&stroke=5'
            parts.append(url_base)
            parts.append(endpoint)
            parts.append(access_token)
            parts.append(format)
            parts.append(overlap)
            parts.append(confidence)
            parts.append(stroke)
            url = ''.join(parts)
            buffered = io.BytesIO()
            image = image.convert("RGB")
            image.save(buffered, quality=90, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            img_str = img_str.decode("ascii")
            headers = {'accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'}
            r = requests.post(url, data=img_str, headers=headers)
            preds = r.json()
            detections = preds['predictions']
            draw = ImageDraw.Draw(image, "RGBA")
            for box in detections:
                color = "#2C67EC"
                x1 = box['x'] - box['width'] / 2
                x2 = box['x'] + box['width'] / 2
                y1 = box['y'] - box['height'] / 2
                y2 = box['y'] + box['height'] / 2
                draw.rectangle([x1, y1, x2, y2], fill = (44, 103, 236, 50), outline=color, width=6)
                font = ImageFont.truetype("OpenSans-Regular.ttf", 25)
                text = f"{round(100 * box['confidence'], 0)}% "
                text_size = font.getsize(text)
                button_size = (text_size[0]+30, text_size[1]+20)
                button_img = Image.new('RGB', button_size, color)
                button_draw = ImageDraw.Draw(button_img)
                button_draw.text((10, 5), text, font=font, stroke_width = 1, fill=(255,255,255))
                image.paste(button_img, (int(x1), int(y1)))
            image = image.convert("RGB")
            col2.caption("Prediction Output")
            col2.image(image, use_column_width=True)
            st.divider()
            preds_dim = []
            for result in detections:
                preds_dim.append((result['width'], result['height']))
            image_dim = (preds['image']['width'], preds['image']['height'])
            area_covered = 0
            for pred in preds_dim:
                area_covered += int(pred[0])*int(pred[1])
            area_total = int(image_dim[0])*int(image_dim[1])
            time = preds['time']
            st.markdown("### Fun facts about the prediction above:")
            st.markdown(f'Proportion of image predicted as waste: {round(100*area_covered/area_total, 0)}%')
            st.markdown(f'Time it took for the model to predict: {int(round(1000 * time, 0))} millisecond(s)')
        else:
            st.divider()
            st.markdown("<h4 style='text-align: center;'>Please upload an image or select an example above to begin</h4>", unsafe_allow_html=True)
            st.divider()

    with tab2:
        st.markdown("# Instructions for finding your own image")
        st.markdown("""
                    ### Easier Method:

                    1. Go to [**Google Maps**](https://www.google.com/maps)
                    2. Select the satellite layer option at the bottom right (on desktop)
                    3. Search for a location to snap an image from
                    4. Zoom into the location enough to match the scale of the example images
                    5. Take a screenshot of the screen
                    6. Crop out the website components so you are left with the satellite image
                    7. Save for uploading purposes!  
                    """)
        st.markdown("""
                    ### In-Depth Method:

                    1. Install [**Google Earth Pro**](https://www.google.com/earth/about/versions)
                    2. Open the application and list location coordinates or an address in the search bar 
                        - You can look around using the zoom slider and movement joysticks at the top right of the 3D viewer
                    """)
        st.image("GEP Toolbar.jpg")
        st.markdown("""
                    3. Select the clock icon and select a date between 2018 and 2019 that has a relatively high resolution satellite image capture for best accuracy
                    4. Zoom in to an altitude of around 200 feet (You can see the altitude at the bottom of the 3D viewer)
                    5. Uncheck all the layers (except Terrain) in the bottom right window (Layers panel)
                    6. Save the image for uploading purposes!
                        - Take a screenshot of the terrain and crop out the surrounding elements
                        Or alternatively
                        - Select the save icon in the toolbar (3rd from the left), and download an approximately 1920 x 1080 image
                    """)


    with tab3:
        st.markdown("# Common issues")
        st.markdown("""
                    - ### Selecting any of the example images does not work!
                        - If you have uploaded an image, click the X to delete the file stored. This should get rid of that issue.
                    """
                    )
        st.divider()
        st.markdown("# Contact")
        st.markdown('Let me know if you have any other issues with the website you wish to report or anything else you want to communicate with me (Suggestions are welcome!)')
        st.markdown('Email: <a href="mailto:shubhadipbiswas@kpmg.com">shubhadipbiswas@kpmg.com</a>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
