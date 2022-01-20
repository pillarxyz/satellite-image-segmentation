import gradio as gr
import numpy as np
import skimage.segmentation as seg
from skimage.io import imshow
from skimage import morphology as morph
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.future import graph
from skimage.feature import canny
from skimage import util
from scipy import ndimage as ndi

method_list = ["Mean", "Local", "Otsu","HSV", "Sobel", "Canny", "SLIC"]

def segment(image, method):
    gimage = color.rgb2gray(image)
    if method == method_list[0]:
        threshold = filters.threshold_mean(gimage)
        segmented = gimage >= threshold
        segmented = segmented.astype('float64')
    
    if method == method_list[1]:
        threshold = filters.threshold_local(gimage, block_size = 15)
        segmented = gimage > threshold
        segmented = segmented.astype('float64')
    
    if method == method_list[2]:
        threshold = filters.threshold_otsu(gimage)
        segmented = gimage >= threshold
        segmented = segmented.astype('float64')
    
    if method == method_list[3]:
        segmented = color.rgb2hsv(image)
     
    if method == method_list[4]:
        segmented = filters.sobel(image)
    
    if method == method_list[5]:
        segmented = ndi.binary_fill_holes(canny(gimage).astype('uint8')).astype('float64')
    
    if method == method_list[6]:
        mask = morph.remove_small_holes(morph.remove_small_objects(gimage < 0.7, 300), 500)
        segmented = seg.slic(image, n_segments=100, mask=mask, start_label=1)
        segmented.astype('float64')
    return segmented

input_image = gr.inputs.Image()
input_method = gr.inputs.Dropdown(method_list, type="value", label="method")

output_image = gr.outputs.Image()

app = gr.Interface(segment, inputs = [input_image, input_method] , outputs = output_image, theme = 'dark-huggingface', live = True, title = "TDM Project Demo")

app.launch(share = True, inbrowser = True)
