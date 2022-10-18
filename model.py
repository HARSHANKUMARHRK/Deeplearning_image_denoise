import streamlit as st
import cv2 
import numpy as np    
import tensorflow as tf
import time
import os
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from PIL import Image
import keras
import json
from streamlit_lottie import st_lottie 
import requests
def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('About the App','Denoise Image')
        )


   # readme_text = st.markdown(get_file_content_as_string("README.md"))
    
    if selected_box == 'About the App':
        st.title("UNDER WATER PHOTOGRAPHY NOISE CANCELLATION USING AI & DEEP LEARNING")
        st.sidebar.success('To try by yourself select "Evaluate the model".')
        st.subheader("What is noise ?")
        st.write("An additional unnecessary pixel values are added to a image causing the loss of information.The noise can be originated by many ways such as while capturing images in low-light situations, damage of electric circuits due to heat, sensor illumination levels of a digital camera or due to the faulty memory locations in hardware or bit errors in transmission of data over long distances. It is essential to remove the noise and recover the original image from the degraded images where getting the original image is important for robust performance or in cases where filling the missing information is very useful like the astronomical images that are taken from very distant objects.")
        st.subheader("Solution :")
        st.subheader("Using DNCNN model :")
        st.write("Therea are many deep learning model that can be used for completing this task of image denoising. Now we will use Deep Convulutional Neural Network model (DnCNN)")
        st.subheader("Architecture of the model :")
        st.image("img.jpeg")
        st.write("Given a noisy image 'y' the model will predict residual image 'R' and clean image 'x' can be obtained by "
        
 "x=y-R")
        st.subheader("Using RIDNET model :")
        st.write("Real Image Denoising with Feature Attention.")
        st.subheader("Architecture of the model:")
        st.image("img2.jpeg")
        st.write("This model is composed of three main modules i.e. feature extraction, feature learning residual on the residual module, and reconstruction, as shown in Figure .")
        st.subheader("Dataset:")
        st.write("We will be using publicly avaliable image and modify it according to our requirement")
        st.write("dataset : https://github.com/BIDS/BSDS500")
        st.write("This Dataset is provided by Berkeley University of California which contains 500 natural images.")
        st.write("Now we create 85600 patches of size 40 x 40 that was created from 400 train images and 21400 patches of size 40 x 40 that was created from 100 test images")
        st.subheader("Training:")
        st.write("Model has been train for 30 epochs with Adam optimizer of learning rate=0.001 and with learning rate decay of 5% per epoch .Mean Squared Error is used as loss function for DNCNN model and Mean Absolute Error for RIDNET.")
        st.subheader("Results :")
        st.write("This results are from DNCNN model.")
        st.write("For an noisy image with psnr of 20.530 obtained denoised image which has psnr of 31.193")
        st.write("Image showing the comparision of ground truth, noisy image and denoised image.")
        st.image("img3.jpeg")
        st.write("Image showing patch wise noisy and denoised images.")
        st.image("img4.jpeg")
        st.write("Below plot shows the model performance on different noise levels")
        st.image("img5.jpeg")
        st.subheader("Comparision of the models :")
        st.write("Tabulating the results(PSNR in db) from the models with different noise level")
        st.image("img6.jpeg")

    if selected_box == 'Denoise Image':

        models()

    
def models():
    st.title("Image Denoising using Deep Learning")
    st.subheader('You can predict on sample images or you can upload a noisy image and get its denoised output.')
    
    selection=st.selectbox("Choose how to load image",["<Select>","Upload an Image","Predict on sample Images"])
    
    if selection=="Upload an Image":
        image = st.file_uploader('Upload the image below')
        predict_button = st.button('Predict on uploaded image')
        if predict_button:
            if image is not None:
                file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
                nsy_img = cv2.imdecode(file_bytes, 1)
                prediction(nsy_img)
            else:
                st.text('Please upload the image')    
    
    if selection=='Predict on sample Images':
        option = st.selectbox('Select a sample image',('<select>','Toy car','Vegetables','Gadget desk','Scrabble board','Shoes','Door','Chess board','A note'))
        if option=='<select>':
            pass
        else:
            path = os.path.join(os.getcwd(),'NoisyImage/')
            nsy_img = cv2.imread(path+option+'.jpg')
            prediction(nsy_img)
            
def patches(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
  return patches

#st.cache
def get_model():
    RIDNet=tf.keras.models.load_model('RIDNet.h5')
    return RIDNet

def prediction(img):
    state = st.text('\n Please wait while the model denoise the image.....')
    progress_bar = st.progress(0)
    start = time.time()
    model = get_model()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nsy_img = cv2.resize(img,(1024,1024))
    nsy_img = nsy_img.astype("float32") / 255.0

    img_patches = patches(nsy_img,256)
    progress_bar.progress(30)
    nsy=[]
    for i in range(4):
        for j in range(4):
            nsy.append(img_patches[i][j][0])
    nsy = np.array(nsy)
    
    pred_img = model.predict(nsy)
    progress_bar.progress(70)
    pred_img = np.reshape(pred_img,(4,4,1,256,256,3))
    pred_img = unpatchify(pred_img, nsy_img.shape)
    end = time.time()
     
    img = cv2.resize(img,(512,512))
    pred_img = cv2.resize(pred_img,(512,512))
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(img) 
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].title.set_text("Noisy Image")
    
    ax[1].imshow(pred_img) 
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].title.set_text("Predicted Image")
    
    st.pyplot(fig)
    progress_bar.progress(100)
    st.write('Time taken for prediction :', str(round(end-start,3))+' seconds')
    progress_bar.empty()
    state.text('\n Completed!')

with st.sidebar:
    def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_idnzhjhq.json")
st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="high", # medium ; high
    
    height=None,
    width=None,
    key=None,
)
    
if __name__ == "__main__":
    main()   
