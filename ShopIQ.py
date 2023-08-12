import tensorflow
import requests
from sklearn.neighbors import NearestNeighbors
import keras
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
import streamlit as st
import pickle
import numpy as np
from numpy.linalg import norm
from PIL import Image, ImageOps
import streamlit as st

@st.cache_resource
def path_embeddings():
    embeddings_url = 'https://media.githubusercontent.com/media/Ketan3102/ShopIQ/main/embeddings.pkl'
    image_paths_url = 'https://media.githubusercontent.com/media/Ketan3102/ShopIQ/main/image_paths.pkl'
    embeddings = np.array(pickle.loads(requests.get(embeddings_url).content))
    image_paths = pickle.loads(requests.get(image_paths_url).content)
    return embeddings, image_paths

@st.cache_resource
def recomm_model():
    model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
    model.trainable=False
    model=keras.Sequential(model)
    model.add(GlobalAveragePooling2D())
    return model

def recommendation_emb(img,model):
    img_array=image.img_to_array(img)
    expanded_img=np.expand_dims(img_array,axis=0)
    preprocessed_img= preprocess_input(expanded_img)
    embed=model.predict(preprocessed_img)
    normalized_embedding=np.array(embed.flatten()/norm(embed.flatten()))
    normalized_embedding=normalized_embedding.reshape(1,-1)
    return normalized_embedding

def main():
    st.set_page_config(page_title='ShopIQ',page_icon='🛒')
    st.title('ShopIQ: Your Shopping Buddy')
    col_upload,col_dropdown=st.columns(2)
    with col_upload:
        data=st.file_uploader("Insert Your Image")
    display_img=None
    if data:
        embeddings,image_paths=path_embeddings()
        model=recomm_model()
        display_img=Image.open(data).resize((224,224))
        st.image(display_img)
        recommendation_embedding=recommendation_emb(display_img,model)
        neighbours=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
        neighbours.fit(embeddings)
        dist,index=neighbours.kneighbors(recommendation_embedding)
        st.subheader("You may also like:")
        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            st.image(image_paths[index[0][0]])
        with col2:
            st.image(image_paths[index[0][1]])
        with col3:
            st.image(image_paths[index[0][2]])
        with col4:
            st.image(image_paths[index[0][3]])
        with col5:
            st.image(image_paths[index[0][4]])

    else:
        st.warning('No image')
    
    


if __name__=="__main__":
    main()