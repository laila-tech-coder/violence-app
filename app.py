import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("Violence Detection App")

# دالة ذكية للتحميل تتخطى أخطاء batch_shape و optional
@st.cache_resource
def load_violence_model():
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        st.error("ملف model.h5 غير موجود في المستودع!")
        return None
    
    try:
        # المحاولة الأولى: التحميل العادي
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        try:
            # المحاولة الثانية: التحميل مع تخصيص الطبقات (حل مشكلة Keras 3)
            from tensorflow.keras.layers import InputLayer
            custom_objects = {'InputLayer': InputLayer}
            return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        except Exception as e:
            st.error(f"فشل التحميل النهائي: {e}")
            return None

model = load_violence_model()
class_names = ["Non-Violence", "Violence"]

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image Uploaded", use_container_width=True)
    
    # المعالجة المسبقة
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model:
        prediction = model.predict(img_array)
        result = np.argmax(prediction)
        label = class_names[result]
        
        if label == "Violence":
            st.error(f"Result: {label}")
        else:
            st.success(f"Result: {label}")
