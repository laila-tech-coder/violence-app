import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# إعداد واجهة التطبيق
st.set_page_config(page_title="Violence Detection App", layout="centered")
st.title("Violence Detection App")

# دالة لتحميل الموديل مع معالجة أخطاء الإصدارات (Keras 3 vs Keras 2)
@st.cache_resource
def load_my_model():
    try:
        # الحل لمشكلة batch_shape هو تحميل الموديل بدون تعقيدات الـ Compile
        return tf.keras.models.load_model('model.h5', compile=False)
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
        return None

model = load_my_model()

class_names = ["Non-Violence", "Violence"]

# رفع الملفات
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # معالجة الصورة
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # عرض الصورة للمستخدم
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # تجهيز الصورة للموديل
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # التطبيع (Normalization) مهم لـ TensorFlow
    img_array = np.expand_dims(img_array, axis=0)

    if model is not None:
        with st.spinner('Analysing...'):
            prediction = model.predict(img_array)
            result = np.argmax(prediction)
            
            # عرض النتيجة بشكل جذاب
            label = class_names[result]
            if label == "Violence":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")
    else:
        st.error("الموديل غير متاح حالياً، تأكد من وجود ملف model.h5")
