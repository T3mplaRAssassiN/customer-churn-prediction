import streamlit as st
import pandas as pd
import joblib
import numpy as np

# تنظیمات صفحه
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# عنوان داشبورد
st.title("📊 پیش‌بینی ریزش مشتریان (Customer Churn)")
st.markdown("این داشبورد برای پیش‌بینی احتمال ریزش مشتریان یک شرکت مخابراتی طراحی شده است.")

# لود کردن مدل
@st.cache_resource
def load_model():
    return joblib.load('models/churn_model.pkl')

model = load_model()

st.sidebar.header("مشخصات مشتری را وارد کنید:")

# گرفتن ورودی از کاربر در سایدبار
tenure = st.sidebar.slider("مدت زمان حضور (ماه):", 0, 72, 12)
monthly_charges = st.sidebar.number_input("هزینه ماهانه ($):", min_value=0.0, value=50.0, step=1.0)
total_charges = st.sidebar.number_input("کل هزینه‌ها ($):", min_value=0.0, value=500.0, step=10.0)
contract = st.sidebar.selectbox("نوع قرارداد:", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("سرویس اینترنت:", ["DSL", "Fiber optic", "No"])

# دکمه پیش‌بینی
if st.sidebar.button("پیش‌بینی کن"):

    # ⚠️ نکته مهم: برای سادگی در این نسخه، ما فرض می‌کنیم بقیه ستون‌ها مقادیر پیش‌فرض دارند.
    # در یک پروژه واقعی باید تمام ۳۰ ستون را از کاربر بپرسیم.

    # ساخت یک دیتافریم خالی با ستون‌های مدل
    # (چون دیتای X_train توی این فایل نیست، از مدل می‌پرسیم چه ستون‌هایی می‌خواد)
    expected_features = model.feature_names_in_
    input_df = pd.DataFrame(0, index=[0], columns=expected_features)

    # پر کردن مقادیری که کاربر وارد کرده
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges

    # تنظیم ستون‌های One-Hot
    if 'Contract_One year' in input_df.columns:
        input_df['Contract_One year'] = 1 if contract == "One year" else 0
    if 'Contract_Two year' in input_df.columns:
        input_df['Contract_Two year'] = 1 if contract == "Two year" else 0
    if 'InternetService_Fiber optic' in input_df.columns:
        input_df['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
    if 'InternetService_No' in input_df.columns:
        input_df['InternetService_No'] = 1 if internet_service == "No" else 0

    # پیش‌بینی
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # نمایش نتیجه
    st.subheader("نتیجه پیش‌بینی:")
    if prediction == 1:
        st.error(f"⚠️ این مشتری *احتمالاً ریزش می‌کند (احتمال: {probability:.1%})")
        st.info("پیشنهاد: پیشنهاد قرارداد بلندمدت یا تخفیف ویژه به این مشتری بدهید.")
    else:
        st.success(f"✅ این مشتری می‌ماند* (احتمال ریزش: {probability:.1%})")