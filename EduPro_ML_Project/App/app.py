import joblib
model = joblib.load("simple_revenue_model.pkl")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page title
st.title("EduPro Predictive Analytics Dashboard")
st.write("Course Demand and Revenue Forecasting")

# Load dataset
master_df = pd.read_csv(r"C:\Users\prach\Python\EduPro_ML_Project\Notebook\master_df.csv")

# Sidebar
page = st.sidebar.selectbox(
    "Choose Page",
    ["Overview", "Prediction"]
)

# Overview page
if page == "Overview":
    st.subheader("Dataset Preview")
    st.dataframe(master_df.head())

    total_courses = master_df["CourseID"].nunique()
    total_revenue = master_df["Total_Revenue"].sum()
    avg_rating = master_df["CourseRating"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Courses", total_courses)
    col2.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col3.metric("Average Rating", f"{avg_rating:.2f}")

    st.subheader("Revenue by Category")

    category_revenue = master_df.groupby("CourseCategory")["Total_Revenue"].sum()

    fig, ax = plt.subplots(figsize=(8,4))
    category_revenue.sort_values().plot(kind="barh", ax=ax)

    st.pyplot(fig)

# Prediction page
elif page == "Prediction":
    st.subheader("Course Prediction")

    price = st.number_input("Course Price", 0, 50000, 5000)
    duration = st.number_input("Course Duration", 1, 500, 30)
    rating = st.slider("Course Rating", 1.0, 5.0, 4.0)
    teacher_exp = st.number_input("Teacher Experience", 0, 30, 5)

    if st.button("Predict"):

        input_data = [[price, duration, rating, teacher_exp]]

        prediction = model.predict(input_data)[0]

        st.success("Prediction Complete!")

        st.metric(
            "Predicted Revenue",
            f"₹{prediction:,.0f}"
        )
