import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# LOAD MODEL & FEATURES
# ----------------------------
dropout_model = joblib.load('D:/Laskar AI/Code/Model/rf_dropout_model.joblib')
graduated_model = joblib.load('D:/Laskar AI/Code/Model/rf_graduated_model.joblib')
with open('D:/Laskar AI/Code/Model/top_features.json') as f:
    top_features = json.load(f)

# Reordered manually for better UX
ordered_features = [
    'Admission_grade', 'Previous_qualification_grade', 'Age_at_enrollment',
    'Course', 'Application_mode', 'Tuition_fees_up_to_date', 'Scholarship_holder',
    'Fathers_occupation', 'Mothers_occupation', 'Fathers_qualification', 'Mothers_qualification',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'GDP', 'Unemployment_rate', 'Inflation_rate'
]

def categorize_dropout(score):
    if score >= 0.6:
        return 'Likely Dropout'
    elif score <= 0.2:
        return 'Likely Graduated'
    else:
        return 'Uncertain'

def categorize_graduated(score):
    if score >= 0.6:
        return 'Likely Graduated'
    elif score <= 0.2:
        return 'Likely Dropout'
    else:
        return 'Uncertain'

# ----------------------------
# STREAMLIT SETUP
# ----------------------------
st.set_page_config(page_title="Jaya Jaya Institut - Skoring Mahasiswa", layout="wide")
st.title("🎓 Student Scoring System - Jaya Jaya Institut")

# Sidebar
st.sidebar.title("📋 Menus")

# Elemen di sidebar
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Manual Inputs"])

# ----------------------------
# DASHBOARD
# ----------------------------
if page == "Dashboard":
    df = pd.read_csv("D:/Laskar AI/Code/student_performance.csv")
    df_enrolled = df[df['Status'] == 'Enrolled'].copy()

    X_enrolled = df_enrolled[top_features]
    dropout_scores = dropout_model.predict_proba(X_enrolled)[:, 1]
    graduated_scores = graduated_model.predict_proba(X_enrolled)[:, 1]

    df_enrolled['DropoutRiskScore'] = dropout_scores
    df_enrolled['GraduationScore'] = graduated_scores

    df_enrolled['DropoutRiskLevel'] = df_enrolled['DropoutRiskScore'].apply(categorize_dropout)
    df_enrolled['GraduationPrediction'] = df_enrolled['GraduationScore'].apply(categorize_graduated)

    # Gabungan prediksi dari kedua model
    def ensembled_prediction(d, g):
        if d >= 0.6 and g <= 0.2:
            return "Likely Dropout"
        elif d <= 0.2 and g >= 0.6:
            return "Likely Graduated"
        else:
            return "Uncertain"

    # Kategori final berdasarkan gabungan dua model
    df_enrolled['FinalRiskCategory'] = df_enrolled.apply(
        lambda row: ensembled_prediction(row['DropoutRiskScore'], row['GraduationScore']), axis=1
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Total Enrolled", len(df_enrolled))
    col2.metric("⚠️ Likely Dropout", (df_enrolled['DropoutRiskLevel'] == 'Likely Dropout').sum())
    col3.metric("🎯 Likely Graduated", (df_enrolled['GraduationPrediction'] == 'Likely Graduated').sum())

    st.markdown("## 📊 Dropout Risk Visualization")
    
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dropout score
    sns.histplot(df_enrolled['DropoutRiskScore'], 
                bins=30, kde=True, 
                color='salmon', 
                label='Dropout Score', 
                alpha=0.6, linewidth=0)

    # Graduation score
    sns.histplot(df_enrolled['GraduationScore'], 
                bins=30, kde=True, 
                color='skyblue', 
                label='Graduation Score', 
                alpha=0.6, linewidth=0)

    ax.set_title("Dropout & Graduation Risk Score Distribution", fontsize=12)
    ax.set_xlabel("Probability Score", fontsize=10)
    ax.set_ylabel("Number of Students", fontsize=10)

    # Grid & legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Score Type", fontsize=8, title_fontsize=10)

    # Format axis
    ax.tick_params(axis='both', labelsize=8)
    st.pyplot(fig)

    st.markdown("## 🧠 Important Features")
    importances = pd.Series(dropout_model.feature_importances_, index=top_features).sort_values(ascending=False)
    fig, ax = plt.subplots()
    importances.head(10).plot(kind='barh', ax=ax)
    plt.title("Top 10 Important Features", fontsize=12) 
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.invert_yaxis()
    st.pyplot(fig)

    # Features not in the top 10
    other_features = importances.iloc[10:].index.tolist()
    st.markdown("#### 📌 Other Features (outside the Top 10) Used in the Model:")
    st.markdown(", ".join(other_features))

    st.markdown("## 📋 Table of High Risk Students")
    high_risk_df = df_enrolled[df_enrolled['FinalRiskCategory'] == 'Likely Dropout']
    table_features = [
        'Admission_grade', 'Previous_qualification_grade',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    ]
    st.dataframe(high_risk_df[['DropoutRiskScore', 'GraduationScore', 'FinalRiskCategory'] + table_features])

    st.download_button(
        label="⬇️ Download High Risk Student Data",
        data=high_risk_df.to_csv(index=False),
        file_name="student_high_risk.csv",
        mime="text/csv"
    )

# ----------------------------
# INPUT MANUAL PAGE
# ----------------------------
elif page == "Manual Inputs":
    st.markdown("### 📥 Dropout Risk Prediction - Manual Input")

    occupation_mapping = {
        "Select ... ":None,
        "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1,
        "Specialists in Intellectual and Scientific Activities": 2,
        "Health professionals": 122,
        "Teachers": 123,
        "Specialists in information and communication technologies (ICT)": 125,
        "Intermediate Level Technicians and Professions": 3,
        "Intermediate level science and engineering technicians and professions": 131,
        "Technicians and professionals, of intermediate level of health": 132,
        "Intermediate level technicians from legal, social, sports, cultural and similar services": 134,
        "Administrative staff": 4,
        "Office workers, secretaries in general and data processing operators": 141,
        "Data, accounting, statistical, financial services and registry-related operators": 143,
        "Other administrative support staff": 144,
        "Personal Services, Security and Safety Workers and Sellers": 5,
        "Personal service workers": 151,
        "Sellers": 152,
        "Personal care workers and the like": 153,
        "Skilled Workers in Industry, Construction and Craftsmen": 7,
        "Skilled construction workers and the like, except electricians": 171,
        "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like": 173,
        "Workers in food processing, woodworking, clothing and other industries and crafts": 175,
        "Installation and Machine Operators and Assembly Workers": 8,
        "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6,
        "Unskilled Workers": 9,
        "Cleaning workers": 191,
        "Unskilled workers in agriculture, animal production, fisheries and forestry": 192,
        "Unskilled workers in extractive industry, construction, manufacturing and transport": 193,
        "Meal preparation assistants": 194,
        "Armed Forces Professions": 10,
        "Student": 0,
        "Other Situation": 90,
        "Others": 99
    }

    qualification_mapping={
        "Select ... ":None,
        "Can't read or write": 35,
        "Can read without having a 4th year of schooling": 36,
        "Basic education 1st cycle (4th/5th year) or equiv.": 37,
        "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38,
        "7th Year (Old)": 11,
        "7th year of schooling": 26,
        "8th year of schooling": 30,
        "9th Year of Schooling - Not Completed": 29,
        "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19,
        "10th Year of Schooling": 14,
        "Other - 11th Year of Schooling": 12,
        "11th Year of Schooling - Not Completed": 10,
        "2nd cycle of the general high school course": 27,
        "12th Year of Schooling - Not Completed": 9,
        "Secondary Education - 12th Year of Schooling or Eq.": 1,
        "General commerce course": 18,
        "Technical-professional course": 22,
        "Technological specialization course": 39,
        "Professional higher technical course": 42,
        "Specialized higher studies course": 41,
        "Higher education - degree (1st cycle)": 40,
        "Higher Education - Bachelor's Degree": 2,
        "Higher Education - Degree": 3,
        "Frequency of Higher Education": 6,
        "Higher Education - Master's": 4,
        "Higher Education - Master (2nd cycle)": 43,
        "Higher Education - Doctorate": 5,
        "Higher Education - Doctorate (3rd cycle)": 44,
        "Unknown": 34
    }

    course_mapping={
        "Select ... ":None,
        "Agronomy": 9003,
        "Biofuel Production Technologies": 33,
        "Equinculture": 9130,
        "Veterinary Nursing": 9085,
        "Informatics Engineering": 9119,
        "Animation and Multimedia Design": 171,
        "Communication Design": 9070,
        "Journalism and Communication": 9773,
        "Advertising and Marketing Management": 9670,
        "Basic Education": 9853,
        "Nursing": 9500,
        "Oral Hygiene": 9556,
        "Management": 9147,
        "Management (evening attendance)": 9991,
        "Tourism": 9254,
        "Social Service": 9238,
        "Social Service (evening attendance)": 8014
    }

    application_mode_mapping={
        "Select ... ":None,
        "1st phase - general contingent": 1,
        "1st phase - special contingent (Azores Island)": 5,
        "1st phase - special contingent (Madeira Island)": 16,
        "2nd phase - general contingent": 17,
        "3rd phase - general contingent": 18,
        "International student (bachelor)": 15,
        "Over 23 years old": 39,
        "Holders of other higher courses": 7,
        "Technological specialization diploma holders": 44,
        "Short cycle diploma holders": 53,
        "Change of course": 43,
        "Change of institution/course": 51,
        "Change of institution/course (International)": 57,
        "Transfer": 42,
        "Ordinance No. 612/93": 2,
        "Ordinance No. 854-B/99": 10,
        "Ordinance No. 533-A/99, item b2) (Different Plan)": 26,
        "Ordinance No. 533-A/99, item b3 (Other Institution)": 27
    }

    input_data = {}
    errors = []
    validation_error = False

    for feature in ordered_features:
        if feature == "Fathers_occupation":
            selected = st.selectbox("Fathers Occupation", list(occupation_mapping.keys()), key=feature)
            if selected:
                input_data[feature] = occupation_mapping[selected]
            else:
                validation_error = True
        elif feature == "Mothers_occupation":
            selected = st.selectbox("Mothers Occupation", list(occupation_mapping.keys()), key=feature)
            if selected:
                input_data[feature] = occupation_mapping[selected]
            else:
                validation_error = True
        elif feature == "Mothers_qualification":
            selected = st.selectbox("Mothers Qualification", list(qualification_mapping.keys()), key=feature)
            if selected:
                input_data[feature] = qualification_mapping[selected]
            else:
                validation_error = True
        elif feature == "Fathers_qualification":
            selected = st.selectbox("Fathers Qualification", list(qualification_mapping.keys()), key=feature)
            if selected:
                input_data[feature] = qualification_mapping[selected]
            else:
                validation_error = True
        elif feature == "Course":
            selected = st.selectbox("Course", list(course_mapping.keys()), key=feature)
            if selected:
                input_data[feature] = course_mapping[selected]
            else:
                validation_error = True
        elif feature == "Application_mode":
            selected = st.selectbox("Application Mode", list(application_mode_mapping.keys()), key=feature)
            if selected:
                input_data[feature] = application_mode_mapping[selected]
            else:
                validation_error = True
        elif feature in [
            'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
            'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_enrolled',
            'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved'
        ]:
            input_data[feature] = st.number_input(
            feature.replace('_', ' '),
            min_value=0,
            value=0,
            step=1)
        elif feature in ["Tuition_fees_up_to_date", "Scholarship_holder"]:
            selected = st.radio(f"{feature.replace('_', ' ')}", ["Yes", "No"], key=feature)
            input_data[feature] = 1 if selected == "Yes" else 0
        elif feature == "Admission_grade":
            input_data[feature] = st.number_input(
                "Admission Grade", min_value=0.0, max_value=200.0, value=0.0)
        elif feature == "Previous_qualification_grade":
            input_data[feature] = st.number_input(
                "Previous Qualification Grade", min_value=0.0, max_value=200.0, value=0.0)
        elif feature == "Curricular_units_1st_sem_grade":
            input_data[feature] = st.number_input(
                "Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0, value=0.0)
        elif feature == "Curricular_units_2nd_sem_grade":
            input_data[feature] = st.number_input(
                "Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0, value=0.0)
        elif feature == "Age_at_enrollment":
            input_data[feature] = st.number_input(
                "Age at Enrollment (Years)", min_value=15, max_value=100, value=18, step=1, format="%d"
    )

        else:
            input_data[feature] = st.number_input(f"{feature.replace('_', ' ')}", value=0.0, key=feature)
    if st.button("Prediction"):
        if validation_error:
            st.error("❗ Please fill in all available options before predicting.")
        else:
            input_df = pd.DataFrame([input_data])[top_features]
            dropout_prob = dropout_model.predict_proba(input_df)[0][1]
            graduated_prob = graduated_model.predict_proba(input_df)[0][1]

            st.metric("Dropout Risk Score", f"{dropout_prob:.2f}")
            st.metric("Graduation Score", f"{graduated_prob:.2f}")

            if dropout_prob >= 0.6:
                st.error("🚨 Students at High Risk of Dropout. Special Treatment Needed!")
            elif dropout_prob >= 0.2:
                st.warning("⚠️ Medium Risk Students. Monitoring Required.")
            else:
                st.success("✅ Students Tend to be Safe from Dropout Risk.")

