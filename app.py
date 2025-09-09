import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('insurance_fraud_model.sav', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'insurance_fraud_model.sav' not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Title and description
st.title("🔍 Insurance Fraud Detection System")
st.markdown("---")
st.write("This application uses machine learning to predict the likelihood of insurance fraud based on claim characteristics.")

# Load model
model = load_model()

if model is not None:
    # Create input form
    st.header("📝 Enter Claim Information")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Date & Time Information")
        
        # Month
        month = st.selectbox(
            "Month",
            ['Dec', 'Jan', 'Oct', 'Jun', 'Feb', 'Nov', 'Apr', 'Mar', 'Aug', 'Jul', 'May', 'Sep']
        )
        
        # Day of Week
        day_of_week = st.selectbox(
            "Day of Week",
            ['Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday']
        )
        
        # Day of Week Claimed
        day_of_week_claimed = st.selectbox(
            "Day of Week Claimed",
            ['Tuesday', 'Monday', 'Thursday', 'Friday', 'Wednesday', 'Saturday', 'Sunday']
        )
        
        # Month Claimed
        month_claimed = st.selectbox(
            "Month Claimed",
            ['Jan', 'Nov', 'Jul', 'Feb', 'Mar', 'Dec', 'Apr', 'Aug', 'May', 'Jun', 'Sep', 'Oct']
        )
    
    with col2:
        st.subheader("🚗 Vehicle & Policy Information")
        
        # Make
        make = st.selectbox(
            "Vehicle Make",
            ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 
             'Mercury', 'Jaguar', 'Nissan', 'VW', 'Saab', 'Saturn', 'Porsche', 'BMW', 
             'Mercedes', 'Ferrari', 'Lexus']
        )
        
        # Vehicle Category
        vehicle_category = st.selectbox(
            "Vehicle Category",
            ['Sport', 'Utility', 'Sedan']
        )
        
        # Policy Type
        policy_type = st.selectbox(
            "Policy Type",
            ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Utility - All Perils',
             'Sedan - All Perils', 'Sedan - Collision', 'Utility - Collision', 'Utility - Liability',
             'Sport - All Perils']
        )
        
        # Base Policy
        base_policy = st.selectbox(
            "Base Policy",
            ['Liability', 'Collision', 'All Perils']
        )
        
        # Agent Type
        agent_type = st.selectbox(
            "Agent Type",
            ['External', 'Internal']
        )
    
    with col3:
        st.subheader("Personal Information")
        
        # Sex
        sex = st.selectbox(
            "Sex",
            ['Female', 'Male']
        )
        
        # Marital Status
        marital_status = st.selectbox(
            "Marital Status",
            ['Single', 'Married', 'Widow', 'Divorced']
        )
        
        # Accident Area
        accident_area = st.selectbox(
            "Accident Area",
            ['Urban', 'Rural']
        )
        
        # Fault
        fault = st.selectbox(
            "Fault",
            ['Policy Holder', 'Third Party']
        )
        
        # Police Report Filed
        police_report_filed = st.selectbox(
            "Police Report Filed",
            ['No', 'Yes']
        )
        
        # Witness Present
        witness_present = st.selectbox(
            "Witness Present",
            ['No', 'Yes']
        )
    
    # Numerical inputs
    st.header("Numerical Information")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Age (based on stats: mean=39.86, std=13.42, min=0, max=80)
        age = st.slider(
            "Age",
            min_value=0,
            max_value=80,
            value=40,
            help="Age of the policy holder"
        )
        
        # Policy Number (based on stats: mean=771, std=445, min=1, max=1542)
        policy_number = st.number_input(
            "Policy Number",
            min_value=1,
            max_value=1542,
            value=771,
            help="Unique policy identifier"
        )
        
        # Rep Number (based on stats: mean=8.48, std=4.6, min=1, max=16)
        rep_number = st.slider(
            "Rep Number",
            min_value=1,
            max_value=16,
            value=8,
            help="Representative number"
        )
    
    with col5:
        # Deductible (based on stats: mean=407, std=433, min=300, max=700)
        deductible = st.slider(
            "Deductible ($)",
            min_value=300,
            max_value=700,
            value=400,
            step=100,
            help="Insurance deductible amount"
        )
        
        # Driver Rating (based on stats: mean=2.49, std=1.19, min=1, max=4)
        driver_rating = st.slider(
            "Driver Rating",
            min_value=1,
            max_value=4,
            value=2,
            help="Driver safety rating"
        )
        
        # Year (based on stats: mean=1995, std=0.80, min=1994, max=1996)
        year = st.selectbox(
            "Year",
            [1994, 1995, 1996],
            index=1
        )
    
    with col6:
        # Vehicle Price ranges
        vehicle_price_ranges = [
            'more than 69000', '20000 to 29000', '30000 to 39000', 'less than 20000',
            '40000 to 59000', '60000 to 69000'
        ]
        vehicle_price = st.selectbox(
            "Vehicle Price Range",
            vehicle_price_ranges
        )
        
        # Days Policy Accident ranges
        days_policy_accident_ranges = [
            'more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'
        ]
        days_policy_accident = st.selectbox(
            "Days Policy Accident",
            days_policy_accident_ranges
        )
        
        # Days Policy Claim ranges
        days_policy_claim_ranges = [
            'more than 30', '15 to 30', '8 to 15', 'none'
        ]
        days_policy_claim = st.selectbox(
            "Days Policy Claim",
            days_policy_claim_ranges
        )
    
    # Additional categorical features
    st.header("Additional Information")
    
    col7, col8 = st.columns(2)
    
    with col7:
        # Past Number of Claims
        past_claims_options = ['none', '1', '2 to 4', 'more than 4']
        past_number_of_claims = st.selectbox(
            "Past Number of Claims",
            past_claims_options
        )
        
        # Age of Vehicle
        age_of_vehicle_options = ['3 years', '6 years', '7 years', 'more than 7', '5 years', 'new', '4 years', '2 years']
        age_of_vehicle = st.selectbox(
            "Age of Vehicle",
            age_of_vehicle_options
        )
        
        # Age of Policy Holder ranges
        age_of_policy_holder_ranges = [
            '26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25', '36 to 40',
            '16 to 17', 'over 65', '18 to 20'
        ]
        age_of_policy_holder = st.selectbox(
            "Age of Policy Holder Range",
            age_of_policy_holder_ranges
        )
    
    with col8:
        # Number of Supplements
        supplements_options = ['none', 'more than 5', '3 to 5', '1 to 2']
        number_of_supplements = st.selectbox(
            "Number of Supplements",
            supplements_options
        )
        
        # Address Change Claim
        address_change_options = ['1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months']
        address_change_claim = st.selectbox(
            "Address Change Claim",
            address_change_options
        )
        
        # Number of Cars
        number_of_cars_options = ['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8']
        number_of_cars = st.selectbox(
            "Number of Cars",
            number_of_cars_options
        )
    
    # Create prediction button
    st.markdown("---")
    
    if st.button("Predict Fraud", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'Month': month,
            'WeekOfMonth': 2.79,  # Using mean value from stats
            'DayOfWeek': day_of_week,
            'Make': make,
            'AccidentArea': accident_area,
            'DayOfWeekClaimed': day_of_week_claimed,
            'MonthClaimed': month_claimed,
            'WeekOfMonthClaimed': 2.69,  # Using mean value from stats
            'Sex': sex,
            'MaritalStatus': marital_status,
            'Age': age,
            'Fault': fault,
            'PolicyType': policy_type,
            'VehicleCategory': vehicle_category,
            'VehiclePrice': vehicle_price,
            'Days_Policy_Accident': days_policy_accident,
            'Days_Policy_Claim': days_policy_claim,
            'PastNumberOfClaims': past_number_of_claims,
            'AgeOfVehicle': age_of_vehicle,
            'AgeOfPolicyHolder': age_of_policy_holder,
            'PoliceReportFiled': police_report_filed,
            'WitnessPresent': witness_present,
            'AgentType': agent_type,
            'NumberOfSuppliments': number_of_supplements,
            'AddressChange_Claim': address_change_claim,
            'NumberOfCars': number_of_cars,
            'BasePolicy': base_policy,
            'PolicyNumber': policy_number,
            'RepNumber': rep_number,
            'Deductible': deductible,
            'DriverRating': driver_rating,
            'Year': year
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display results
            st.markdown("---")
            st.header("Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.error(" **FRAUD DETECTED**")
                    st.write("This claim has been flagged as potentially fraudulent.")
                else:
                    st.success(" **LEGITIMATE CLAIM**")
                    st.write("This claim appears to be legitimate.")
            
            with col_result2:
                st.write("**Prediction Confidence:**")
                fraud_prob = prediction_proba[1] * 100
                legitimate_prob = prediction_proba[0] * 100
                
                st.write(f"• Fraud Probability: {fraud_prob:.1f}%")
                st.write(f"• Legitimate Probability: {legitimate_prob:.1f}%")
                
                # # Progress bar for fraud probability
                # st.progress(fraud_prob / 100)
            
            # # Display input summary
            # with st.expander(" Input Summary"):
            #     st.write("**Personal Information:**")
            #     st.write(f"• Age: {age}, Sex: {sex}, Marital Status: {marital_status}")
            #     st.write(f"• Driver Rating: {driver_rating}")
                
            #     st.write("**Vehicle Information:**")
            #     st.write(f"• Make: {make}, Category: {vehicle_category}")
            #     st.write(f"• Age of Vehicle: {age_of_vehicle}, Price Range: {vehicle_price}")
                
            #     st.write("**Policy Information:**")
            #     st.write(f"• Policy Type: {policy_type}, Base Policy: {base_policy}")
            #     st.write(f"• Deductible: ${deductible}")
                
            #     st.write("**Claim Information:**")
            #     st.write(f"• Accident Area: {accident_area}, Fault: {fault}")
            #     st.write(f"• Police Report: {police_report_filed}, Witness Present: {witness_present}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check that all inputs are valid and the model is properly loaded.")

else:
    st.warning("⚠️ Model not loaded. Please ensure 'insurance_fraud_model.sav' is available in the application directory.")
    
# Footer
st.markdown("---")
st.markdown("*This tool is for demonstration purposes. Always consult with fraud investigation specialists for final decisions.*")