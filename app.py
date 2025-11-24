import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('water_quality_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        return model, scaler, imputer
    except FileNotFoundError:
        st.error("Model files not found. Please run train_model.py first.")
        return None, None, None

model, scaler, imputer = load_artifacts()

def main():
    st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="wide")
    
    st.title("💧 Water Quality Prediction App")
    st.markdown("""
    This application predicts whether water is **Safe** or **Unsafe** for consumption based on its chemical properties.
    """)
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Visualization"])
    
    if page == "Prediction":
        show_prediction_page()
    else:
        show_visualization_page()

def show_prediction_page():
    st.header("Check Water Potability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.number_input("pH Level (0-14)", min_value=0.0, max_value=14.0, value=7.0)
        hardness = st.number_input("Hardness (mg/L)", value=200.0)
        solids = st.number_input("Solids (ppm)", value=20000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.0)
        sulfate = st.number_input("Sulfate (mg/L)", value=300.0)
        
    with col2:
        conductivity = st.number_input("Conductivity (μS/cm)", value=400.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=10.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0)
        turbidity = st.number_input("Turbidity (NTU)", value=4.0)
    
    if st.button("Predict Quality"):
        if model is not None:
            # Create input array
            input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                    conductivity, organic_carbon, trihalomethanes, turbidity]])
            
            # Preprocess
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            st.divider()
            
            if prediction == 1:
                st.success(f"### Result: The water is SAFE to drink! 🥤")
                st.info(f"Probability of being safe: {probability:.2%}")
            else:
                st.error(f"### Result: The water is UNSAFE to drink! ⚠️")
                st.warning(f"Probability of being safe: {probability:.2%}")
                
                # Simple explanation
                reasons = []
                if ph < 6.5 or ph > 8.5: reasons.append("pH is out of safe range (6.5-8.5)")
                if turbidity > 5: reasons.append("Turbidity is too high (> 5 NTU)")
                if sulfate > 400: reasons.append("Sulfate levels are high (> 400 mg/L)")
                
                if reasons:
                    st.write("Potential issues:")
                    for r in reasons:
                        st.write(f"- {r}")

def show_visualization_page():
    st.header("Dataset Visualization")
    
    try:
        df = pd.read_csv('water_quality.csv')
        
        st.write("### Data Overview")
        st.dataframe(df.head())
        
        st.write("### Distribution of Potability")
        fig, ax = plt.subplots()
        sns.countplot(x='Potability', data=df, ax=ax, palette='viridis')
        ax.set_xticklabels(['Unsafe', 'Safe'])
        st.pyplot(fig)
        
        st.write("### Feature Distributions")
        feature = st.selectbox("Select Feature to Visualize", df.columns[:-1])
        
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x=feature, hue='Potability', kde=True, element="step", palette='viridis', ax=ax2)
        st.pyplot(fig2)
        
        st.write("### Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
        st.pyplot(fig3)
        
    except FileNotFoundError:
        st.error("Dataset not found. Please run generate_data.py first.")

if __name__ == "__main__":
    main()
