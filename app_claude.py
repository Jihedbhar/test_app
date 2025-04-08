import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import hashlib
import io
import base64
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Plateforme d'Optimisation Opérationnelle",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- INITIALIZATION -----

# Initialize session states
if "data" not in st.session_state:
    st.session_state.data = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "q_table" not in st.session_state:
    st.session_state.q_table = {}
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []
if "notifications" not in st.session_state:
    st.session_state.notifications = []

# ----- HELPER FUNCTIONS -----

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Simulated user database (in a real app, this would be in a secure database)
users_db = {
    "admin": {"password": hash_password("admin123"), "role": "top_level", "store": None, "full_name": "Administrator"},
    "manager_menzah": {"password": hash_password("pass1"), "role": "local", "store": "Baristas Menzah 1", "full_name": "Manager Menzah"},
    "manager_jardin": {"password": hash_password("pass2"), "role": "local", "store": "Baristas Jardin de Carthage", "full_name": "Manager Jardin"},
    "manager_marsa": {"password": hash_password("pass3"), "role": "local", "store": "Baristas la Marsa", "full_name": "Manager Marsa"}
}

# Check for session timeout (15 minutes)
def check_session_timeout():
    if st.session_state.authenticated:
        time_diff = datetime.now() - st.session_state.last_activity
        if time_diff > timedelta(minutes=15):
            st.session_state.authenticated = False
            st.session_state.user = None
            return True
    return False

# Update last activity timestamp
def update_activity():
    st.session_state.last_activity = datetime.now()

# Function to create a download link for dataframes
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Function to generate sample data
def generate_sample_data():
    stores = ["Baristas Menzah 1", "Baristas Jardin de Carthage", "Baristas la Marsa"]
    dates = pd.date_range(start="2023-01-01", end="2023-01-07", freq="H")
    
    data = []
    for store in stores:
        for date in dates:
            # Create realistic patterns:
            # - Weekday morning peak (7-9AM)
            # - Lunch peak (12-2PM)
            # - Afternoon lull (3-5PM)
            # - Evening peak for some locations (6-8PM)
            hour = date.hour
            day_of_week = date.dayofweek  # 0-6, Monday is 0
            
            # Base amount with store variation
            if store == "Baristas Menzah 1":
                base = 100
            elif store == "Baristas Jardin de Carthage":
                base = 120
            else:
                base = 90
                
            # Hour effect
            if 7 <= hour <= 9:  # Morning peak
                hourly_factor = 1.5
            elif 12 <= hour <= 14:  # Lunch peak
                hourly_factor = 2.0
            elif 15 <= hour <= 17:  # Afternoon lull
                hourly_factor = 0.7
            elif 18 <= hour <= 20:  # Evening activity
                if store in ["Baristas Jardin de Carthage", "Baristas la Marsa"]:
                    hourly_factor = 1.8
                else:
                    hourly_factor = 1.0
            elif 0 <= hour <= 6:  # Early morning (closed or very slow)
                hourly_factor = 0.1
            elif 21 <= hour <= 23:  # Late evening (closing time)
                hourly_factor = 0.3
            else:
                hourly_factor = 1.0
                
            # Weekend effect
            weekend_factor = 1.5 if day_of_week >= 5 else 1.0  # Higher on weekends
            
            # Random variation
            random_factor = np.random.normal(1, 0.2)
            
            # Calculate final amount
            amount = base * hourly_factor * weekend_factor * random_factor
            
            # Add some outliers
            if np.random.random() < 0.01:  # 1% chance of outlier
                amount = amount * np.random.choice([0.1, 3.0])  # Either very low or very high
                
            data.append({
                "timestamp": date,
                "amount": max(0, round(amount, 2)),  # Ensure no negative amounts
                "store_id": store,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": day_of_week >= 5
            })
    
    return pd.DataFrame(data)

# Weather impact calculator (simulated)
def get_weather_impact(date, store):
    # In a real application, this would fetch actual weather data from an API
    # For now, we'll simulate with random values based on date seed
    np.random.seed(int(pd.Timestamp(date).timestamp()) % 100000)
    
    # Generate weather conditions (sunny, cloudy, rainy)
    conditions = np.random.choice(["sunny", "cloudy", "rainy"], p=[0.6, 0.3, 0.1])
    
    # Impact coefficients (sunny increases sales for outdoor cafes, rainy decreases)
    impacts = {
        "Baristas Menzah 1": {"sunny": 1.1, "cloudy": 1.0, "rainy": 0.8},
        "Baristas Jardin de Carthage": {"sunny": 1.2, "cloudy": 0.9, "rainy": 0.7},
        "Baristas la Marsa": {"sunny": 1.15, "cloudy": 0.95, "rainy": 0.75}
    }
    
    return {
        "condition": conditions,
        "impact_factor": impacts.get(store, {"sunny": 1.0, "cloudy": 1.0, "rainy": 1.0})[conditions]
    }

# Apply anomaly detection
def detect_anomalies(df, contamination=0.05):
    if len(df) < 10:  # Need enough data for meaningful detection
        return pd.Series([False] * len(df))
    
    # Use only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return pd.Series([False] * len(df))
    
    model = IsolationForest(contamination=contamination, random_state=42)
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    try:
        predictions = model.fit_predict(df_numeric)
        return pd.Series(predictions == -1, index=df.index)
    except:
        return pd.Series([False] * len(df))

# Function to forecast sales
def forecast_sales(data, store, periods=24):
    store_data = data[data['store_id'] == store].copy()
    store_data = store_data.sort_values('timestamp')
    
    if len(store_data) < 10:
        return None, "Pas assez de données pour générer des prévisions"
    
    try:
        store_hourly = store_data.set_index('timestamp')['amount'].resample('H').sum()
        store_hourly = store_hourly.fillna(store_hourly.rolling(window=24, min_periods=1).mean())
        
        # Fit ARIMA model
        model = ARIMA(store_hourly, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Make forecast
        forecast = model_fit.forecast(steps=periods)
        
        # Create forecast DataFrame
        last_date = store_hourly.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=periods, freq='H')
        forecast_df = pd.DataFrame({
            'timestamp': forecast_index,
            'amount': forecast,
            'type': 'forecast'
        })
        
        # Create historical DataFrame
        historical_df = pd.DataFrame({
            'timestamp': store_hourly.index,
            'amount': store_hourly.values,
            'type': 'historical'
        })
        
        # Combine both DataFrames
        result_df = pd.concat([historical_df, forecast_df])
        return result_df, None
    except Exception as e:
        return None, f"Erreur de prévision: {str(e)}"

# Generate recommendations
def get_recommendation(store_data, store, full_data=None):
    if store_data.empty:
        return "Pas assez de données pour générer des recommandations", None, None, {}
    
    insights = {}
    
    # Basic statistics
    hourly_avg = store_data["amount"].mean()
    total_sales = store_data["amount"].sum()
    low_hour = store_data.loc[store_data["amount"].idxmin(), "hour"]
    high_hour = store_data.loc[store_data["amount"].idxmax(), "hour"]
    low_sales = store_data["amount"].min()
    high_sales = store_data["amount"].max()
    
    insights["stats"] = {
        "hourly_avg": hourly_avg,
        "total_sales": total_sales,
        "peak_hour": high_hour,
        "peak_sales": high_sales,
        "low_hour": low_hour,
        "low_sales": low_sales
    }
    
    # Detect anomalies if we have full data
    if full_data is not None and not full_data.empty:
        store_full_data = full_data[full_data["store_id"] == store].copy()
        if not store_full_data.empty:
            store_full_data["day"] = store_full_data["timestamp"].dt.date
            store_full_data["is_anomaly"] = detect_anomalies(store_full_data[["amount", "hour"]])
            anomalies = store_full_data[store_full_data["is_anomaly"]]
            insights["anomalies"] = len(anomalies)
            
            if not anomalies.empty:
                insights["anomaly_details"] = anomalies[["timestamp", "amount", "hour"]].to_dict(orient="records")
    
    # Generate different recommendations based on data patterns
    recommendations = []
    
    # Low hour recommendation
    key = (store, low_hour, "promo")
    q_value = st.session_state.q_table.get(key, 0.5)
    
    if low_sales < hourly_avg * 0.7:  # Very low sales compared to average
        if q_value > 0.3:  # If previous promotions worked
            recommendations.append({
                "type": "promo",
                "hour": low_hour,
                "message": f"Baisse significative d'activité à {low_hour}h ({low_sales:.2f} TND vs moyenne {hourly_avg:.2f} TND)",
                "action": "Lancer une promotion 'Happy Hour' avec 20% de réduction",
                "expected_impact": "Augmentation estimée de 30-40% du chiffre d'affaires",
                "confidence": q_value
            })
        else:
            recommendations.append({
                "type": "staff",
                "hour": low_hour,
                "message": f"Faible activité à {low_hour}h ({low_sales:.2f} TND)",
                "action": "Réduire le personnel pendant cette période pour optimiser les coûts",
                "expected_impact": "Réduction de 15% des coûts opérationnels",
                "confidence": 0.7
            })
    
    # Peak hour recommendation
    if high_sales > hourly_avg * 1.5:  # Very high sales compared to average
        recommendations.append({
            "type": "staff",
            "hour": high_hour,
            "message": f"Forte affluence à {high_hour}h ({high_sales:.2f} TND)",
            "action": "Augmenter le personnel pour cette période de pointe",
            "expected_impact": "Amélioration de l'expérience client et réduction des temps d'attente",
            "confidence": 0.8
        })
    
    # Weather-based recommendation (if available)
    if "timestamp" in store_data.columns and len(store_data) > 0:
        latest_date = store_data["timestamp"].max() if isinstance(store_data["timestamp"].iloc[0], pd.Timestamp) else datetime.now()
        weather = get_weather_impact(latest_date, store)
        if weather["condition"] == "sunny" and weather["impact_factor"] > 1.05:
            recommendations.append({
                "type": "weather",
                "hour": None,
                "message": f"Prévision météo favorable (Ensoleillé)",
                "action": "Préparer plus de boissons froides et ouvrir l'espace extérieur",
                "expected_impact": "Augmentation potentielle de 15% des ventes",
                "confidence": 0.6
            })
        elif weather["condition"] == "rainy" and weather["impact_factor"] < 0.9:
            recommendations.append({
                "type": "weather",
                "hour": None,
                "message": f"Prévision météo défavorable (Pluvieux)",
                "action": "Promouvoir les offres de livraison et les boissons chaudes",
                "expected_impact": "Limitation de l'impact négatif sur les ventes",
                "confidence": 0.6
            })
    
    # Choose the highest confidence recommendation
    if recommendations:
        best_rec = max(recommendations, key=lambda x: x["confidence"])
        return best_rec["message"], best_rec["hour"], best_rec["type"], recommendations
    else:
        return "Activité stable. Aucune action particulière requise.", None, "monitor", []

# ----- UI COMPONENTS -----

# Custom CSS for styling
def load_css():
    st.markdown("""
        <style>
        .main {background-color: #FAFAFA;}
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .notification {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            background-color: #f8d7da;
            border-left: 5px solid #842029;
        }
        .insight-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .nav-item {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            cursor: pointer;
        }
        .nav-item:hover {
            background-color: #f0f0f0;
        }
        .active-nav {
            background-color: #e0f7fa;
            border-left: 3px solid #00acc1;
        }
        .recommendation {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #4CAF50;
        }
        .recommendation-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .recommendation-action {
            background-color: #f1f8e9;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            border-left: 3px solid #7cb342;
        }
        .feedback-button {
            margin-right: 10px;
        }
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
            margin-bottom: 20px;
        }
        .help-text {
            font-size: 0.85rem;
            color: #666;
            font-style: italic;
        }
        .promo-tag {
            background-color: #ff9800;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 10px;
        }
        .staff-tag {
            background-color: #2196f3;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 10px;
        }
        .weather-tag {
            background-color: #9c27b0;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 10px;
        }
        .monitor-tag {
            background-color: #607d8b;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 10px;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

# Function to show notifications
def show_notifications():
    if st.session_state.notifications:
        for i, note in enumerate(st.session_state.notifications):
            st.markdown(f"""
                <div class="notification">
                    <strong>{note['title']}</strong>: {note['message']}
                </div>
            """, unsafe_allow_html=True)
        # Clear notifications after showing them
        if st.button("Effacer les notifications"):
            st.session_state.notifications = []
            st.experimental_rerun()

# Add notification
def add_notification(title, message):
    st.session_state.notifications.append({
        "title": title,
        "message": message,
        "timestamp": datetime.now()
    })

# Display header with user info
def show_header():
    if st.session_state.authenticated:
        user = st.session_state.user
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
                <div class="header-container">
                    <h1>Plateforme d'Optimisation Opérationnelle</h1>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style="text-align: right; padding-top: 10px;">
                    <p>Connecté en tant que: <strong>{user['full_name']}</strong><br>
                    Rôle: {user['role']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="header-container">
                <h1>Plateforme d'Optimisation Opérationnelle</h1>
            </div>
        """, unsafe_allow_html=True)

# ----- AUTHENTICATION -----

# Verify credentials
def check_credentials(username, password):
    if username in users_db and users_db[username]["password"] == hash_password(password):
        return True, users_db[username]["role"], users_db[username]["store"], users_db[username].get("full_name", username)
    return False, None, None, None

# Login page
def show_login_page():
    st.header("📊 Connexion")
    
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit_button = st.form_submit_button("Se connecter")
        
        if submit_button:
            valid, role, store, full_name = check_credentials(username, password)
            if valid:
                st.session_state.authenticated = True
                st.session_state.user = {
                    "username": username, 
                    "role": role, 
                    "store": store,
                    "full_name": full_name
                }
                st.session_state.login_attempts = 0
                update_activity()
                st.success(f"Connexion réussie ! Bienvenue, {full_name}.")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= 5:
                    st.error("Trop de tentatives de connexion échouées. Veuillez réessayer plus tard.")
                    time.sleep(3)
                else:
                    st.error("Identifiants incorrects. Veuillez réessayer.")
    
    # Demo account info
    with st.expander("Information de démonstration"):
        st.write("""
        Pour tester l'application, vous pouvez utiliser les comptes suivants:
        - Administrateur: `admin` / `admin123`
        - Manager Menzah: `manager_menzah` / `pass1`
        - Manager Jardin: `manager_jardin` / `pass2`
        - Manager Marsa: `manager_marsa` / `pass3`
        """)
    
    # Sample data generation
    st.subheader("🔍 Explorer avec des données d'exemple")
    st.write("Vous pouvez générer des données d'exemple pour tester la plateforme sans vous connecter.")
    
    if st.button("Générer des données d'exemple et explorer"):
        with st.spinner("Génération des données d'exemple..."):
            sample_data = generate_sample_data()
            st.session_state.data = sample_data
            st.session_state.authenticated = True
            st.session_state.user = {
                "username": "demo_user", 
                "role": "top_level", 
                "store": None,
                "full_name": "Utilisateur Démo"
            }
            update_activity()
            add_notification("Bienvenue", "Vous utilisez la plateforme en mode démo avec des données générées automatiquement.")
            st.experimental_rerun()

# ----- PAGE IMPLEMENTATION -----

# Dashboard page
def show_dashboard():
    st.header("📈 Tableau de bord")
    
    if st.session_state.data is None:
        st.warning("⚠️ Veuillez importer des données pour voir le tableau de bord.")
        if st.button("Générer des données d'exemple"):
            with st.spinner("Génération des données d'exemple..."):
                st.session_state.data = generate_sample_data()
                add_notification("Données générées", "Des données d'exemple ont été générées pour le tableau de bord.")
                st.experimental_rerun()
        return
    
    user_role = st.session_state.user["role"]
    user_store = st.session_state.user["store"]
    
    # Process data for visualization
    df = st.session_state.data.copy()
    
    # Add date components if timestamp is datetime
    if "timestamp" in df.columns and isinstance(df["timestamp"].iloc[0], pd.Timestamp):
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["day_name"] = df["timestamp"].dt.day_name()
        df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    
    # Get stores based on user role
    if user_role == "top_level":
        stores = df["store_id"].unique()
    else:
        stores = [user_store]
        df = df[df["store_id"] == user_store]
    
    # Show store selector for top-level users
    selected_store = None
    if user_role == "top_level":
        st.subheader("🏪 Sélection du magasin")
        view_mode = st.radio("Mode d'affichage", ["Vue globale", "Vue par magasin"], horizontal=True)
        
        if view_mode == "Vue par magasin":
            selected_store = st.selectbox("Sélectionner un magasin", stores)
            df_store = df[df["store_id"] == selected_store]
        else:
            df_store = df.copy()
    else:
        df_store = df.copy()
        selected_store = user_store
    
    # Display data download option
    st.sidebar.markdown("### 📥 Télécharger les données")
    st.sidebar.markdown(get_table_download_link(df, "data_export", "Télécharger les données (CSV)"), unsafe_allow_html=True)
    
    # Date range filter if we have timestamp data
    date_filter_active = False
    if "timestamp" in df.columns and isinstance(df["timestamp"].iloc[0], pd.Timestamp):
        st.sidebar.markdown("### 📅 Filtre par date")
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        
        date_filter = st.sidebar.date_input(
            "Sélectionner une plage de dates",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_filter) == 2:
            start_date, end_date = date_filter
            mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
            df_store = df_store[mask]
            date_filter_active = True
    
    # Display total sales and average metrics
    st.subheader("💰 Performance des ventes")
    
    if view_mode == "Vue globale" and user_role == "top_level":
        col1, col2, col3, col4 = st.columns(4)
        
        total_sales = df_store["amount"].sum()
        avg_sales = df_store["amount"].mean()
        peak_hour = df_store.groupby("hour")["amount"].sum().idxmax()
        num_transactions = len(df_store)
        
        col1.metric(
            "Chiffre d'affaires total", 
            f"{total_sales:.2f} TND",
            delta=None
        )
        col2.metric(
            "Vente moyenne", 
            f"{avg_sales:.2f} TND",
            delta=None
        )
        col3.metric(
            "Heure de pointe", 
            f"{peak_hour}h",
            delta=None
        )
        col4.metric(
            "Nombre de transactions", 
            f"{num_transactions}",
            delta=None
        )
        
        # Global charts
        st.subheader("📊 Répartition des ventes")
        
        tab1, tab2, tab3 = st.tabs(["Par magasin", "Par heure", "Tendances"])
        
        with tab1:
            # Sales by store
            sales_by_store = df.groupby("store_id")["amount"].sum().reset_index()
            fig = px.pie(
                sales_by_store, 
                values="amount", 
                names="store_id", 
                title="Répartition du chiffre d'affaires par magasin",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart comparison
            fig_bar = px.bar(
                sales_by_store,
                x="store_id",
                y="amount",
                title="Comparaison des ventes totales par magasin",
                color="store_id",
                text_auto='.2s'
            )
            fig_bar.update_layout(xaxis_title="Magasin", yaxis_title="Chiffre d'affaires (TND)")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            # Sales by hour
            hourly_sales = df.groupby("hour")["amount"].sum().reset_index()
            fig = px.line(
                hourly_sales, 
                x="hour", 
                y="amount", 
                title="Ventes totales par heure",
                markers=True,
                line_shape="spline"
            )
            fig.update_layout(xaxis_title="Heure", yaxis_title="Chiffre d'affaires (TND)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of sales by hour and store
            pivot = df.pivot_table(
                index="store_id", 
                columns="hour", 
                values="amount", 
                aggfunc="sum"
            )
            fig_heatmap = px.imshow(
                pivot,
                labels=dict(x="Heure", y="Magasin", color="Ventes (TND)"),
                x=pivot.columns,
                y=pivot.index,
                title="Heatmap des ventes par heure et par magasin",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        with tab3:
            if "date" in df.columns:
                # Time series of sales
                daily_sales = df.groupby(["date", "store_id"])["amount"].sum().reset_index()
                fig = px.line(
                    daily_sales, 
                    x="date", 
                    y="amount", 
                    color="store_id",
                    title="Évolution des ventes quotidiennes par magasin",
                    markers=True
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Chiffre d'affaires (TND)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekday analysis
            if "day_name" in df.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday_sales = df.groupby(["day_name", "store_id"])["amount"].sum().reset_index()
                weekday_sales["day_name"] = pd.Categorical(weekday_sales["day_name"], categories=day_order, ordered=True)
                weekday_sales = weekday_sales.sort_values("day_name")
                    
                fig = px.bar(
                    weekday_sales, 
                    x="day_name", 
                    y="amount", 
                    color="store_id", 
                    barmode="group",
                    title="Ventes par jour de la semaine"
                )
                fig.update_layout(xaxis_title="Jour", yaxis_title="Chiffre d'affaires (TND)")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Single store view
        store_data = df_store.copy()
                    
        # Calculate metrics
        total_sales = store_data["amount"].sum()
        avg_hourly_sales = store_data.groupby("hour")["amount"].mean().mean()
        if "date" in store_data.columns:
            num_days = store_data["date"].nunique() 
            daily_avg = total_sales / num_days if num_days > 0 else 0
        else:
            daily_avg = total_sales / 7  # Assume a week of data if no date column
        
        peak_hour = store_data.groupby("hour")["amount"].sum().idxmax()
        peak_sales = store_data.groupby("hour")["amount"].sum().max()


        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{total_sales:.2f} TND</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Chiffre d'affaires total</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{daily_avg:.2f} TND</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Moyenne quotidienne</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{peak_hour}h</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Heure de pointe ({peak_sales:.2f} TND)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{avg_hourly_sales:.2f} TND</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Vente moyenne par heure</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display charts
        st.subheader(f"📊 Analyse détaillée pour {selected_store or user_store}")
        
        tab1, tab2 = st.tabs(["Analyse horaire", "Analyse temporelle"])
        
        with tab1:
            # Hourly sales
            hourly_data = store_data.groupby("hour")["amount"].sum().reset_index()
            fig = px.bar(
                hourly_data,
                x="hour",
                y="amount",
                title="Ventes par heure",
                color="amount",
                color_continuous_scale="Viridis",
                text_auto='.2s'
            )
            fig.update_layout(xaxis_title="Heure", yaxis_title="Chiffre d'affaires (TND)")
            st.plotly_chart(fig, use_container_width=True)
            # Detect outliers in hourly pattern
            hourly_data["is_outlier"] = detect_anomalies(hourly_data)
            if hourly_data["is_outlier"].any():
                st.subheader("⚠️ Heures atypiques détectées")
                outlier_hours = hourly_data[hourly_data["is_outlier"]]
                for _, row in outlier_hours.iterrows():
                    if row["amount"] > hourly_data["amount"].mean():
                        st.info(f"📈 Pic de ventes inhabituel à {row['hour']}h ({row['amount']:.2f} TND)")
                    else:
                        st.warning(f"📉 Baisse de ventes inhabituelle à {row['hour']}h ({row['amount']:.2f} TND)")
        
        with tab2:
            if "date" in store_data.columns and "timestamp" in store_data.columns:
                # Time series analysis
                daily_data = store_data.groupby("date")["amount"].sum().reset_index()
                fig = px.line(
                    daily_data,
                    x="date",
                    y="amount",
                    title="Évolution des ventes quotidiennes",
                    markers=True,
                    line_shape="spline"
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Chiffre d'affaires (TND)")
                st.plotly_chart(fig, use_container_width=True)
                # Forecast future sales
                st.subheader("🔮 Prévision des ventes")
                forecast_periods = st.slider("Nombre d'heures à prévoir", 6, 72, 24)
                if st.button("Générer une prévision"):
                    with st.spinner("Génération de la prévision en cours..."):
                        forecast_data, error_msg = forecast_sales(df, selected_store or user_store, periods=forecast_periods)
                        if forecast_data is not None:
                            fig = px.line(
                                forecast_data,
                                x="timestamp",
                                y="amount",
                                color="type",
                                title=f"Prévision des ventes pour {selected_store or user_store}",
                                color_discrete_map={"historical": "blue", "forecast": "red"}
                            )
                            fig.update_layout(
                                xaxis_title="Date et heure",
                                yaxis_title="Chiffre d'affaires (TND)",
                                legend_title="Type de données"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display forecast data in a table
                            forecast_only = forecast_data[forecast_data["type"] == "forecast"]
                            st.write("Valeurs prévues:")
                            st.dataframe(forecast_only[["timestamp", "amount"]].set_index("timestamp"))
                        else:
                            st.error(f"Impossible de générer la prévision: {error_msg}")
    # Footer for dashboard
    st.markdown("---")
    st.markdown("*Données mises à jour le: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "*")
    if date_filter_active:
        st.markdown("*Filtrage actif: Les données affichées sont filtrées par date.*")            

def show_data_import():
    st.header("📥 Importer des données")
    
    st.write("""
    Sur cette page, vous pouvez importer vos données de ventes pour analyse. 
    L'application accepte les fichiers CSV avec différentes options de séparateur et d'encodage.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
    
    # Options for data import
    col1, col2 = st.columns(2)
    with col1:
        delimiter = st.selectbox("Séparateur du CSV", [",", ";", "\t"], index=0)
    with col2:
        encoding = st.selectbox("Encodage du fichier", ["utf-8", "latin-1", "cp1252"], index=0)
    
    # Sample data generation option
    st.subheader("🔄 Ou générer des données d'exemple")
    if st.button("Générer des données d'exemple"):
        with st.spinner("Génération des données d'exemple..."):
            st.session_state.data = generate_sample_data()
            st.success("Données d'exemple générées avec succès !")
            st.dataframe(st.session_state.data.head())
            add_notification("Données générées", "Des données d'exemple ont été générées avec succès.")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding)
            st.success("Fichier importé avec succès !")
            
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            st.subheader("Mapper les colonnes")
            cols = df.columns.tolist()
            
            # Column mapping
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Colonne de date/heure", cols, key="date_col")
                amount_col = st.selectbox("Colonne de montant", cols, key="amount_col")
            with col2:
                store_col = st.selectbox("Colonne de magasin", cols, key="store_col")
                # Optional mappings
                extra_cols = st.multiselect("Colonnes additionnelles (optionnel)", cols)
            
            # Data format helper
            with st.expander("Aide sur les formats de données"):
                st.write("""
                - **Date/Heure**: Le format attendu est JJ/MM/AAAA HH:MM:SS ou JJ/MM/AAAA HH:MM. 
                  Alternativement, vous pouvez utiliser des valeurs numériques pour les heures (0-23).
                - **Montant**: Des valeurs numériques comme 12.34 ou 12,34 sont acceptées.
                - **Magasin**: Le nom ou identifiant du magasin (ex: "Baristas Menzah 1").
                """)
            
            # Confirm mapping
            if st.button("Confirmer le mapping et importer les données"):
                with st.spinner("Traitement des données..."):
                    # Create a copy of the dataframe
                    processed_df = df.copy()
                    
                    # Rename columns
                    processed_df = processed_df.rename(columns={
                        date_col: "timestamp", 
                        amount_col: "amount", 
                        store_col: "store_id"
                    })
                    
                    # Keep selected extra columns
                    cols_to_keep = ["timestamp", "amount", "store_id"] + extra_cols
                    processed_df = processed_df[cols_to_keep]
                    
                    # Convert amount to numeric
                    try:
                        processed_df["amount"] = pd.to_numeric(processed_df["amount"], errors="raise")
                    except ValueError:
                        # Try to replace commas with periods
                        try:
                            processed_df["amount"] = processed_df["amount"].str.replace(",", ".").astype(float)
                        except (ValueError, AttributeError):
                            st.error("Erreur : La colonne 'montant' contient des valeurs non numériques.")
                            st.stop()
                    
                    # Process timestamp
                    if processed_df["timestamp"].dtype in ["int64", "float64"]:
                        processed_df["hour"] = processed_df["timestamp"].astype(int)
                    else:
                        try:
                            # Clean whitespace
                            processed_df["timestamp"] = processed_df["timestamp"].str.replace(r'\s+', ' ', regex=True).str.strip()
                            
                            # Try different date formats
                            try:
                                processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"], format="%d/%m/%Y %H:%M:%S")
                            except ValueError:
                                try:
                                    processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"], format="%d/%m/%Y %H:%M")
                                except ValueError:
                                    processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"])
                            
                            # Extract hour
                            processed_df["hour"] = processed_df["timestamp"].dt.hour
                            
                        except Exception as e:
                            st.error(f"Erreur lors du traitement de la colonne date/heure: {str(e)}")
                            st.stop()
                    
                    # Aggregate data by store and hour
                    aggregated_df = processed_df.groupby(["store_id", "hour"])["amount"].sum().reset_index()
                    
                    # Store both the raw and aggregated data
                    st.session_state.data = processed_df
                    
                    st.success("Données importées et traitées avec succès !")
                    
                    # Show data quality report
                    st.subheader("Rapport de qualité des données")
                    
                    # Check for missing values
                    missing_values = processed_df.isnull().sum()
                    if missing_values.sum() > 0:
                        st.warning("⚠️ Des valeurs manquantes ont été détectées:")
                        st.write(missing_values[missing_values > 0])
                    else:
                        st.success("✅ Aucune valeur manquante détectée.")
                    
                    # Check for duplicate entries
                    duplicates = processed_df.duplicated().sum()
                    if duplicates > 0:
                        st.warning(f"⚠️ {duplicates} entrées dupliquées détectées.")
                    else:
                        st.success("✅ Aucune entrée dupliquée détectée.")
                    
                    # Check data range
                    if "timestamp" in processed_df.columns and isinstance(processed_df["timestamp"].iloc[0], pd.Timestamp):
                        min_date = processed_df["timestamp"].min()
                        max_date = processed_df["timestamp"].max()
                        st.info(f"📅 Plage de dates: du {min_date.strftime('%d/%m/%Y')} au {max_date.strftime('%d/%m/%Y')}")
                    
                    # Display unique stores
                    unique_stores = processed_df["store_id"].unique()
                    st.info(f"🏪 Magasins identifiés: {', '.join(unique_stores)}")
                    
                    # Preview processed data
                    st.subheader("Aperçu des données traitées")
                    st.dataframe(processed_df.head())
                    
                    # Add notification
                    add_notification("Données importées", f"Importation réussie de {len(processed_df)} enregistrements.")
        
        except Exception as e:
            st.error(f"Erreur lors de l'importation du fichier: {str(e)}")
            
            # Display raw content for debugging
            st.subheader("Contenu brut du fichier (pour diagnostic)")
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode(encoding)
                st.text_area("Contenu", content, height=200)
            except:
                st.error("Impossible d'afficher le contenu du fichier.")

# Recommendations page
def show_recommendations():
    st.header("🧠 Recommandations intelligentes")
    
    if st.session_state.data is None:
        st.warning("⚠️ Veuillez importer des données pour voir les recommandations.")
        if st.button("Générer des données d'exemple"):
            with st.spinner("Génération des données d'exemple..."):
                st.session_state.data = generate_sample_data()
                add_notification("Données générées", "Des données d'exemple ont été générées pour visualiser les recommandations.")
                st.experimental_rerun()
        return
    
    user_role = st.session_state.user["role"]
    user_store = st.session_state.user["store"]
    
    # Process data
    df = st.session_state.data.copy()
    
    # Generate recommendations based on user role
    if user_role == "top_level":
        st.subheader("👁️ Vue d'ensemble des recommandations")
        
        # Get unique stores
        stores = df["store_id"].unique()
        
        # Generate recommendations for each store
        all_recommendations = []
        
        for store in stores:
            store_data = df[df["store_id"] == store]
            message, hour, rec_type, details = get_recommendation(store_data, store, df)
            
            if details:
                for detail in details:
                    detail["store"] = store
                    all_recommendations.append(detail)
        
        # Also generate global recommendation
        global_data = df.groupby("hour")["amount"].sum().reset_index()
        global_message, global_hour, global_type, global_details = get_recommendation(global_data, "global", df)
        
        if global_details:
            for detail in global_details:
                detail["store"] = "Tous les magasins"
                all_recommendations.append(detail)
        
        # Display recommendations
        if all_recommendations:
            # Sort by confidence
            all_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Show as cards
            for i, rec in enumerate(all_recommendations):
                # Create a unique card style based on recommendation type
                tag_class = f"{rec['type']}-tag"
                
                st.markdown(f"""
                <div class="recommendation">
                    <div class="recommendation-title">
                        {rec['message']} <span class="{tag_class}">{rec['type']}</span>
                    </div>
                    <p><strong>Magasin:</strong> {rec['store']}</p>
                    <div class="recommendation-action">
                        <strong>Action recommandée:</strong> {rec['action']}<br>
                        <strong>Impact estimé:</strong> {rec['expected_impact']}<br>
                        <strong>Confiance:</strong> {rec['confidence']*100:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune recommandation n'a pu être générée avec les données actuelles.")
        
        # Show recommendation details
        with st.expander("Comprendre les recommandations"):
            st.write("""
            Les recommandations sont générées à partir de l'analyse des données de ventes et prennent en compte:
            - Les variations horaires de l'activité
            - Les périodes creuses et de pointe
            - Les conditions météorologiques (simulées)
            - L'historique des feedbacks sur les recommandations précédentes
            
            Types de recommandations:
            - **Promo**: Suggestions de promotions ou offres spéciales
            - **Staff**: Recommandations d'optimisation du personnel
            - **Weather**: Actions basées sur les prévisions météo
            - **Monitor**: Surveillance continue sans action spécifique
            """)
    
    else:
        # Local manager view
        st.subheader(f"🎯 Recommandations pour {user_store}")
        
        # Get store data
        store_data = df[df["store_id"] == user_store]
        
        if store_data.empty:
            st.warning("Aucune donnée disponible pour votre magasin.")
            return
        
        # Generate recommendation
        message, hour, rec_type, details = get_recommendation(store_data, user_store, df)
        
        # Display recommendation
        if details:
            for rec in details:
                # Create a unique card style based on recommendation type
                tag_class = f"{rec['type']}-tag"
                
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation">
                        <div class="recommendation-title">
                            {rec['message']} <span class="{tag_class}">{rec['type']}</span>
                        </div>
                        <div class="recommendation-action">
                            <strong>Action recommandée:</strong> {rec['action']}<br>
                            <strong>Impact estimé:</strong> {rec['expected_impact']}<br>
                            <strong>Confiance:</strong> {rec['confidence']*100:.0f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add feedback section
                    st.subheader("📝 Votre avis sur cette recommandation")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        feedback = st.radio("Que pensez-vous de cette recommandation ?", ("Accepter", "Rejeter"), key=f"feedback_{rec['type']}")
                    with col2:
                        comment = st.text_area("Commentaire (optionnel)", key=f"comment_{rec['type']}", height=100)
                    
                    if st.button("Envoyer le feedback", key=f"send_{rec['type']}"):
                        reward = 1 if feedback == "Accepter" else -1
                        key = (user_store, rec.get('hour', 0), rec['type'])
                        old_q = st.session_state.q_table.get(key, 0.5)
                        st.session_state.q_table[key] = old_q + 0.1 * (reward - old_q)
                        
                        # Store feedback in history
                        feedback_entry = {
                            "timestamp": datetime.now(),
                            "store": user_store,
                            "recommendation_type": rec['type'],
                            "recommendation": rec['message'],
                            "feedback": feedback,
                            "comment": comment,
                            "new_q_value": st.session_state.q_table[key]
                        }
                        st.session_state.feedback_history.append(feedback_entry)
                        
                        st.success(f"Feedback enregistré : {feedback} - Nouvelle valeur de confiance : {st.session_state.q_table[key]:.2f}")
                        
                        # Add notification
                        add_notification("Feedback enregistré", f"Votre avis sur la recommandation '{rec['type']}' a été pris en compte.")
        else:
            st.info("Aucune recommandation n'a pu être générée avec les données actuelles.")
    
    # Show feedback history
    if st.session_state.feedback_history:
        st.subheader("📜 Historique des feedbacks")
        feedback_df = pd.DataFrame(st.session_state.feedback_history)
        
        # Limit to current store for local managers
        if user_role == "local":
            feedback_df = feedback_df[feedback_df["store"] == user_store]
        
        if not feedback_df.empty:
            feedback_df = feedback_df.sort_values("timestamp", ascending=False)
            st.dataframe(feedback_df[["timestamp", "store", "recommendation_type", "feedback", "comment", "new_q_value"]])
        else:
            st.info("Aucun historique de feedback disponible.")
    
    # Advanced settings
    with st.expander("⚙️ Paramètres avancés du système de recommandation"):
        st.write("""
        Le système de recommandation utilise un algorithme d'apprentissage par renforcement 
        simplifié (Q-learning) pour améliorer la qualité des recommandations en fonction de vos feedbacks.
        """)
        
        # Show current Q-table
        if st.session_state.q_table:
            st.subheader("Table de valeurs Q")
            q_data = []
            for (store, hour, action), value in st.session_state.q_table.items():
                q_data.append({
                    "Magasin": store,
                    "Heure": hour,
                    "Type d'action": action,
                    "Valeur Q": value
                })
            q_df = pd.DataFrame(q_data)
            st.dataframe(q_df)
            
            # Allow reset of Q-table
            if st.button("Réinitialiser le système d'apprentissage"):
                st.session_state.q_table = {}
                st.session_state.feedback_history = []
                st.success("Système d'apprentissage réinitialisé avec succès.")
                add_notification("Système réinitialisé", "Le système de recommandation a été réinitialisé.")
                st.experimental_rerun()
        else:
            st.info("Le système d'apprentissage n'a pas encore de données.")

# Settings page
def show_settings():
    st.header("⚙️ Paramètres")
    
    # User profile section
    st.subheader("👤 Profil utilisateur")
    user = st.session_state.user
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Nom d'utilisateur:** {user['username']}")
        st.info(f"**Nom complet:** {user['full_name']}")
    with col2:
        st.info(f"**Rôle:** {user['role']}")
        if user['store']:
            st.info(f"**Magasin:** {user['store']}")
    
    # App settings
    st.subheader("🖥️ Paramètres de l'application")
    
    # Theme selection
    theme = st.selectbox(
        "Thème de l'interface",
        ["Clair", "Sombre", "Système"],
        index=0
    )
    
    # Language selection
    language = st.selectbox(
        "Langue",
        ["Français", "Anglais", "Arabe"],
        index=0
    )
    
    # Notification settings
    st.subheader("🔔 Notifications")
    
    enable_notifications = st.checkbox("Activer les notifications dans l'application", value=True)
    enable_email = st.checkbox("Recevoir des notifications par email", value=False)
    
    if enable_email:
        email = st.text_input("Adresse email pour les notifications")
    
    # Data settings
    st.subheader("📊 Paramètres des données")
    
    # Data retention
    data_retention = st.slider(
        "Période de conservation des données (jours)",
        min_value=30,
        max_value=365,
        value=90,
        step=30
    )
    
    # Export data option
    st.subheader("📤 Exporter les données")
    
    if st.session_state.data is not None:
        export_format = st.selectbox(
            "Format d'exportation",
            ["CSV", "Excel", "JSON"],
            index=0
        )
        
        if st.button("Exporter les données"):
            if export_format == "CSV":
                tmp_download_link = get_table_download_link(st.session_state.data, "export_data", "Télécharger le fichier CSV")
                st.markdown(tmp_download_link, unsafe_allow_html=True)
            else:
                st.info(f"Export au format {export_format} sera disponible dans une future mise à jour.")
    else:
        st.warning("Aucune donnée disponible à exporter.")
    
    # Save settings
    if st.button("Enregistrer les paramètres"):
        st.success("Paramètres enregistrés avec succès !")
        
        # Apply theme change
        if theme == "Sombre":
            st.markdown("""
                <style>
                .main {background-color: #121212; color: #e0e0e0;}
                .st-bd {border-color: #333333;}
                .st-bb {background-color: #121212;}
                .st-at {background-color: #1e1e1e;}
                .st-cv {color: #e0e0e0;}
                </style>
            """, unsafe_allow_html=True)
        
        # Add notification
        if enable_notifications:
            add_notification("Paramètres mis à jour", "Vos préférences ont été enregistrées avec succès.")
            
# Pied de page
st.sidebar.markdown("---")
st.sidebar.write("Projet PFE - Plateforme générique B2B")
st.sidebar.write("Développée par un stagiaire")

