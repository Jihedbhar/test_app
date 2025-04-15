import streamlit as st
import pandas as pd
import plotly.express as px
import hashlib
import os
from pathlib import Path
import datetime

# --- Constants ---
DATA_DIR = Path("app_data")
PROCESSED_DATA_FILE = DATA_DIR / "processed_sales_data.parquet"
FEEDBACK_LOG_FILE = DATA_DIR / "feedback_log.csv"
ROLE_TOP_LEVEL = "top_level"
ROLE_LOCAL = "local"

# --- Utility Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    return stored_hash == hash_password(provided_password)

def ensure_data_dir_exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_processed_data(df):
    ensure_data_dir_exists()
    try:
        df.to_parquet(PROCESSED_DATA_FILE, index=False)
        st.success(f"Data saved successfully to {PROCESSED_DATA_FILE}")
    except Exception as e:
        st.error(f"Error saving data: {e}")

def log_feedback(user, store, recommendation_details, feedback, comment, q_value_change):
    ensure_data_dir_exists()
    log_entry = {
        "timestamp": datetime.datetime.now(),
        "user": user,
        "store_context": store,
        "recommendation_details": recommendation_details,
        "feedback": feedback,
        "comment": comment,
        "q_value_change": q_value_change
    }
    log_df = pd.DataFrame([log_entry])
    try:
        if FEEDBACK_LOG_FILE.exists():
            log_df.to_csv(FEEDBACK_LOG_FILE, mode='a', header=False, index=False, encoding='utf-8')
        else:
            log_df.to_csv(FEEDBACK_LOG_FILE, mode='w', header=True, index=False, encoding='utf-8')
    except Exception as e:
        st.warning(f"Could not log feedback to file: {e}")

# --- Initialization ---
ensure_data_dir_exists()

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "Accueil"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "q_table" not in st.session_state:
    st.session_state.q_table = {}

# --- User Database ---
users_db = {
    "admin": {
        "password_hash": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # admin123
        "role": ROLE_TOP_LEVEL,
        "store": None
    },
    "manager_menzah": {
        "password_hash": "e6c3da5b206634d7f3f3586d747ffdb36b5c675757b380c6a5fe5c570c714349",  # pass1
        "role": ROLE_LOCAL,
        "store": "Baristas Menzah 1"
    },
    "manager_jardin": {
        "password_hash": "1ba3d16e9881959f8c9a9762854f72c6e6321cdd44358a10a4e939033117eab9",  # pass2
        "role": ROLE_LOCAL,
        "store": "Baristas Jardin de Carthage"
    },
    "manager_marsa": {
        "password_hash": "3acb59306ef6e660cf832d1d34c4fba3d88d616f0bb5c2a9e0f82d18ef6fc167",  # pass3
        "role": ROLE_LOCAL,
        "store": "Baristas la Marsa"
    }
}

# --- Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover { background-color: #1b5e20; }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #bdbdbd;
        border-radius: 8px;
        padding: 10px;
    }
    h1 { color: #1e3a8a; font-family: 'Arial', sans-serif; }
    h2 { color: #2e7d32; }
    h3 { color: #374151; }
    .sidebar .sidebar-content { background-color: #e8f0fe; }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 8px; }
    .welcome-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Authentication Logic ---
def check_credentials(username, password):
    user_data = users_db.get(username)
    if user_data and verify_password(user_data["password_hash"], password):
        return True, user_data["role"], user_data["store"]
    return False, None, None

# --- Page Rendering ---
def render_homepage():
    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
    st.title("Bienvenue sur la Plateforme d'Optimisation Opérationnelle")
    st.markdown("""
        <h3 style="color: #374151;">Boostez vos performances commerciales</h3>
        <p style="font-size: 18px; color: #4b5563;">
            Analysez vos données de vente, obtenez des recommandations personnalisées et prenez des décisions éclairées pour maximiser vos revenus.
        </p>
    """, unsafe_allow_html=True)
    if st.button("Se connecter", key="home_login"):
        st.session_state.page = "Connexion"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6b7280;">
            <p><strong>Pourquoi choisir notre plateforme ?</strong></p>
            <p>Visualisations intuitives • Recommandations basées sur les données • Interface sécurisée</p>
        </div>
    """, unsafe_allow_html=True)

def render_login():
    st.header("Connexion Sécurisée")
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
        if submitted:
            if not username or not password:
                st.warning("Veuillez entrer un nom d'utilisateur et un mot de passe.")
            else:
                valid, role, store = check_credentials(username, password)
                if valid:
                    st.session_state.authenticated = True
                    st.session_state.user = {"username": username, "role": role, "store": store}
                    st.session_state.page = "Importer des données"
                    st.success(f"Connexion réussie ! Bienvenue, {username}.")
                    st.rerun()
                else:
                    st.error("Identifiants incorrects. Veuillez réessayer.")

def render_data_import():
    st.header("📥 Importer vos Données")
    st.info("Téléchargez un fichier CSV avec au moins une colonne pour la date/heure, une pour les montants et une pour l'identifiant (ex: magasin).")

    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])

    if uploaded_file:
        st.markdown("---")
        st.subheader("Paramètres d'importation")
        col1, col2 = st.columns(2)
        with col1:
            delimiter = st.selectbox("Séparateur", [",", ";", "\t", "|"], index=0)
        with col2:
            encoding = st.selectbox("Encodage", ["utf-8", "latin-1", "iso-8859-1", "cp1252"], index=0)

        date_formats = [
            "Automatique",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M"
        ]
        selected_date_format = st.selectbox("Format de date", date_formats, index=0)

        try:
            with st.spinner('Lecture du fichier CSV...'):
                df_raw = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding, low_memory=False)

            st.markdown("---")
            st.subheader("Aperçu des données brutes")
            st.dataframe(df_raw.head())
            st.write(f"Nombre total de lignes : {len(df_raw)}")

            st.markdown("---")
            st.subheader("Mappage des colonnes")
            cols = ["(Ignorer)"] + df_raw.columns.tolist()
            required_cols = {
                "timestamp": st.selectbox("Colonne Date/Heure", cols, index=0, key="date_col"),
                "amount": st.selectbox("Colonne Montant", cols, index=0, key="amount_col"),
                "store_id": st.selectbox("Colonne Identifiant (ex: Magasin)", cols, index=0, key="store_col")
            }

            if st.button("🔄 Traiter les Données"):
                with st.spinner('Traitement des données...'):
                    selected_cols = [v for k, v in required_cols.items() if v != "(Ignorer)"]
                    if not selected_cols:
                        st.error("Veuillez sélectionner au moins une colonne.")
                        return

                    df_processed = df_raw[selected_cols].copy()
                    df_processed.columns = [k for k, v in required_cols.items() if v != "(Ignorer)"]

                    errors = []
                    # Amount Conversion
                    if "amount" in df_processed.columns:
                        try:
                            df_processed["amount"] = df_processed["amount"].astype(str).str.replace(",", ".", regex=False)
                            df_processed["amount"] = pd.to_numeric(df_processed["amount"], errors='coerce')
                            null_amounts = df_processed["amount"].isnull().sum()
                            if null_amounts > 0:
                                errors.append(f"Suppression de {null_amounts} lignes avec montants non valides.")
                                df_processed.dropna(subset=["amount"], inplace=True)
                            st.write(f"Total brut des montants après conversion : {df_processed['amount'].sum():,.2f} TND")
                        except Exception as e:
                            errors.append(f"Erreur lors de la conversion des montants : {e}")

                    # Timestamp Conversion
                    if "timestamp" in df_processed.columns:
                        try:
                            if selected_date_format == "Automatique":
                                df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"], errors='coerce')
                            else:
                                df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"], format=selected_date_format, errors='coerce')
                            null_timestamps = df_processed["timestamp"].isnull().sum()
                            if null_timestamps > 0:
                                errors.append(f"Suppression de {null_timestamps} lignes avec dates non valides.")
                                df_processed.dropna(subset=["timestamp"], inplace=True)
                        except Exception as e:
                            errors.append(f"Erreur lors de la conversion des dates : {e}")

                    # Store ID
                    if "store_id" in df_processed.columns:
                        df_processed["store_id"] = df_processed["store_id"].astype(str).str.strip()

                    # Feature Engineering
                    if "timestamp" in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed["timestamp"]):
                        df_processed["date"] = df_processed["timestamp"].dt.date
                        df_processed["hour"] = df_processed["timestamp"].dt.hour
                        df_processed["weekday"] = df_processed["timestamp"].dt.day_name()

                    # Debug: Show processed data stats
                    st.markdown("---")
                    st.subheader("Statistiques après traitement")
                    if "amount" in df_processed.columns:
                        st.write(f"Total des montants : {df_processed['amount'].sum():,.2f} TND")
                        st.write(f"Nombre de transactions : {len(df_processed)}")
                    if "timestamp" in df_processed.columns:
                        min_date = df_processed["timestamp"].min()
                        max_date = df_processed["timestamp"].max()
                        st.write(f"Plage de dates : {min_date.strftime('%d/%m/%Y')} à {max_date.strftime('%d/%m/%Y')}")
                    if "store_id" in df_processed.columns:
                        st.write(f"Magasins uniques : {df_processed['store_id'].nunique()}")

                    if errors:
                        st.error("Erreurs rencontrées :")
                        for error in errors:
                            st.warning(f"- {error}")

                    if not df_processed.empty:
                        st.success("Données traitées avec succès !")
                        st.dataframe(df_processed.head())
                        st.session_state.processed_data = df_processed
                        save_processed_data(df_processed)
                        st.session_state.page = "Choix"
                        st.rerun()
                    else:
                        st.error("Aucune donnée valide après traitement. Vérifiez votre fichier.")

        except pd.errors.EmptyDataError:
            st.error("Fichier CSV vide.")
        except pd.errors.ParserError as e:
            st.error(f"Erreur de lecture CSV : {e}. Vérifiez le délimiteur.")
        except UnicodeDecodeError:
            st.error("Erreur d'encodage. Essayez un autre encodage.")
        except Exception as e:
            st.error(f"Erreur inattendue : {str(e)}")

def render_choice_page():
    st.header("🚀 Que souhaitez-vous faire ?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Voir le Tableau de Bord"):
            st.session_state.page = "Tableau de bord"
            st.rerun()
    with col2:
        if st.button("💡 Voir les Recommandations"):
            st.session_state.page = "Recommandations"
            st.rerun()

def render_dashboard():
    st.header("📈 Tableau de Bord Analytique")
    if st.session_state.processed_data is None:
        st.warning("Aucune donnée disponible. Veuillez importer des données.")
        st.session_state.page = "Importer des données"
        st.rerun()
        return

    df = st.session_state.processed_data.copy()
    user_role = st.session_state.user["role"]
    user_store = st.session_state.user["store"]

    # Apply store filter immediately for local managers
    if user_role == ROLE_LOCAL and "store_id" in df.columns:
        df = df[df["store_id"] == user_store]
        if df.empty:
            st.warning(f"Aucune donnée pour {user_store}.")
            return

    # Debug: Raw data stats (now store-specific for local managers)
    st.markdown("**Debug : Statistiques des données brutes**")
    st.write(f"Nombre de lignes (items) : {len(df)}")
    if "amount" in df.columns:
        st.write(f"Total des montants bruts : {df['amount'].sum():,.2f} TND")
    if user_role == ROLE_TOP_LEVEL and "store_id" in df.columns:
        st.write(f"Magasins uniques : {df['store_id'].nunique()}")

    # Filters
    st.sidebar.header("Filtres")
    if "timestamp" in df.columns:
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.sidebar.date_input(
            "Période",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter"
        )
        if len(date_range) != 2:
            st.warning("Veuillez sélectionner une plage de dates complète.")
            return
        start_date = datetime.datetime.combine(date_range[0], datetime.datetime.min.time())
        end_date = datetime.datetime.combine(date_range[1], datetime.datetime.max.time())
        df_filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    else:
        df_filtered = df
        start_date, end_date = None, None

    # Store Filter (only for top-level users)
    if user_role == ROLE_TOP_LEVEL and "store_id" in df_filtered.columns:
        available_stores = sorted(df_filtered["store_id"].unique())
        selected_stores = st.sidebar.multiselect("Magasins", available_stores, default=available_stores)
        if not selected_stores:
            st.sidebar.warning("Sélectionnez au moins un magasin.")
            return
        df_filtered = df_filtered[df_filtered["store_id"].isin(selected_stores)]
    # For local managers, df_filtered is already filtered to their store

    # Aggregate by exact timestamp and store_id to define transactions
    if "amount" in df_filtered.columns and "store_id" in df_filtered.columns and "timestamp" in df_filtered.columns:
        # Group items into transactions (same timestamp, same store)
        df_transactions = df_filtered.groupby(["store_id", "timestamp"])["amount"].sum().reset_index()
        df_transactions["hour"] = df_transactions["timestamp"].dt.hour
        st.markdown("**Debug : Statistiques des transactions**")
        st.write(f"Nombre de transactions (après groupement) : {len(df_transactions)}")
        st.write(f"Total des montants des transactions : {df_transactions['amount'].sum():,.2f} TND")
        st.write(f"Exemple de transactions :")
        st.dataframe(df_transactions.head())

        # Aggregate transactions by store and hour for visualization
        df_agg_hourly = df_transactions.groupby(["store_id", "hour"])["amount"].sum().reset_index()
        df_agg_total = df_transactions.groupby("store_id")["amount"].sum().reset_index()
        st.markdown("**Debug : Agrégation horaire des transactions**")
        st.dataframe(df_agg_hourly.head())
        st.markdown("**Debug : Agrégation totale par magasin**")
        st.dataframe(df_agg_total)
    else:
        df_transactions = pd.DataFrame()
        df_agg_hourly = pd.DataFrame()
        df_agg_total = pd.DataFrame(columns=["store_id", "amount"]) if "store_id" in df_filtered.columns else pd.DataFrame(columns=["amount"])
        if "amount" in df_filtered.columns:
            df_agg_total.loc[0, "amount"] = df_filtered["amount"].sum()
        st.warning("Certaines colonnes nécessaires (montant, magasin, timestamp) sont manquantes.")

    # Display
    st.subheader(f"Analyse {f'du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}' if start_date else 'Globale'}")
    if user_role == ROLE_TOP_LEVEL:
        total_revenue = df_transactions["amount"].sum() if not df_transactions.empty else 0
        total_transactions = len(df_transactions) if not df_transactions.empty else 0
        avg_transaction_value = total_revenue / total_transactions if total_transactions > 0 else 0
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Chiffre d'Affaires", f"{total_revenue:,.2f} TND")
        m_col2.metric("Transactions", f"{total_transactions:,}")
        m_col3.metric("Panier Moyen", f"{avg_transaction_value:,.2f} TND")

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            if not df_agg_total.empty and "store_id" in df_agg_total.columns:
                fig_pie = px.pie(df_agg_total, values='amount', names='store_id', title="Répartition CA", hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            elif not df_agg_total.empty:
                st.write(f"Chiffre d'affaires global : {df_agg_total['amount'].sum():,.2f} TND")
        with viz_col2:
            if not df_agg_total.empty and "store_id" in df_agg_total.columns:
                fig_bar = px.bar(df_agg_total.sort_values('amount', ascending=False), x='store_id', y='amount', title="CA par Magasin")
                st.plotly_chart(fig_bar, use_container_width=True)

        if not df_agg_hourly.empty:
            fig_line = px.line(df_agg_hourly, x="hour", y="amount", color="store_id", title="Ventes par Heure", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

    else:
        # Local manager view
        if not df_agg_total.empty:
            store_data_total = df_agg_total["amount"].sum()
            total_transactions = len(df_transactions) if not df_transactions.empty else 0
            avg_transaction_value = store_data_total / total_transactions if total_transactions > 0 else 0
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Ventes", f"{store_data_total:,.2f} TND")
            m_col2.metric("Transactions", f"{total_transactions:,}")
            m_col3.metric("Panier Moyen", f"{avg_transaction_value:,.2f} TND")
        if not df_agg_hourly.empty:
            fig_bar = px.bar(df_agg_hourly, x="hour", y="amount", title=f"Ventes Horaires - {user_store}")
            st.plotly_chart(fig_bar, use_container_width=True)
def render_recommendations():
    st.header("💡 Recommandations Opérationnelles")
    if st.session_state.processed_data is None:
        st.warning("Aucune donnée disponible. Veuillez importer des données.")
        st.session_state.page = "Importer des données"
        st.rerun()
        return

    df = st.session_state.processed_data.copy()
    user_role = st.session_state.user["role"]
    user_store = st.session_state.user["store"]

    # Date Filter
    if "timestamp" in df.columns and 'date_filter' in st.session_state and len(st.session_state.date_filter) == 2:
        start_date = datetime.datetime.combine(st.session_state.date_filter[0], datetime.datetime.min.time())
        end_date = datetime.datetime.combine(st.session_state.date_filter[1], datetime.datetime.max.time())
        df_filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
        st.info(f"Recommandations pour {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}")
    else:
        df_filtered = df
        st.info("Recommandations sur toutes les données.")

    def get_recommendation(transactions, context_name):
        if transactions.empty:
            return "Pas assez de données.", None, None, None
        # Aggregate transactions by hour for recommendation
        transactions["hour"] = transactions["timestamp"].dt.hour
        data_hourly = transactions.groupby("hour")["amount"].sum().reset_index()
        if len(data_hourly) < 3:
            return "Pas assez de données horaires.", None, None, None
        hourly_avg = data_hourly["amount"].mean()
        low_hour_data = data_hourly.loc[data_hourly["amount"].idxmin()]
        low_hour = int(low_hour_data["hour"])
        low_sales = low_hour_data["amount"]
        q_key_promo = (context_name, low_hour, "promo")
        q_value_promo = st.session_state.q_table.get(q_key_promo, 0.5)
        recommendation_text = f"Activité basse à {low_hour}h ({low_sales:,.2f} TND)."
        details = f"Low hour: {low_hour}h, Sales: {low_sales:.2f}, Avg: {hourly_avg:.2f}, Q-Promo: {q_value_promo:.2f}"
        if low_sales < hourly_avg * 0.3 and q_value_promo > 0.1:
            recommendation = f"{recommendation_text} **Suggestion :** Promotion ciblée."
            action = "promo"
        else:
            recommendation = f"{recommendation_text} **Suggestion :** Surveiller."
            action = "monitor"
        return recommendation, low_hour, action, details

    recommendations = []
    if user_role == ROLE_TOP_LEVEL:
        st.subheader("Recommandations par Magasin")
        if 'store_filter' in st.session_state and st.session_state.store_filter:
            selected_stores = st.session_state.store_filter
            df_filtered = df_filtered[df_filtered['store_id'].isin(selected_stores)]
        else:
            selected_stores = df_filtered["store_id"].unique() if "store_id" in df_filtered.columns else ["Global"]

        for store in selected_stores:
            if "store_id" in df_filtered.columns:
                store_data = df_filtered[df_filtered["store_id"] == store]
            else:
                store_data = df_filtered
            if "timestamp" in store_data.columns and "amount" in store_data.columns:
                # Group items into transactions
                store_transactions = store_data.groupby(["store_id", "timestamp"])["amount"].sum().reset_index()
                recommendation, low_hour, action, details_for_log = get_recommendation(store_transactions, store)
                recommendations.append({
                    "store": store,
                    "recommendation": recommendation,
                    "low_hour": low_hour,
                    "action": action,
                    "details_for_log": details_for_log
                })

    elif user_role == ROLE_LOCAL:
        st.subheader(f"Recommandations pour {user_store}")
        store_data = df_filtered[df_filtered["store_id"] == user_store] if "store_id" in df_filtered.columns else df_filtered
        if "timestamp" in store_data.columns and "amount" in store_data.columns:
            # Group items into transactions
            store_transactions = store_data.groupby(["store_id", "timestamp"])["amount"].sum().reset_index()
            recommendation, low_hour, action, details_for_log = get_recommendation(store_transactions, user_store)
            recommendations.append({
                "store": user_store,
                "recommendation": recommendation,
                "low_hour": low_hour,
                "action": action,
                "details_for_log": details_for_log
            })

    for rec in recommendations:
        st.markdown(f"### {rec['store']}")
        st.markdown(f"> {rec['recommendation']}")
        st.markdown("---")
        st.subheader("📝 Votre Avis")
        with st.form(f"feedback_form_{rec['store']}"):
            feedback = st.radio("Pertinence ?", ("👍 Pertinente", "👎 Non pertinente"), key=f"feedback_{rec['store']}")
            comment = st.text_area("Commentaire (optionnel)", placeholder="Ex: 'Bonne idée' ou 'Déjà essayé.'", key=f"comment_{rec['store']}")
            submitted = st.form_submit_button("Envoyer")
            if submitted and rec["action"]:
                reward = 1 if feedback == "👍 Pertinente" else -1
                q_key = (rec["store"], rec["low_hour"], rec["action"])
                old_q = st.session_state.q_table.get(q_key, 0.5)
                new_q = old_q + 0.1 * (reward - old_q)
                st.session_state.q_table[q_key] = new_q
                q_info = f"Q-Value ({rec['action']} @ {rec['low_hour']}h): {old_q:.2f} -> {new_q:.2f}"
                st.success(f"Avis enregistré pour {rec['store']} : {feedback}")
                st.info(q_info)
                log_feedback(
                    user=st.session_state.user['username'],
                    store=rec["store"],
                    recommendation_details=rec["details_for_log"],
                    feedback=feedback,
                    comment=comment,
                    q_value_change=q_info
                )

# --- Sidebar Navigation ---
def render_sidebar():
    st.sidebar.header("Navigation")
    if st.session_state.authenticated:
        st.sidebar.success(f"Connecté: **{st.session_state.user['username']}**")
        st.sidebar.info(f"Rôle: {st.session_state.user['role']}")
        if st.session_state.user['store']:
            st.sidebar.info(f"Magasin: {st.session_state.user['store']}")
        if st.session_state.processed_data is not None:
            options = ["Choix", "Tableau de bord", "Recommandations", "Importer des données", "Déconnexion"]
        else:
            options = ["Importer des données", "Déconnexion"]
        page = st.sidebar.selectbox("Choisir une page", options, index=options.index(st.session_state.page) if st.session_state.page in options else 0)
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
    else:
        st.sidebar.write("Veuillez vous connecter.")

# --- Main Logic ---
render_sidebar()

if st.session_state.page == "Accueil" and not st.session_state.authenticated:
    render_homepage()
elif st.session_state.page == "Connexion" and not st.session_state.authenticated:
    render_login()
elif st.session_state.authenticated:
    if st.session_state.page == "Importer des données":
        render_data_import()
    elif st.session_state.page == "Choix":
        render_choice_page()
    elif st.session_state.page == "Tableau de bord":
        render_dashboard()
    elif st.session_state.page == "Recommandations":
        render_recommendations()
    elif st.session_state.page == "Déconnexion":
        for key in ["authenticated", "user", "processed_data", "q_table", "page"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.authenticated = False
        st.session_state.page = "Accueil"
        st.success("Déconnexion réussie.")
        st.rerun()
else:
    st.session_state.page = "Accueil"
    st.rerun()

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Plateforme B2B - PFE")
st.sidebar.write("Version 2.2")