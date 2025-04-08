import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Initialisation des états
if "data" not in st.session_state:
    st.session_state.data = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "q_table" not in st.session_state:
    st.session_state.q_table = {}

# Base d'utilisateurs simulée
users_db = {
    "admin": {"password": "admin123", "role": "top_level", "store": None},
    "manager_menzah": {"password": "pass1", "role": "local", "store": "Baristas Menzah 1"},
    "manager_jardin": {"password": "pass2", "role": "local", "store": "Baristas Jardin de Carthage"},
    "manager_marsa": {"password": "pass3", "role": "local", "store": "Baristas la Marsa"}
}

# Style CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput {background-color: #ffffff; border-radius: 5px;}
    h1 {color: #2c3e50;}
    h2 {color: #34495e;}
    .sidebar .sidebar-content {background-color: #ecf0f1;}
    </style>
""", unsafe_allow_html=True)

# Titre
st.title("Plateforme d'Optimisation Opérationnelle")

# Sidebar pour navigation
st.sidebar.header("Navigation")
if st.session_state.authenticated:
    page = st.sidebar.selectbox("Choisir une page", ["Tableau de bord", "Recommandations", "Importer des données", "Déconnexion"])
else:
    page = "Connexion"

# Vérification des identifiants
def check_credentials(username, password):
    if username in users_db and users_db[username]["password"] == password:
        return True, users_db[username]["role"], users_db[username]["store"]
    return False, None, None

# Page Connexion
if page == "Connexion" and not st.session_state.authenticated:
    st.header("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        valid, role, store = check_credentials(username, password)
        if valid:
            st.session_state.authenticated = True
            st.session_state.user = {"username": username, "role": role, "store": store}
            st.success(f"Connexion réussie ! Bienvenue, {username}.")
        else:
            st.error("Identifiants incorrects.")

# Déconnexion
elif page == "Déconnexion" and st.session_state.authenticated:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.data = None
    st.success("Déconnexion réussie.")
    st.experimental_rerun()

# Pages protégées
elif st.session_state.authenticated:
   # Page Importer des données
    if page == "Importer des données":
        st.header("Importer vos données")
        uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            delimiter = st.selectbox("Séparateur du CSV", [",", ";", "\t"], index=0)
            encoding = st.selectbox("Encodage du fichier", ["utf-8", "latin-1", "cp1252"], index=0)
            
            try:
                df = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding)
                st.write("Aperçu des données :")
                st.dataframe(df.head())
                
                st.subheader("Mapper les colonnes")
                cols = df.columns.tolist()
                date_col = st.selectbox("Colonne de date/heure", cols, key="date_col")
                amount_col = st.selectbox("Colonne de montant", cols, key="amount_col")
                store_col = st.selectbox("Colonne de magasin", cols, key="store_col")
                
                if st.button("Confirmer le mapping"):
                    df = df.rename(columns={date_col: "timestamp", amount_col: "amount", store_col: "store_id"})
                    try:
                        df["amount"] = pd.to_numeric(df["amount"], errors="raise")
                    except ValueError:
                        # Si la conversion échoue, essayer de remplacer les virgules par des points (ex. "12,5" -> "12.5")
                        try:
                            df["amount"] = df["amount"].str.replace(",", ".").astype(float)
                        except (ValueError, AttributeError):
                            st.error("Erreur : La colonne 'montant' contient des valeurs qui ne peuvent pas être converties en nombres (ex. '12.5' ou '12,5' attendus).")
                        
                    if df["timestamp"].dtype in ["int64", "float64"]:
                        df["hour"] = df["timestamp"].astype(int)
                    else:
                        try:
                            df["timestamp"] = df["timestamp"].str.replace(r'\s+', ' ', regex=True).str.strip()
                            try:
                                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M:%S")
                            except ValueError:
                                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M")
                            df["hour"] = df["timestamp"].dt.hour
                        except ValueError:
                            st.error("Erreur : Le format de la colonne date/heure doit être 'JJ/MM/AAAA HH:MM:SS' (ex. '01/08/2023 16:38:46') ou des heures numériques (ex. 12, 15, 0).")
                        else:
                            st.session_state.data = df.groupby(["store_id", "hour"])["amount"].sum().reset_index()
                            st.success("Données importées et agrégées avec succès !")
                            st.dataframe(st.session_state.data.head())
            except (pd.errors.ParserError, UnicodeDecodeError) as e:
                st.error(f"Erreur lors de la lecture du CSV : {str(e)}")
                st.write("Contenu brut du fichier pour diagnostic :")
                uploaded_file.seek(0)
                raw_content = uploaded_file.read().decode("latin-1")  # Décodage alternatif pour affichage
                st.text(raw_content)
    # Page Tableau de bord
    elif page == "Tableau de bord":
        st.header("Tableau de bord")
        if st.session_state.data is None:
            st.warning("Veuillez importer des données.")
        else:
            user_role = st.session_state.user["role"]
            user_store = st.session_state.user["store"]
            stores = st.session_state.data["store_id"].unique()

            if user_role == "top_level":
                st.subheader("Vue globale des performances")

                # Calcul des chiffres d'affaires totaux par magasin
                total_sales_by_store = st.session_state.data.groupby("store_id")["amount"].sum().reset_index()

                # 1. Pie Chart : Répartition du chiffre d'affaires par magasin
                st.write("### Répartition du chiffre d'affaires par magasin")
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                ax_pie.pie(total_sales_by_store["amount"], labels=total_sales_by_store["store_id"], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                ax_pie.axis('equal')  # Égaliser les proportions
                st.pyplot(fig_pie)

                # 2. Bar Chart : Comparaison des chiffres d'affaires totaux
                st.write("### Comparaison des chiffres d'affaires totaux")
                fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                total_sales_by_store.plot(kind="bar", x="store_id", y="amount", ax=ax_bar, color="#3498db")
                ax_bar.set_xlabel("Magasin")
                ax_bar.set_ylabel("Chiffre d'affaires (TND)")
                ax_bar.set_title("Chiffre d'affaires total par magasin")
                st.pyplot(fig_bar)

                # 3. Line Chart : Comparaison des ventes par heure pour tous les magasins
                st.write("### Ventes par heure - Comparaison entre magasins")
                fig_line, ax_line = plt.subplots(figsize=(12, 6))
                for store in stores:
                    store_data = st.session_state.data[st.session_state.data["store_id"] == store]
                    store_data.plot(x="hour", y="amount", kind="line", ax=ax_line, label=store, marker='o')
                ax_line.set_xlabel("Heure")
                ax_line.set_ylabel("Montant (TND)")
                ax_line.set_title("Ventes par heure pour tous les magasins")
                ax_line.legend(title="Magasins")
                st.pyplot(fig_line)

                # Détails par magasin (comme avant)
                for store in stores:
                    store_data = st.session_state.data[st.session_state.data["store_id"] == store]
                    total_sales = store_data["amount"].sum()
                    peak_hour = store_data.loc[store_data["amount"].idxmax(), "hour"]
                    low_hour = store_data.loc[store_data["amount"].idxmin(), "hour"]

                    st.write(f"### {store}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total des ventes", f"{total_sales:.2f} TND")
                    col2.metric("Heure de pointe", f"{peak_hour}h", f"{store_data['amount'].max():.2f} TND")
                    col3.metric("Heure creuse", f"{low_hour}h", f"{store_data['amount'].min():.2f} TND")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    store_data.plot(x="hour", y="amount", kind="bar", ax=ax, color="#3498db", title=f"Ventes par heure - {store}")
                    ax.set_xlabel("Heure")
                    ax.set_ylabel("Montant (TND)")
                    st.pyplot(fig)

            else:
                st.subheader(f"Performances de {user_store}")
                store_data = st.session_state.data[st.session_state.data["store_id"] == user_store]
                total_sales = store_data["amount"].sum()
                peak_hour = store_data.loc[store_data["amount"].idxmax(), "hour"]
                low_hour = store_data.loc[store_data["amount"].idxmin(), "hour"]

                col1, col2, col3 = st.columns(3)
                col1.metric("Total des ventes", f"{total_sales:.2f} TND")
                col2.metric("Heure de pointe", f"{peak_hour}h", f"{store_data['amount'].max():.2f} TND")
                col3.metric("Heure creuse", f"{low_hour}h", f"{store_data['amount'].min():.2f} TND")

                fig, ax = plt.subplots(figsize=(10, 5))
                store_data.plot(x="hour", y="amount", kind="bar", ax=ax, color="#e74c3c", title=f"Ventes par heure - {user_store}")
                ax.set_xlabel("Heure")
                ax.set_ylabel("Montant (TND)")
                st.pyplot(fig)

    # Page Recommandations
    elif page == "Recommandations":
        st.header("Recommandations")
        if st.session_state.data is None:
            st.warning("Veuillez importer des données.")
        else:
            user_role = st.session_state.user["role"]
            user_store = st.session_state.user["store"]

            def get_recommendation(store_data, store):
                hourly_avg = store_data["amount"].mean()
                low_hour = store_data.loc[store_data["amount"].idxmin(), "hour"]
                low_sales = store_data["amount"].min()
                
                key = (store, low_hour, "promo")
                q_value = st.session_state.q_table.get(key, 0.5)
                
                if q_value > 0.3 and low_sales < hourly_avg * 0.1:
                    recommendation = f"Baisse d’activité à {low_hour}h ({low_sales:.2f} TND). Suggestion : lancer une promotion."
                    action = "promo"
                else:
                    recommendation = f"Activité stable à {low_hour}h ({low_sales:.2f} TND). Suggestion : surveiller."
                    action = "monitor"
                return recommendation, low_hour, action

            if user_role == "top_level":
                st.subheader("Recommandations globales")
                global_data = st.session_state.data.groupby("hour")["amount"].sum().reset_index()
                recommendation, low_hour, action = get_recommendation(global_data, "global")
                st.write(recommendation)
            else:
                st.subheader(f"Recommandations pour {user_store}")
                store_data = st.session_state.data[st.session_state.data["store_id"] == user_store]
                recommendation, low_hour, action = get_recommendation(store_data, user_store)
                st.write(recommendation)

            st.subheader("Votre avis")
            feedback = st.radio("Que pensez-vous de cette recommandation ?", ("Accepter", "Rejeter"))
            comment = st.text_input("Commentaire (optionnel)")
            if st.button("Envoyer le feedback"):
                reward = 1 if feedback == "Accepter" else -1
                key = (user_store, low_hour, action) if user_role == "local" else ("global", low_hour, action)
                old_q = st.session_state.q_table.get(key, 0.5)
                st.session_state.q_table[key] = old_q + 0.1 * (reward - old_q)
                st.success(f"Feedback enregistré : {feedback} - Nouvelle Q-value : {st.session_state.q_table[key]:.2f}")

# Pied de page
st.sidebar.markdown("---")
st.sidebar.write("Projet PFE - Plateforme générique B2B")
st.sidebar.write("Développée par un stagiaire")