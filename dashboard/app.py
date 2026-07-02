import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="NewsOps Dashboard", layout="wide")
st.title("📊 AI NewsOps Platform — Monitoring Dashboard")

# URL de l'API (à adapter si besoin)
API_URL = "http://localhost:8000"

# Sidebar
st.sidebar.header("🔍 Navigation")
page = st.sidebar.selectbox("Choisir une vue", ["🏠 Accueil", "📈 Prédiction", "📊 Monitoring", "🔄 Drift"])

# --- Page Accueil ---
if page == "🏠 Accueil":
    st.markdown(f"""
    ## Bienvenue sur le dashboard de supervision

    Ce dashboard vous permet de :
    - Visualiser les prédictions en temps réel
    - Suivre les métriques du modèle
    - Détecter les dérives (drift)
    - Consulter l'historique des logs

    **API** : en cours d'exécution sur `{API_URL}`
    """)

    # Vérification de l'état de l'API
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API opérationnelle")
            st.json(health.json())
        else:
            st.error("❌ API non disponible")
    except:
        st.error("❌ Impossible de contacter l'API")

# --- Page Prédiction ---
elif page == "📈 Prédiction":
    st.header("🧪 Tester une prédiction")
    col1, col2 = st.columns(2)

    with col1:
        headline = st.text_area("Titre de l'article", "Congress passes new climate bill")
        short_desc = st.text_area("Description courte", "Senate votes in favor")
        if st.button("🚀 Prédire"):
            if headline:
                payload = {"headline": headline, "short_description": short_desc}
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Prédiction réussie")
                        st.json(result)
                        # Stocker le résultat pour l'affichage dans la colonne 2
                        st.session_state['result'] = result
                    else:
                        st.error(f"Erreur {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Erreur de connexion: {e}")
            else:
                st.warning("Veuillez saisir un titre")

    with col2:
        st.subheader("📊 Top 3 des catégories")
        if 'result' in st.session_state:
            result = st.session_state['result']
            data = pd.DataFrame(result['top3'], columns=["Catégorie", "Score"])
            fig = px.bar(data, x="Catégorie", y="Score", title="Distribution des scores")
            st.plotly_chart(fig, use_container_width=True)

# --- Page Monitoring ---
elif page == "📊 Monitoring":
    st.header("📈 Métriques du modèle")

    col1, col2, col3 = st.columns(3)
    try:
        # Récupérer les métriques depuis le fichier training_metrics.json (ou via l'API)
        # Ici on affiche des valeurs statiques pour l'exemple
        st.metric("📊 F1 Score", "0.6791")
        st.metric("🎯 Accuracy", "0.7382")
        st.metric("📈 Baseline", "0.6515")
    except:
        st.warning("Métriques non disponibles")

    # Health check du monitoring
    try:
        health = requests.get(f"{API_URL}/monitoring/health")
        if health.status_code == 200:
            data = health.json()
            st.subheader("🏥 État du monitoring")
            st.json(data)
    except:
        st.warning("Monitoring non disponible")

# --- Page Drift ---
elif page == "🔄 Drift":
    st.header("🔄 Détection de dérive (Drift)")

    if st.button("🔍 Vérifier le drift maintenant"):
        try:
            response = requests.get(f"{API_URL}/monitoring/drift")
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Analyse de drift effectuée")
                st.json(data)
            else:
                st.error(f"Erreur {response.status_code}")
        except Exception as e:
            st.error(f"Erreur de connexion: {e}")

    # Logs de drift
    st.subheader("📜 Historique des drifts")
    try:
        logs_response = requests.get(f"{API_URL}/monitoring/drift/logs?limit=10")
        if logs_response.status_code == 200:
            logs = logs_response.json()
            if logs.get('logs'):
                df_logs = pd.DataFrame(logs['logs'])
                st.dataframe(df_logs)
            else:
                st.info("Aucun log de drift trouvé")
        else:
            st.warning("Logs non disponibles")
    except:
        st.warning("Impossible de récupérer les logs")
