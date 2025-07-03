import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, isnan, count, desc, avg, max as sql_max
from datetime import datetime
from pyspark.sql import functions as F
import pandas as pd
import os
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly non disponible. Les graphiques utiliseront le style par défaut.")

# Configuration de l'environnement PySpark
os.environ["PYSPARK_PYTHON"] = "C:/Users/Lenovo/Documents/4IIR/S2/PFA/PFAI/venv/Scripts/python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:/Users/Lenovo/Documents/4IIR/S2/PFA/PFAI/venv/Scripts/python.exe"


# Configuration du style Amazon
st.set_page_config(
    layout="wide",
    page_title="Recommendation System",
    page_icon="images/amazone.png"
)


# Couleurs Amazon
AMAZON_ORANGE = "#FF9900"
AMAZON_DARK = "#131921"
AMAZON_LIGHT = "#232F3E"
AMAZON_GRAY = "#EAEDED"

# Application du style Amazon amélioré
st.markdown(f"""
    <style>
        /* Arrière-plan principal avec dégradé subtil */
        .main {{
            background: linear-gradient(135deg, {AMAZON_GRAY} 0%, #F5F5F5 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}

        /* Sidebar avec style premium */
        .sidebar .sidebar-content {{
            background: linear-gradient(180deg, {AMAZON_DARK} 0%, {AMAZON_LIGHT} 100%) !important;
            color: white;
            border-right: 3px solid {AMAZON_ORANGE};
        }}

        /* Boutons avec animations et effets */
        .stButton>button {{
            background: linear-gradient(45deg, {AMAZON_ORANGE} 0%, #FFB84D 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255,153,0,0.3);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stButton>button:hover {{
            background: linear-gradient(45deg, #e68a00 0%, {AMAZON_ORANGE} 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255,153,0,0.4);
        }}
        .stButton>button:active {{
            transform: translateY(0px);
        }}

        /* Inputs avec style moderne */
        .stTextInput>div>div>input {{
            border-radius: 8px;
            border: 2px solid #E0E0E0;
            padding: 0.7rem;
            transition: all 0.3s ease;
            background: white;
        }}
        .stTextInput>div>div>input:focus {{
            border-color: {AMAZON_ORANGE};
            box-shadow: 0 0 0 3px rgba(255,153,0,0.1);
        }}

        /* Métriques avec style carte premium */
        .metric {{
            background: linear-gradient(135deg, white 0%, #FAFAFA 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border: 1px solid #F0F0F0;
            transition: all 0.3s ease;
        }}
        .metric:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.12);
        }}

        /* DataFrames avec style élégant */
        .stDataFrame {{
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(255,153,0,0.15);
            border: 2px solid {AMAZON_ORANGE};
            overflow: hidden;
            background: white;
        }}
        .stDataFrame table {{
            border-collapse: separate;
            border-spacing: 0;
        }}
        /* En-têtes de tableaux avec style orange uniforme */
        .stDataFrame th {{
            background: linear-gradient(90deg, {AMAZON_ORANGE} 0%, #FFB84D 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            border: none !important;
        }}

        /* Tous les autres en-têtes de tableaux */
        table th, thead th, .dataframe th {{
            background: linear-gradient(90deg, {AMAZON_ORANGE} 0%, #FFB84D 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            border: none !important;
            padding: 0.75rem !important;
        }}

        /* En-têtes de colonnes Streamlit */
        .stDataFrame thead tr th {{
            background: linear-gradient(90deg, {AMAZON_ORANGE} 0%, #FFB84D 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }}

        /* En-têtes pour les métriques et autres composants */
        .metric h3, .stMetric h3 {{
            color: {AMAZON_ORANGE} !important;
        }}

        /* Alertes avec style moderne */
        .stAlert {{
            border-left: 5px solid {AMAZON_ORANGE};
            border-radius: 8px;
            background: linear-gradient(90deg, #FFF8F0 0%, white 100%);
            box-shadow: 0 4px 15px rgba(255,153,0,0.1);
        }}

        /* Selectbox avec style premium */
        .stSelectbox [data-baseweb="select"] > div {{
            background: linear-gradient(135deg, {AMAZON_ORANGE} 0%, #FFB84D 100%) !important;
            border-radius: 8px !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(255,153,0,0.2);
            transition: all 0.3s ease;
        }}
        .stSelectbox [data-baseweb="select"] input {{
            background: transparent !important;
            color: white !important;
            font-weight: 600 !important;
        }}
        .stSelectbox [data-baseweb="select"]:hover > div {{
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(255,153,0,0.3);
        }}

        /* Titres avec emoji - style premium */
        .st-emojified-title {{
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
            background: linear-gradient(135deg, {AMAZON_ORANGE} 0%, #FFB84D 50%, #FF6B35 100%);
            border-radius: 15px;
            padding: 1rem 2rem;
            box-shadow: 0 10px 30px rgba(255,153,0,0.3);
            margin-bottom: 2rem;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        }}
        .st-emojified-title::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}
        .st-emojified-title:hover::before {{
            left: 100%;
        }}

        /* Sous-titres avec style moderne */
        .st-emojified-subheader {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {AMAZON_DARK};
            background: linear-gradient(135deg, #FFF8F0 0%, white 100%);
            border-left: 6px solid {AMAZON_ORANGE};
            border-radius: 8px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(255,153,0,0.1);
            transition: all 0.3s ease;
        }}
        .st-emojified-subheader:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(255,153,0,0.15);
        }}

        /* Cartes pour graphiques */
        .stCardBox {{
            background: linear-gradient(135deg, white 0%, #FAFAFA 100%);
            border: 2px solid {AMAZON_ORANGE};
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(255,153,0,0.15);
            padding: 2rem;
            margin-bottom: 2rem;
            transition: all 0.3s ease;
        }}
        .stCardBox:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(255,153,0,0.2);
        }}

        /* Séparateur orange avec animation */
        .orange-separator {{
            height: 8px;
            border: none;
            background: linear-gradient(90deg, {AMAZON_ORANGE} 0%, #FFB84D 50%, {AMAZON_ORANGE} 100%);
            border-radius: 4px;
            margin: 2rem 0;
            position: relative;
            overflow: hidden;
        }}
        .orange-separator::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: shimmer 2s infinite;
        }}
        @keyframes shimmer {{
            0% {{ left: -100%; }}
            100% {{ left: 100%; }}
        }}

        /* Radio buttons avec style */
        .stRadio>div>label {{
            color: {AMAZON_DARK} !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
        }}
        .stRadio>div>label:hover {{
            color: {AMAZON_ORANGE} !important;
        }}

        /* Colonnes avec espacement amélioré */
        .css-1v3fvcr {{
            padding: 1.5rem;
        }}

        /* Sidebar title avec style */
        .sidebar .sidebar-content h1 {{
            color: {AMAZON_ORANGE} !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            font-weight: 800;
        }}

        /* Amélioration des expanders */
        .streamlit-expanderHeader {{
            background: linear-gradient(135deg, #FFF8F0 0%, white 100%);
            border: 1px solid {AMAZON_ORANGE};
            border-radius: 8px;
            font-weight: 600;
        }}

        /* Style pour les download buttons */
        .stDownloadButton>button {{
            background: linear-gradient(45deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(40,167,69,0.3);
        }}
        .stDownloadButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(40,167,69,0.4);
        }}
    </style>
""", unsafe_allow_html=True)

# Remplacement de st.markdown("---") par un séparateur stylisé
def orange_separator():
    st.markdown('<hr class="orange-separator">', unsafe_allow_html=True)

# Logo Amazon dans la sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=350)

# Fonction pour convertir timestamp en date
def timestamp_to_date(timestamp):
    try:
        if timestamp:
            return datetime.fromtimestamp(float(timestamp)).strftime('%d/%m/%Y')
        return "N/A"
    except:
        return "N/A"

# Fonction pour afficher les produits populaires
def show_popular_products(df):
    st.info("ℹ️ Voici les produits les mieux notés:")
    top_products = df.groupBy("product_id", "title") \
        .agg(avg("rating").alias("avg_rating"), count("rating").alias("count")) \
        .filter(col("count") >= 10) \
        .sort(desc("avg_rating")) \
        .limit(10) \
        .toPandas()

    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True,
        column_config={
            "product_id": "ID Produit",
            "title": "Nom du produit",
            "avg_rating": st.column_config.NumberColumn(
                "Note moyenne",
                format="%.2f",
                min_value=0,
                max_value=5
            ),
            "count": "Nombre d'avis"
        }
    )

# Initialisation de Spark
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("AmazonRecommender") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

spark = get_spark()

# Chargement des données avec sélection de catégorie
@st.cache_resource
def load_and_clean_data(category):
    if category == "Industrial":
        file_path = "data/processed/industrial_cleaned.csv"
    else:
        raise ValueError("Catégorie non valide")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")

    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Conversion et nettoyage
    df = df.withColumn("rating", col("rating").cast("float"))
    df_clean = df.filter(~isnan(col("rating")) & col("rating").isNotNull())
    df_clean = df_clean.dropna(subset=["product_id", "user_id", "rating"])

    return df_clean

# Modèle ALS
@st.cache_resource
def train_als_model(_df_clean):
    df_for_model = _df_clean.select("user_id", "product_id", "rating")

    # Indexation
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
    product_indexer = StringIndexer(inputCol="product_id", outputCol="product_index")

    df_indexed = user_indexer.fit(df_for_model).transform(df_for_model)
    df_indexed = product_indexer.fit(df_indexed).transform(df_indexed)
    df_indexed = df_indexed.withColumn("user_index", col("user_index").cast("float"))

    # Division des données
    train_df, test_df = df_indexed.randomSplit([0.8, 0.2], seed=42)

    # Entraînement
    als = ALS(
        userCol="user_index",
        itemCol="product_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=2,
        maxIter=3,
        regParam=0.60
    )

    model = als.fit(train_df)

    # Calcul du RMSE
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions.na.drop())
    # Suppression de l'affichage du RMSE dans la page d'accueil
    # st.metric("📉 RMSE du modèle ALS", f"{rmse:.4f}")

    # Tables de référence
    products_lookup = df_indexed.join(
        _df_clean.select("product_id", "title"),
        "product_id",
        "left"
    ).select("product_index", "product_id", "title").distinct().toPandas()

    users_lookup = df_indexed.join(
        _df_clean.select("user_id", "user_name"),
        "user_id",
        "left"
    ).select("user_index", "user_id", "user_name").distinct().toPandas()

    return model, products_lookup, users_lookup

# Sélection de la catégorie dans la sidebar
st.sidebar.title("Catégorie de produits")
category = st.sidebar.selectbox(
    "",
    ["Industrial"],
    index=0,
    key="category_select"
)


 
# Chargement des données pour la catégorie sélectionnée
try:
    df_clean = load_and_clean_data(category)
except Exception as e:
    st.error(f"Erreur de chargement des données: {str(e)}")
    st.stop()

# Interface
st.sidebar.title("Navigation Bar")
page = st.sidebar.selectbox(
    "Menu",
    ["🏠 Accueil", "📊 Statistiques", "⚙️ Gestion de l’Algorithme", "🧩 Analyse Avancée"],
    key="nav_menu"
)
#produit ----------------------------------------------------------------------------------------
if page == "🏠 Accueil":
    st.markdown('<div class="st-emojified-title">🏠 Dash Bord - {}</div>'.format(category), unsafe_allow_html=True)

    search_option = st.selectbox(
        "🔍 Mode de recherche:",
        ["👤 Par utilisateur", "📦 Par produit"],
        key="search_option"
    )
    orange_separator()
# UTILISATEUR --------------------------------------------------------------------------------------
    if search_option == "👤 Par utilisateur":
        # Recherche utilisateur
        users_pd = df_clean.select("user_id", "user_name").distinct().toPandas()
        users_pd["label"] = users_pd["user_name"] + " (" + users_pd["user_id"] + ")"

        search_term = st.text_input("🔎 Rechercher un utilisateur:", key="user_search")
        filtered_users = users_pd[users_pd["label"].str.contains(search_term, case=False, na=False)]

        if not filtered_users.empty:
            selected_user_label = st.selectbox("Sélectionner un utilisateur:", filtered_users["label"], key="user_select")
            user_name = selected_user_label.split(" (")[0]
            selected_user = selected_user_label.split("(")[1].replace(")", "").strip()

            # Statistiques utilisateur
            user_data = df_clean.filter(col("user_id") == selected_user)

            total_reviews = user_data.count()
            avg_rating = user_data.select(avg("rating")).collect()[0][0] if total_reviews > 0 else 0
            last_review = user_data.select(sql_max("review_time")).collect()[0][0] if total_reviews > 0 else None

            user_stats = {
                "total_reviews": total_reviews,
                "avg_rating": avg_rating,
                "last_review_date": timestamp_to_date(last_review)
            }

            # Affichage des statistiques avec style amélioré
            st.markdown(f'<div class="st-emojified-subheader">👤 Profil Utilisateur: {user_name}</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric">
                    <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">📊 Avis Total</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{}</p>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Contributions</p>
                </div>
                """.format(user_stats['total_reviews']), unsafe_allow_html=True)
            with col2:
                avg_display = f"{user_stats['avg_rating']:.1f}/5" if user_stats['avg_rating'] else "N/A"
                st.markdown("""
                <div class="metric">
                    <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">⭐ Note Moyenne</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{}</p>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Satisfaction</p>
                </div>
                """.format(avg_display), unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric">
                    <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">📅 Dernier Avis</h3>
                    <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{}</p>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Activité récente</p>
                </div>
                """.format(user_stats['last_review_date']), unsafe_allow_html=True)

            # Historique des avis
            orange_separator()
            st.markdown('<div class="st-emojified-subheader">📋 Historique des Avis</div>', unsafe_allow_html=True)
            if user_stats['total_reviews'] > 0:
                history_cols = ["product_id", "title", "rating", "helpfulness", "review_summary", "review_time", "review_text"]
                user_history = user_data.select(*history_cols).sort(desc("review_time")).toPandas()
                user_history["date"] = user_history["review_time"].apply(timestamp_to_date)
                display_history = user_history[["product_id", "title", "rating", "helpfulness", "review_summary", "date"]]
                display_history["Détails"] = [f"Voir" for _ in range(len(display_history))]

                selected_detail = st.selectbox(
                    "Sélectionner un avis pour voir les détails:",
                    [f"{row['title']} ({row['product_id']}) - {row['date']}" for _, row in display_history.iterrows()],
                    key="history_detail_select"
                ) if not display_history.empty else None

                st.dataframe(
                    display_history.drop(columns=["Détails"]),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "product_id": "ID Produit",
                        "title": "Nom du produit",
                        "rating": st.column_config.NumberColumn(
                            "Note",
                            format="%.1f",
                            min_value=0,
                            max_value=5
                        ),
                        "helpfulness": "Utilité",
                        "review_summary": "Résumé avis",
                        "date": "Date"
                    }
                )

                if selected_detail:
                    idx = [f"{row['title']} ({row['product_id']}) - {row['date']}" for _, row in display_history.iterrows()].index(selected_detail)
                    detail_row = user_history.iloc[idx]
                    with st.expander("🔍 Détails de l'avis sélectionné", expanded=True):
                        # Utilisation de colonnes et composants Streamlit natifs pour éviter les problèmes HTML
                        st.markdown("### 📦 Informations du Produit")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**🏷️ Produit :** {detail_row['title']}")
                            st.markdown(f"**⭐ Note :** {detail_row['rating']}/5")
                            st.markdown(f"**💰 Prix :** {detail_row.get('price', 'Non disponible')}")
                        with col2:
                            st.markdown(f"**🆔 ID :** `{detail_row['product_id']}`")
                            st.markdown(f"**📅 Date :** {detail_row['date']}")

                        st.markdown("---")
                        st.markdown("### 💬 Contenu de l'Avis")

                        st.markdown(f"**📝 Résumé :** *{detail_row['review_summary']}*")
                        st.markdown(f"**👍 Utilité :** {detail_row['helpfulness']}")

                        st.markdown("**📖 Description complète :**")
                        st.info(detail_row.get('review_text', 'Non disponible'))

                csv_history = user_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger l'historique complet",
                    data=csv_history,
                    file_name=f"historique_{selected_user}_{category}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ Aucun avis trouvé pour cet utilisateur")

            # Recommandations personnalisées
            orange_separator()
            st.markdown('<div class="st-emojified-subheader">✨ Recommandations Personnalisées</div>', unsafe_allow_html=True)
            try:
                if user_stats['total_reviews'] > 0:
                    with st.spinner('Entraînement du modèle...'):
                        model, products_lookup, users_lookup = train_als_model(df_clean)

                    user_index_row = users_lookup[users_lookup['user_id'] == selected_user]
                    if not user_index_row.empty:
                        user_index = float(user_index_row.iloc[0]['user_index'])
                        user_df = spark.createDataFrame([(user_index,)], ["user_index"])
                        try:
                            recs = model.recommendForUserSubset(user_df, 10)
                            if recs.count() > 0:
                                recs_pd = recs.toPandas()
                                recommendations = []
                                for _, row in recs_pd.iterrows():
                                    for rec in row['recommendations']:
                                        product_match = products_lookup[products_lookup['product_index'] == rec['product_index']]
                                        if not product_match.empty:
                                            product_info = product_match.iloc[0]
                                            recommendations.append({
                                                'product_id': product_info['product_id'],
                                                'title': product_info.get('title', 'Titre non disponible'),
                                                'score': rec['rating']
                                            })
                                if recommendations:
                                    final_recs = pd.DataFrame(recommendations)
                                    final_recs["score"] = final_recs["score"].round(2)
                                    final_recs["Détails"] = [f"Voir" for _ in range(len(final_recs))]

                                    selected_rec_detail = (
                                        st.selectbox(
                                            "Sélectionner une recommandation pour voir les détails:",
                                            [f"{row['title']} ({row['product_id']})" for _, row in final_recs.iterrows()],
                                            key="rec_detail_select"
                                        ) if not final_recs.empty else None
                                    )

                                    st.dataframe(
                                        final_recs.drop(columns=["Détails"]),
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "product_id": "ID Produit",
                                            "title": "Nom du produit",
                                            "score": st.column_config.NumberColumn(
                                                "Score prédit",
                                                format="%.2f",
                                                min_value=0,
                                                max_value=5
                                            )
                                        }
                                    )

                                    if selected_rec_detail:
                                        idx = [f"{row['title']} ({row['product_id']})" for _, row in final_recs.iterrows()].index(selected_rec_detail)
                                        rec_row = final_recs.iloc[idx]
                                        # Recherche d'infos supplémentaires dans df_clean
                                        product_info = df_clean.filter(col("product_id") == rec_row['product_id']).select(
                                            "title", "product_id", "rating", "review_time", "review_summary", "review_text", "helpfulness", "price"
                                        ).sort(desc("review_time")).toPandas()
                                        with st.expander("✨ Détails de la recommandation", expanded=True):
                                            if not product_info.empty:
                                                avg_rating = product_info['rating'].mean()
                                                stars = "⭐" * int(avg_rating) + "☆" * (5 - int(avg_rating))

                                                st.markdown("### 🎯 Produit Recommandé")

                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.markdown(f"**🏷️ Produit :** {product_info.iloc[0]['title']}")
                                                    st.markdown(f"**⭐ Note moyenne :** {avg_rating:.2f}/5 {stars}")
                                                    st.markdown(f"**💰 Prix :** {product_info.iloc[0].get('price', 'Non disponible')}")
                                                with col2:
                                                    st.markdown(f"**🆔 ID :** `{product_info.iloc[0]['product_id']}`")
                                                    st.markdown(f"**🔮 Score recommandation :** {rec_row['score']:.2f}")

                                                st.markdown(f"**📅 Dernier avis :** {timestamp_to_date(product_info.iloc[0]['review_time'])}")

                                                st.markdown("---")
                                                st.markdown("### 💭 Avis Récent")

                                                st.markdown(f"**📝 Résumé :** *{product_info.iloc[0]['review_summary']}*")
                                                st.markdown(f"**👍 Utilité :** {product_info.iloc[0]['helpfulness']}")

                                                st.markdown("**📖 Description complète :**")
                                                st.success(product_info.iloc[0].get('review_text', 'Non disponible'))
                                            else:
                                                st.info("ℹ️ Aucun détail supplémentaire disponible pour ce produit.")

                                    csv_rec = final_recs.drop(columns=["Détails"]).to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "📥 Télécharger les recommandations",
                                        data=csv_rec,
                                        file_name=f"recommandations_{selected_user}_{category}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.warning("⚠️ Aucune recommandation exploitable.")
                            else:
                                st.warning("⚠️ Le modèle n'a rien retourné.")
                        except Exception as e:
                            st.error(f"Erreur dans la génération des recommandations : {str(e)}")
                            show_popular_products(df_clean)
                    else:
                        st.warning("⚠️ Utilisateur non présent dans le modèle.")
                        show_popular_products(df_clean)
                else:
                    st.warning("⚠️ Pas assez d'avis pour recommander.")
                    show_popular_products(df_clean)
            except Exception as e:
                st.error(f"❌ Erreur système : {str(e)}")
                show_popular_products(df_clean)
        else:
            st.info("ℹ️ Aucun utilisateur trouvé.")
    # produit -----------------------------------------------------------------------------------------------------------------
    else:
        # Recherche par produit
        products_pd = df_clean.select("product_id", "title").distinct().toPandas()
        products_pd["label"] = products_pd["title"] + " (" + products_pd["product_id"] + ")"

        search_term = st.text_input("🔎 Rechercher un produit:", key="product_search")
        filtered_products = products_pd[products_pd["label"].str.contains(search_term, case=False, na=False)]

        if not filtered_products.empty:
            selected_product_label = st.selectbox("Sélectionner un produit:", filtered_products["label"], key="product_select")
            product_title = selected_product_label.split(" (")[0]
            selected_product = selected_product_label.split("(")[1].replace(")", "").strip()

            # Statistiques produit
            product_data = df_clean.filter(col("product_id") == selected_product)
            product_stats = product_data.agg(
                count("rating").alias("total_reviews"),
                avg("rating").alias("avg_rating"),
                sql_max("review_time").alias("last_review")
            ).collect()[0]

            # Affichage des statistiques avec style amélioré
            st.markdown(f'<div class="st-emojified-subheader">📦 Fiche Produit: {product_title}</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric">
                    <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">📊 Avis Total</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{}</p>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Évaluations</p>
                </div>
                """.format(product_stats['total_reviews']), unsafe_allow_html=True)
            with col2:
                avg_display = f"{product_stats['avg_rating']:.1f}/5" if product_stats['avg_rating'] else "N/A"
                st.markdown("""
                <div class="metric">
                    <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">⭐ Note Moyenne</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{}</p>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Qualité</p>
                </div>
                """.format(avg_display), unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric">
                    <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">📅 Dernier Avis</h3>
                    <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{}</p>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Activité récente</p>
                </div>
                """.format(timestamp_to_date(product_stats['last_review'])), unsafe_allow_html=True)

            st.markdown("---")

            # Détails des avis
            orange_separator()
            st.markdown('<div class="st-emojified-subheader">💬 Détails des Avis</div>', unsafe_allow_html=True)
            if product_stats['total_reviews'] > 0:
                product_details = product_data.select(
                    "user_name", "rating", "helpfulness", "review_summary", "review_text", "review_time", "price"
                ).sort(desc("review_time")).toPandas()

                product_details["date"] = product_details["review_time"].apply(timestamp_to_date)
                display_details = product_details[["user_name", "rating", "helpfulness", "review_summary", "review_text", "date"]]

                st.dataframe(
                    display_details,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "user_name": "Utilisateur",
                        "rating": st.column_config.NumberColumn(
                            "Note",
                            format="%.1f",
                            min_value=0,
                            max_value=5
                        ),
                        "helpfulness": "Utilité",
                        "review_summary": "Résumé",
                        "review_text": "Texte complet",
                        "date": "Date"
                    }
                )

                # Ajout de l'option Détails pour chaque avis produit
                display_details["Détails"] = [f"Voir" for _ in range(len(display_details))]
                select_options = [f"{row['user_name']} - {row['date']}" for _, row in display_details.iterrows()]
                selected_detail = st.selectbox(
                    "Sélectionner un avis pour voir les détails:",
                    select_options,
                    key="product_history_detail_select"
                ) if not display_details.empty else None

                if selected_detail:
                    idx = select_options.index(selected_detail)
                    detail_row = product_details.iloc[idx]
                    with st.expander("🔍 Détails de l'avis sélectionné", expanded=True):
                        st.markdown("### 👤 Informations de l'Avis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**👤 Utilisateur :** {detail_row['user_name']}")
                            st.markdown(f"**⭐ Note :** {detail_row['rating']}/5")
                            st.markdown(f"**💰 Prix :** {detail_row.get('price', 'Non disponible')}")
                        with col2:
                            st.markdown(f"**📅 Date :** {detail_row['date']}")
                            st.markdown(f"**👍 Utilité :** {detail_row['helpfulness']}")
                        st.markdown("---")
                        st.markdown("### 💬 Contenu de l'Avis")
                        st.markdown(f"**📝 Résumé :** *{detail_row['review_summary']}*")
                        st.markdown("**📖 Description complète :**")
                        st.info(detail_row.get('review_text', 'Non disponible'))

                csv_details = product_details.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Télécharger les détails",
                    data=csv_details,
                    file_name=f"produit_{selected_product}_{category}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ Aucun avis trouvé pour ce produit")

            # Produits similaires
            orange_separator()
            st.markdown('<div class="st-emojified-subheader">🔗 Produits Similaires</div>', unsafe_allow_html=True)
            similar_products = df_clean.filter(col("product_id") != selected_product) \
                .groupBy("product_id", "title") \
                .agg(
                    count("rating").alias("total_reviews"),
                    avg("rating").alias("avg_rating")
                ) \
                .sort(desc("avg_rating")) \
                .limit(10) \
                .toPandas()

            st.dataframe(
                similar_products,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "product_id": "ID Produit",
                    "title": "Nom du produit",
                    "total_reviews": "Nombre d'avis",
                    "avg_rating": st.column_config.NumberColumn(
                        "Note moyenne",
                        format="%.1f",
                        min_value=0,
                        max_value=5
                    )
                }
            )

            csv_similar = similar_products.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Télécharger les similaires",
                data=csv_similar,
                file_name=f"similaires_{selected_product}_{category}.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Aucun produit trouvé.")
# statistique -------------------------------------------------------------------------------------------------------------------
elif page == "📊 Statistiques":
    st.markdown(f'<div class="st-emojified-title">📈 Tableau de Bord Statistiques - {category}</div>', unsafe_allow_html=True)

    try:
        total_users = df_clean.select("user_id").distinct().count()
        total_products = df_clean.select("product_id").distinct().count()
        total_reviews = df_clean.count()

        # Métriques principales avec style amélioré
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric">
                <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">👥 Utilisateurs Uniques</h3>
                <p style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{:,}</p>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Communauté active</p>
            </div>
            """.format(total_users), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric">
                <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">📦 Produits Uniques</h3>
                <p style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{:,}</p>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Catalogue diversifié</p>
            </div>
            """.format(total_products), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric">
                <h3 style="color: #FF9900; margin: 0; font-size: 1.2rem;">📝 Évaluations Totales</h3>
                <p style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{:,}</p>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Données collectées</p>
            </div>
            """.format(total_reviews), unsafe_allow_html=True)

        orange_separator()
        st.markdown('<div class="st-emojified-subheader">📊 Distribution des Notes (1 à 5) ⭐</div>', unsafe_allow_html=True)

        rating_dist = df_clean.groupBy("rating") \
            .count() \
            .sort("rating") \
            .toPandas()

        all_ratings = pd.DataFrame({'rating': [1.0, 2.0, 3.0, 4.0, 5.0]})
        rating_dist = all_ratings.merge(rating_dist, on='rating', how='left').fillna(0)

        rating_dist["rating"] = rating_dist["rating"].astype(int)
        rating_dist["count"] = rating_dist["count"].astype(int)

        # Ajout d'emojis et couleurs pour l'histogramme
        rating_dist["emoji"] = rating_dist["rating"].map({
            1: "😞", 2: "😐", 3: "🙂", 4: "😊", 5: "🤩"
        })
        rating_dist["label"] = rating_dist["emoji"] + " " + rating_dist["rating"].astype(str) + " étoiles"

        # Graphique avec couleurs personnalisées
        import plotly.express as px
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#32CD32']
        fig_rating = px.bar(
            rating_dist,
            x='rating',
            y='count',
            color='rating',
            color_continuous_scale=colors,
            labels={'rating': '⭐ Note', 'count': '📊 Nombre d\'avis'},
            text='emoji'
        )
        fig_rating.update_traces(textposition='outside')
        fig_rating.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            title_font_size=16
        )
        st.plotly_chart(fig_rating, use_container_width=True)

        st.dataframe(
            rating_dist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rating": "Note",
                "count": "Nombre d'avis"
            }
        )

        csv_rating = rating_dist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger la distribution",
            data=csv_rating,
            file_name=f"distribution_notes_{category}.csv",
            mime="text/csv"
        )

        orange_separator()
        st.markdown('<div class="st-emojified-subheader">🏆 Top 10 Produits les Mieux Notés</div>', unsafe_allow_html=True)
        top_products = df_clean.groupBy("product_id", "title") \
            .agg(
                count("rating").alias("nombre_avis"),
                avg("rating").alias("note_moyenne")
            ) \
            .filter(col("nombre_avis") >= 10) \
            .sort(desc("note_moyenne")) \
            .limit(10) \
            .toPandas()

        st.dataframe(
            top_products,
            use_container_width=True,
            hide_index=True,
            column_config={
                "product_id": "ID Produit",
                "title": "Nom du produit",
                "nombre_avis": "Nombre d'avis",
                "note_moyenne": st.column_config.NumberColumn(
                    "Note moyenne",
                    format="%.2f",
                    min_value=0,
                    max_value=5
                )
            }
        )

        csv_top = top_products.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger le top produits",
            data=csv_top,
            file_name=f"top_produits_{category}.csv",
            mime="text/csv"
        )

        orange_separator()
        st.markdown('<div class="st-emojified-subheader">📥 Export des Données</div>', unsafe_allow_html=True)
        sample_data = df_clean.limit(10000).toPandas()
        csv_all = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger un échantillon",
            data=csv_all,
            file_name=f"donnees_{category}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erreur lors du calcul des statistiques: {str(e)}")
elif page == "⚙️ Gestion de l’Algorithme":#----------------------------------------------------------------------------------------
    # Page de gestion de l'algorithme
    st.title("⚙️ Gestion de l’Algorithme - Statistiques et Performances")
    try:
        # Statistiques générales
        total_users = df_clean.select("user_id").distinct().count()
        total_products = df_clean.select("product_id").distinct().count()
        total_reviews = df_clean.count()
        avg_rating = df_clean.select(avg("rating")).collect()[0][0]
        min_rating = df_clean.select(F.min("rating")).collect()[0][0]
        max_rating = df_clean.select(F.max("rating")).collect()[0][0]

        # Métriques avec style premium
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric">
                <h3 style="color: #FF9900; margin: 0; font-size: 1rem;">👥 Clients Uniques</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.3rem 0; color: #232F3E;">{:,}</p>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">Base utilisateurs</p>
            </div>
            """.format(total_users), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric">
                <h3 style="color: #FF9900; margin: 0; font-size: 1rem;">📦 Produits Uniques</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.3rem 0; color: #232F3E;">{:,}</p>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">Catalogue</p>
            </div>
            """.format(total_products), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric">
                <h3 style="color: #FF9900; margin: 0; font-size: 1rem;">📝 Nombre d'Avis</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.3rem 0; color: #232F3E;">{:,}</p>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">Dataset</p>
            </div>
            """.format(total_reviews), unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric">
                <h3 style="color: #FF9900; margin: 0; font-size: 1rem;">⭐ Note Moyenne</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.3rem 0; color: #232F3E;">{:.2f}/5</p>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">Satisfaction globale</p>
            </div>
            """.format(avg_rating), unsafe_allow_html=True)

        orange_separator()
        st.markdown('<div class="st-emojified-subheader">📊 Distribution des Notes (Histogramme ) </div>', unsafe_allow_html=True)
        rating_dist = df_clean.groupBy("rating").count().sort("rating").toPandas()

        # Ajout d'emojis et couleurs pour l'histogramme
        rating_dist["emoji"] = rating_dist["rating"].map({
            1.0: "😞", 2.0: "😐", 3.0: "🙂", 4.0: "😊", 5.0: "🤩"
        })

        # Graphique coloré avec emojis
        import plotly.express as px
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#32CD32']
        fig_rating_algo = px.bar(
            rating_dist,
            x='rating',
            y='count',
            color='rating',
            color_discrete_sequence=colors,
            title="🏆 Distribution des Notes",
            labels={'rating': '⭐ Note', 'count': '📊 Nombre d\'avis'},
            text='emoji'
        )
        fig_rating_algo.update_traces(textposition='outside', textfont_size=20)
        fig_rating_algo.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            title_font_size=16,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        st.plotly_chart(fig_rating_algo, use_container_width=True)


        orange_separator()
        st.markdown('<div class="st-emojified-subheader">👥 Répartition des Avis par Utilisateur (Top 10) 🏆</div>', unsafe_allow_html=True)
        # Récupérer les noms d'utilisateurs pour le top 10, sans ID
        user_review_counts = df_clean.groupBy("user_name").count().sort(desc("count")).limit(10).toPandas()

        # Graphique coloré pour les utilisateurs
        import plotly.express as px
        fig_users = px.bar(
            user_review_counts,
            x='user_name',
            y='count',
            color='count',
            color_continuous_scale=['#FFE5B4', '#FF9900', '#FF6B35'],
            title="🏆 Top 10 des Utilisateurs les Plus Actifs",
            labels={'user_name': '👤 Utilisateur', 'count': '📊 Nombre d\'avis'}
        )
        fig_users.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_users, use_container_width=True)
        st.dataframe(
            user_review_counts[["user_name", "count"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "user_name": "Utilisateur",
                "count": "Nombre d'avis"
            }
        )

        st.markdown("---")
        st.subheader("Performances du modèle ALS")
        # Entraînement du modèle et calcul des métriques
        import time
        model, products_lookup, users_lookup = train_als_model(df_clean)
        df_for_model = df_clean.select("user_id", "product_id", "rating")
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
        product_indexer = StringIndexer(inputCol="product_id", outputCol="product_index")
        df_indexed = user_indexer.fit(df_for_model).transform(df_for_model)
        df_indexed = product_indexer.fit(df_indexed).transform(df_indexed)
        df_indexed = df_indexed.withColumn("user_index", col("user_index").cast("float"))
        train_df, test_df = df_indexed.randomSplit([0.8, 0.2], seed=42)
        als = ALS(
            userCol="user_index",
            itemCol="product_index",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
            rank=2,
            maxIter=3,
            regParam=0.60
        )
        model = als.fit(train_df)
        predictions = model.transform(test_df).na.drop()
        evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        # Pour précision, rappel, F1, on doit binariser les notes (ex: >=3 = positif)
        pred_pd = predictions.select("rating", "prediction").toPandas().dropna()
        pred_pd["pred_label"] = (pred_pd["prediction"] >= 3).astype(int)
        pred_pd["true_label"] = (pred_pd["rating"] >= 3).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(pred_pd["true_label"], pred_pd["pred_label"])
        recall = recall_score(pred_pd["true_label"], pred_pd["pred_label"])
        f1 = f1_score(pred_pd["true_label"], pred_pd["pred_label"])

        # Ajout : Graphique en courbe de l'évolution du RMSE et MAE sur plusieurs itérations
        st.markdown("---")
        st.subheader("Évolution du modèle (RMSE & MAE par itération)")
        max_iters = 8
        rmse_list = []
        mae_list = []
        for i in range(1, max_iters+1):
            als_iter = ALS(
                userCol="user_index",
                itemCol="product_index",
                ratingCol="rating",
                coldStartStrategy="drop",
                nonnegative=True,
                rank=2,
                maxIter=3,
                regParam=0.60
            )
            model_iter = als_iter.fit(train_df)
            preds_iter = model_iter.transform(test_df).na.drop()
            rmse_iter = evaluator_rmse.evaluate(preds_iter)
            mae_iter = evaluator_mae.evaluate(preds_iter)
            rmse_list.append(rmse_iter)
            mae_list.append(mae_iter)
        import plotly.graph_objects as go
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=list(range(1, max_iters+1)), y=rmse_list, mode='lines+markers', name='RMSE'))
        fig_curve.add_trace(go.Scatter(x=list(range(1, max_iters+1)), y=mae_list, mode='lines+markers', name='MAE'))
        fig_curve.update_layout(title="Évolution du RMSE et MAE selon le nombre d'itérations", xaxis_title="Itérations", yaxis_title="Score", legend_title="Métrique")
        st.plotly_chart(fig_curve, use_container_width=True)

        # Ajout : Mesure du temps de réponse/latence pour générer une recommandation
        st.markdown("---")
        st.subheader("Temps de réponse moyen pour une recommandation")
        import numpy as np
        if not users_lookup.empty:
            user_indices = users_lookup["user_index"].sample(n=min(10, len(users_lookup)), random_state=42).tolist()
            times = []
            for idx in user_indices:
                user_df = spark.createDataFrame([(float(idx),)], ["user_index"])
                start = time.time()
                try:
                    _ = model.recommendForUserSubset(user_df, 10)
                except Exception:
                    pass
                end = time.time()
                times.append(end - start)
            avg_time = np.mean(times)
            st.metric("Temps moyen (s)", f"{avg_time:.3f}")
            st.caption("Temps moyen pour générer une recommandation pour un utilisateur (échantillon de 10 utilisateurs)")
        else:
            st.info("Pas d'utilisateurs pour mesurer la latence.")

        # Graphe radar des métriques
        orange_separator()
        st.markdown('<div class="st-emojified-subheader">🎯 Synthèse détaillée des performances du modèle ALS</div>', unsafe_allow_html=True)
        st.markdown("""
        **📊 Métriques de performance avec explications détaillées :**
        """)

        # Métriques stylisées avec couleurs et emojis
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric" style="background: linear-gradient(135deg, #FFE5E5 0%, white 100%); border-left: 5px solid #FF6B6B;">
                <h3 style="color: #FF6B6B; margin: 0; font-size: 1.2rem;">📉 RMSE (Root Mean Square Error)</h3>
                <p style="font-size: 2.2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{rmse:.6f}</p>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">🎯 Erreur quadratique moyenne - Plus bas = mieux</p>
                <div style="background: #FF6B6B; height: 3px; width: {min(100, (1-rmse/5)*100)}%; border-radius: 2px; margin-top: 0.5rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric" style="background: linear-gradient(135deg, #E5F3FF 0%, white 100%); border-left: 5px solid #4A90E2;">
                <h3 style="color: #4A90E2; margin: 0; font-size: 1.2rem;">📊 MAE (Mean Absolute Error)</h3>
                <p style="font-size: 2.2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{mae:.6f}</p>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">🎯 Erreur absolue moyenne - Plus bas = mieux</p>
                <div style="background: #4A90E2; height: 3px; width: {min(100, (1-mae/5)*100)}%; border-radius: 2px; margin-top: 0.5rem;"></div>
            </div>
            """, unsafe_allow_html=True)

        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown(f"""
            <div class="metric" style="background: linear-gradient(135deg, #E8F5E8 0%, white 100%); border-left: 5px solid #28A745;">
                <h3 style="color: #28A745; margin: 0; font-size: 1.1rem;">🎯 Précision</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{precision:.4f}</p>
                <p style="color: #666; margin: 0; font-size: 0.85rem;">✅ Recommandations positives correctes</p>
                <div style="background: #28A745; height: 3px; width: {precision*100}%; border-radius: 2px; margin-top: 0.5rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric" style="background: linear-gradient(135deg, #FFF3E0 0%, white 100%); border-left: 5px solid #FF9900;">
                <h3 style="color: #FF9900; margin: 0; font-size: 1.1rem;">🔍 Rappel (Recall)</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{recall:.4f}</p>
                <p style="color: #666; margin: 0; font-size: 0.85rem;">📈 Couverture des vraies positives</p>
                <div style="background: #FF9900; height: 3px; width: {recall*100}%; border-radius: 2px; margin-top: 0.5rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric" style="background: linear-gradient(135deg, #F3E5F5 0%, white 100%); border-left: 5px solid #8E44AD;">
                <h3 style="color: #8E44AD; margin: 0; font-size: 1.1rem;">⚖️ F1-Score</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #232F3E;">{f1:.4f}</p>
                <p style="color: #666; margin: 0; font-size: 0.85rem;">🎯 Moyenne harmonique Précision/Rappel</p>
                <div style="background: #8E44AD; height: 3px; width: {f1*100}%; border-radius: 2px; margin-top: 0.5rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        Le graphique radar ci-dessous permet de visualiser la performance globale du modèle sur ces différents aspects. Plus la surface est grande, meilleur est le modèle.
        """)
        metrics_labels = ["RMSE", "MAE", "Précision", "Rappel", "F1-score"]
        # Normalisation pour le radar (RMSE et MAE: plus petit = mieux, on inverse pour le radar)
        max_rmse = 5.0
        max_mae = 5.0
        norm_rmse = 1 - min(rmse / max_rmse, 1)
        norm_mae = 1 - min(mae / max_mae, 1)
        norm_precision = precision
        norm_recall = recall
        norm_f1 = f1
        radar_values = [norm_rmse, norm_mae, norm_precision, norm_recall, norm_f1]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=metrics_labels + [metrics_labels[0]],
            fill='toself',
            name='ALS'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Radar des performances du modèle ALS"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors du calcul des statistiques ou des performances : {str(e)}")

# Nouvelle page : Analyse Avancée---------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == "🧩 Analyse Avancée":
    st.markdown('<div class="st-emojified-title">🧩 Laboratoire d\'Analyse Avancée</div>', unsafe_allow_html=True)
    try:
        st.markdown('<div class="st-emojified-subheader">🔥 Heatmap des Interactions Utilisateur-Produits</div>', unsafe_allow_html=True)
        # Limiter le nombre d'utilisateurs et de produits pour la lisibilité
        max_users = 30
        max_products = 30
        users_top = df_clean.groupBy("user_id").count().sort(desc("count")).limit(max_users).toPandas()["user_id"].tolist()
        products_top = df_clean.groupBy("product_id", "title").count().sort(desc("count")).limit(max_products).toPandas()["product_id"].tolist()
        df_heatmap = df_clean.filter(col("user_id").isin(users_top) & col("product_id").isin(products_top))
        pd_heatmap = df_heatmap.groupBy("user_id", "product_id").count().toPandas().pivot(index="user_id", columns="product_id", values="count").fillna(0)
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.heatmap(pd_heatmap, cmap="YlOrRd", ax=ax)
        ax.set_xlabel("Produits")
        ax.set_ylabel("Utilisateurs")
        st.pyplot(fig)
        st.caption("Zones chaudes = beaucoup d'interactions, zones froides = peu d'interactions.")

        orange_separator()
        st.markdown('<div class="st-emojified-subheader">📈 Matrice de Corrélation Avancée</div>', unsafe_allow_html=True)
        # 1. Corrélation nombre d’avis et note moyenne (par produit)
        prod_stats = df_clean.groupBy("product_id").agg(
            count("rating").alias("nb_avis"),
            avg("rating").alias("note_moyenne")
        ).toPandas()
        corr1 = prod_stats[["nb_avis", "note_moyenne"]].corr().iloc[0,1]
        st.metric("Corrélation nb d'avis / note moyenne (produit)", f"{corr1:.2f}")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=prod_stats, x="nb_avis", y="note_moyenne", ax=ax1)
        ax1.set_title("Nb d'avis vs Note moyenne (produit)")
        st.pyplot(fig1)

        # 2. Corrélation activité utilisateur et satisfaction moyenne
        user_stats = df_clean.groupBy("user_id").agg(
            count("rating").alias("nb_avis"),
            avg("rating").alias("note_moyenne")
        ).toPandas()
        corr2 = user_stats[["nb_avis", "note_moyenne"]].corr().iloc[0,1]
        st.metric("Corrélation activité utilisateur / satisfaction moyenne", f"{corr2:.2f}")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=user_stats, x="nb_avis", y="note_moyenne", ax=ax2)
        ax2.set_title("Activité utilisateur vs Satisfaction moyenne")
        st.pyplot(fig2)

        # 3. Fréquence de recommandation vs popularité réelle
        # Fréquence de recommandation = nombre de fois où le produit est recommandé par ALS
        # Popularité réelle = nombre d'avis
        with st.spinner('Calcul des recommandations ALS pour chaque produit (échantillon)...'):
            model, products_lookup, users_lookup = train_als_model(df_clean)
            # On prend un échantillon d'utilisateurs
            user_indices = users_lookup["user_index"].sample(n=min(30, len(users_lookup)), random_state=42).tolist()
            rec_counts = {}
            for idx in user_indices:
                user_df = spark.createDataFrame([(float(idx),)], ["user_index"])
                try:
                    recs = model.recommendForUserSubset(user_df, 10)
                    if recs.count() > 0:
                        recs_pd = recs.toPandas()
                        for _, row in recs_pd.iterrows():
                            for rec in row['recommendations']:
                                pid = products_lookup[products_lookup['product_index'] == rec['product_index']]['product_id'].values[0]
                                rec_counts[pid] = rec_counts.get(pid, 0) + 1
                except Exception:
                    continue
            rec_df = pd.DataFrame(list(rec_counts.items()), columns=["product_id", "freq_recommandation"])
            popu_df = df_clean.groupBy("product_id").count().toPandas().rename(columns={"count": "popularite_reelle"})
            merge_df = pd.merge(rec_df, popu_df, on="product_id", how="inner")
            corr3 = merge_df[["freq_recommandation", "popularite_reelle"]].corr().iloc[0,1] if not merge_df.empty else 0
            st.metric("Corrélation fréquence recommandation / popularité réelle", f"{corr3:.2f}")
            fig3, ax3 = plt.subplots()
            if not merge_df.empty:
                sns.scatterplot(data=merge_df, x="freq_recommandation", y="popularite_reelle", ax=ax3)
            ax3.set_title("Fréquence de recommandation vs Popularité réelle")
            st.pyplot(fig3)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse avancée : {str(e)}")
# Pied de page Amazon avec style amélioré
st.sidebar.markdown('<hr class="orange-separator">', unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style="
        background: linear-gradient(135deg, #FF9900 0%, #FFB84D 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: small;
        box-shadow: 0 4px 15px rgba(255,153,0,0.3);
        margin-top: 1rem;
    ">
        <p style="margin: 0; font-weight: bold; font-size: 0.9rem;">🚀 Système de Recommandation </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.9;">© 2025 EMSI | Made with ❤️ by JOSKA</p>
    </div>
""", unsafe_allow_html=True)
