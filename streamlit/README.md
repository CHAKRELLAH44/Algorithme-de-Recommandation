# 🚀 Système de Recommandation IA - Interface Streamlit

## 🎨 Interface Moderne et Attrayante

Cette application Streamlit présente un système de recommandation basé sur l'intelligence artificielle avec une interface utilisateur moderne et intuitive.

### ✨ Fonctionnalités Principales

- **🏠 Tableau de Bord** : Vue d'ensemble avec recherche par utilisateur ou produit
- **📊 Statistiques** : Analyses détaillées et visualisations interactives
- **⚙️ Gestion Algorithme** : Performances et métriques du modèle IA
- **🧩 Analyse Avancée** : Heatmaps et corrélations avancées

### 🎯 Améliorations Visuelles

#### Design System
- **Police moderne** : Google Fonts (Inter)
- **Couleurs cohérentes** : Palette Amazon avec variables CSS
- **Animations fluides** : Transitions et effets de survol
- **Responsive design** : Adaptation mobile et desktop

#### Composants Interactifs
- **Cartes métriques** : Affichage moderne des statistiques
- **Boutons animés** : Effets de brillance et transformations 3D
- **Bannières d'information** : Messages stylisés avec icônes
- **Spinner de chargement** : Indicateurs visuels personnalisés

#### Layout Amélioré
- **Bannière d'accueil** : En-tête avec dégradé et effets visuels
- **Sidebar redesignée** : Navigation moderne avec statut système
- **Footer stylisé** : Informations techniques et version
- **Séparateurs animés** : Lignes décoratives avec effets

### 🛠️ Installation et Lancement

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

### 📁 Structure des Fichiers

```
streamlit/
├── app.py                 # Application principale
├── .streamlit/
│   └── config.toml       # Configuration Streamlit
├── requirements.txt      # Dépendances Python
└── README.md            # Documentation
```

### 🎨 Personnalisation

L'interface utilise un système de variables CSS pour faciliter la personnalisation :

```css
:root {
    --amazon-orange: #FF9900;
    --amazon-dark: #131921;
    --amazon-light: #232F3E;
    --amazon-gray: #EAEDED;
}
```

### 🚀 Fonctionnalités Techniques

- **Cache intelligent** : Optimisation des performances avec `@st.cache_resource`
- **Gestion d'erreurs** : Messages d'erreur informatifs et récupération gracieuse
- **Responsive design** : Interface adaptative pour tous les écrans
- **Accessibilité** : Contrastes et navigation optimisés

### 📊 Métriques et Analytics

L'application affiche des métriques en temps réel :
- Performances du modèle (RMSE, MAE, F1-score)
- Statistiques utilisateurs et produits
- Temps de réponse du système
- Indicateurs de santé de l'application

---

**Développé avec ❤️ par JOSKA | © 2025 EMSI**
