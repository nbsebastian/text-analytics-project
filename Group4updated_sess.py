import string
import numpy as np

# --- SciPy fallback shim (fix: no _sla; use la consistently) ---
try:
    import scipy.linalg as la
    if not hasattr(la, "triu"):
        la.triu = np.triu
except Exception:
    pass

import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from gensim.models import FastText
from wordcloud import WordCloud
from nltk.corpus import stopwords
from io import StringIO
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, hamming_loss
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Toxic Comment Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Session state initialization
# ---------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}
if "test_cache" not in st.session_state:
    st.session_state.test_cache = {}

# ---------------------------
# Data utilities
# ---------------------------
from nltk.corpus import stopwords
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# FIX: correct variable name and path
DATA_PATH = Path("/Users/borteley/Downloads/text_anal_project/final project/train.csv")

stop_words = set(stopwords.words('english'))
stop_words.update({'article', 'wikipedia', 'page', 'edit', 'talk', 'user', 'please', 'thanks', 'thank'})

@st.cache_data
def load_data_cached(path_str: str):
    return pd.read_csv(path_str)

# FIX: check that DATA_PATH exists before loading
def load_data():
    if st.session_state.dataset is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"File not found: {DATA_PATH}")
        df = load_data_cached(str(DATA_PATH))
        st.session_state.dataset = df
    return st.session_state.dataset

def clean_text_simple(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def plot_label_distribution(df, label_cols):
    label_sums = df[label_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=label_sums.values, y=label_sums.index, palette="viridis", ax=ax)
    st.pyplot(fig)

def plot_label_correlation(df, label_cols):
    corr = df[label_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

def plot_wordcloud(words_freq, title):
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100)
    wc.generate_from_frequencies(dict(words_freq))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    st.pyplot(fig)

def text_summary(df, text_col):
    texts = df[text_col].dropna().astype(str)
    unique_count = texts.nunique()
    lengths = texts.str.len()
    all_words = ' '.join(texts.apply(clean_text_simple))
    words = [w for w in re.findall(r'\b[a-z]+\b', all_words) if w not in stop_words and len(w) > 2]
    common_words = Counter(words).most_common(50)

    st.write(f"### Summary for `{text_col}`")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- Total comments: {len(texts)}")
        st.write(f"- Unique comments: {unique_count}")
        st.write(f"- Avg length: {lengths.mean():.1f} chars")
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(lengths, bins=50, kde=True, ax=ax)
        st.pyplot(fig)

    plot_wordcloud(common_words, "Most Frequent Words in All Comments")

def show_label_overlap(df):
    df = df.copy()
    df['num_labels'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)
    multi_label_count = (df['num_labels'] > 1).sum()
    st.write(f"- Comments with multiple labels: {multi_label_count}")

# ---------------------------
# Homepage
# ---------------------------
def homepage():
    st.title("Toxic Comment Detection System")
    st.markdown("""
        <div style="color: #666; font-size: 1.1rem;">
        Identify and classify toxic online content using machine learning
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        ### 👥 Group 4 Members
        <ul>
            <li><strong>Priscilla D. Gborbitey</strong> — 22253220</li>
            <li><strong>Philip Kwasi Adjartey</strong> — 22252449</li>
            <li><strong>Nicco-Annan Marilyn</strong> — 11410745</li>
            <li><strong>Naa Borteley Sebastian-Kumah</strong> — 22253153</li>
            <li><strong>Bernice Baadawo Abbe</strong> — 22253447</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚀 About This Project")
    st.write("""
    Detect and classify toxic comments across categories: Toxic, Severe toxic, Obscene, Threat, Insult, and Identity hate.
    Trained on the Jigsaw dataset.
    """)

    st.subheader("✨ Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Multi-Label Classification\nDetects multiple toxicity types.")
    with col2:
        st.markdown("### 🤖 Dual Model Approach\nCompare Logistic Regression & Random Forest.")
    with col3:
        st.markdown("### 📈 Comprehensive Metrics\nPrecision, Recall, F1-Score, ROC-AUC.")

    st.markdown("---")
    st.subheader("🧭 Navigation Guide")
    st.write("Use the sidebar to explore: Dataset Preview, Modeling & Evaluation, Prediction.")

# ---------------------------
# Page: Dataset Preview
# ---------------------------
def dataset_preview():
    try:
        df = load_data()
        st.write(f"### Dataset Overview: {df.shape[0]:,} comments, {df.shape[1]} columns")
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Text Analysis", "Label Analysis"])
        with tab1:
            st.dataframe(df.head(10))
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        with tab2:
            text_col = 'comment_text' if 'comment_text' in df.columns else df.columns[1]
            text_summary(df, text_col)
        with tab3:
            label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            plot_label_distribution(df, label_cols)
            plot_label_correlation(df, label_cols)
            show_label_overlap(df)
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ---------------------------
# Page: Data Preprocessing
# ---------------------------
def page2():
    st.title("🧼 Data Preprocessing")
    st.caption("This page prepares the data for analysis and modeling.")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please make sure `train.csv` is at DATA_PATH.")
        return

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        sw = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in sw and word.isalpha()]
        stemmer = PorterStemmer()
        cleaned_tokens = []
        for word in tokens:
            try:
                cleaned_tokens.append(stemmer.stem(word))
            except RecursionError:
                continue
        return " ".join(cleaned_tokens)

    if st.session_state.processed_data is None:
        proc = df.copy()
        proc["cleaned_text"] = proc["comment_text"].apply(clean_text)
        proc = proc[proc["cleaned_text"].str.strip() != ""]
        proc = proc.drop_duplicates(subset="cleaned_text")
        proc["comment_length"] = proc["cleaned_text"].apply(len)
        proc["num_labels"] = proc.iloc[:, 2:8].sum(axis=1)
        st.session_state.processed_data = proc
        proc.to_csv("cleaned_train.csv", index=False)

    st.subheader("📊 Preview of Cleaned Data")
    st.dataframe(st.session_state.processed_data[["comment_text", "cleaned_text", "comment_length", "num_labels"]])
    st.success("✅ Cleaned dataset saved as `cleaned_train.csv` and stored in session.")

# ---------------------------
# Page: Modeling & Evaluation
# ---------------------------
def page3():
    st.title("📊 Modeling & Evaluation")
    st.markdown("""
    This page trains and evaluates two machine learning models — Logistic Regression and Random Forest —
    using FastText embeddings trained on your cleaned dataset.
    """)

    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        st.success("✅ Cleaned dataset loaded from session")
    else:
        try:
            df = pd.read_csv("cleaned_train.csv")
            st.success("✅ Preloaded cleaned dataset loaded successfully (disk)")
        except FileNotFoundError:
            st.error("❌ Cleaned dataset not found. Please run 'Data Preprocessing' first.")
            return

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    COMMENT_COL = "cleaned_text"
    LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    if "ft" not in st.session_state.models:
        st.subheader("🔧 Training FastText Embedding Model")
        sentences = df[COMMENT_COL].astype(str).apply(lambda x: x.split()).tolist()
        ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)
        st.session_state.models["ft"] = ft_model
        st.success("✅ FastText model trained")
    else:
        ft_model = st.session_state.models["ft"]
        st.info("ℹ️ Using FastText model from session cache")

    def embed_comment(comment):
        tokens = str(comment).split()
        vectors = [ft_model.wv[word] for word in tokens if word in ft_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(ft_model.vector_size)

    X = np.vstack(df[COMMENT_COL].apply(embed_comment).values)
    y = df[LABEL_COLS]

    if "X_test" not in st.session_state.test_cache or "y_test" not in st.session_state.test_cache:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        st.session_state.test_cache["X_test"] = X_test
        st.session_state.test_cache["y_test"] = y_test
        st.session_state.test_cache["X_train"] = X_train
        st.session_state.test_cache["y_train"] = y_train
    else:
        X_train = st.session_state.test_cache["X_train"]
        y_train = st.session_state.test_cache["y_train"]
        X_test  = st.session_state.test_cache["X_test"]
        y_test  = st.session_state.test_cache["y_test"]

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    f1_macro_scorer = make_scorer(f1_score, average="macro")

    if "lr" not in st.session_state.models:
        st.subheader("🛠️ Tuning Logistic Regression (One-vs-Rest)")
        base_lr = LogisticRegression(max_iter=1000)
        ovr_lr = OneVsRestClassifier(base_lr, n_jobs=-1)

        lr_param_grid = [
            {
                "estimator__solver": ["lbfgs"],
                "estimator__penalty": ["l2"],
                "estimator__C": [0.1, 1, 3, 10],
                "estimator__class_weight": [None, "balanced"],
            },
            {
                "estimator__solver": ["liblinear"],
                "estimator__penalty": ["l1", "l2"],
                "estimator__C": [0.1, 1, 3, 10],
                "estimator__class_weight": [None, "balanced"],
            },
        ]

        lr_grid = GridSearchCV(
            estimator=ovr_lr,
            param_grid=lr_param_grid,
            scoring=f1_macro_scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        lr_grid.fit(X_train, y_train)
        lr_model = lr_grid.best_estimator_
        st.session_state.models["lr"] = lr_model

        st.info(f"Best LR params: {lr_grid.best_params_}")
    else:
        lr_model = st.session_state.models["lr"]
        st.info("ℹ️ Using tuned Logistic Regression from session cache")

    if "rf" not in st.session_state.models:
        st.subheader("🛠️ Tuning Random Forest")
        rf_base = RandomForestClassifier(random_state=42)

        rf_param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        rf_grid = GridSearchCV(
            estimator=rf_base,
            param_grid=rf_param_grid,
            scoring=f1_macro_scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        rf_grid.fit(X_train, y_train)
        rf_model = rf_grid.best_estimator_
        st.session_state.models["rf"] = rf_model

        st.info(f"Best RF params: {rf_grid.best_params_}")
    else:
        rf_model = st.session_state.models["rf"]
        st.info("ℹ️ Using tuned Random Forest from session cache")

    st.success("✅ Models tuned and trained successfully")

    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    lr_report = classification_report(y_test, lr_preds, output_dict=True, zero_division=0)
    rf_report = classification_report(y_test, rf_preds, output_dict=True, zero_division=0)

    lr_auc = roc_auc_score(y_test, lr_preds, average="macro")
    rf_auc = roc_auc_score(y_test, rf_preds, average="macro")

    st.session_state.model_metrics = {
        "lr_report": lr_report,
        "rf_report": rf_report,
        "lr_auc": lr_auc,
        "rf_auc": rf_auc,
    }

    joblib.dump(lr_model, "logistic_model.pkl")
    joblib.dump(rf_model, "random_forest_model.pkl")
    ft_model.save("fasttext_model.bin")
    joblib.dump(st.session_state.test_cache["X_test"], "X_test.pkl")
    joblib.dump(st.session_state.test_cache["y_test"], "y_test.pkl")

    st.subheader("📈 Evaluation Results")
    st.markdown("### Logistic Regression")
    st.write(pd.DataFrame(lr_report).transpose())
    st.metric("ROC-AUC", f"{lr_auc:.2f}")

    st.markdown("### Random Forest")
    st.write(pd.DataFrame(rf_report).transpose())
    st.metric("ROC-AUC", f"{rf_auc:.2f}")

    st.subheader("📊 F1-Score Comparison")
    f1_scores = {
        "Logistic Regression": lr_report["weighted avg"]["f1-score"],
        "Random Forest": rf_report["weighted avg"]["f1-score"],
    }
    st.bar_chart(pd.Series(f1_scores))

    st.success("✅ Models and test data saved and cached in session for Prediction interface")

# ---------------------------
# Page: Prediction
# ---------------------------
def page4():
    st.title("🔍 Toxic Comment Prediction")
    st.subheader("Prediction")
    st.write("Enter your text to predict toxicity here.")

    if "ft" in st.session_state.models and "lr" in st.session_state.models and "rf" in st.session_state.models:
        ft_model = st.session_state.models["ft"]
        lr_model = st.session_state.models["lr"]
        rf_model = st.session_state.models["rf"]
        st.info("ℹ️ Using models from session cache")
    else:
        ft_model = FastText.load("fasttext_model.bin")
        lr_model = joblib.load("logistic_model.pkl")
        rf_model = joblib.load("random_forest_model.pkl")
        st.info("ℹ️ Loaded models from disk")

    LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    st.sidebar.header("⚙️ Options")
    model_choice = st.sidebar.radio("Choose a model:", ["Logistic Regression", "Random Forest"])

    def embed_comment(comment):
        tokens = str(comment).split()
        vectors = [ft_model.wv[word] for word in tokens if word in ft_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(ft_model.vector_size)

    st.markdown("### 💬 Enter a comment to analyze:")
    user_input = st.text_area("", height=100)

    if st.button("🔍 Predict"):
        if user_input.strip():
            embedded = embed_comment(user_input).reshape(1, -1)
            if np.all(embedded == 0):
                st.error("⚠️ No recognizable words found in FastText vocabulary.")
            else:
                model = lr_model if model_choice == "Logistic Regression" else rf_model
                prediction = model.predict(embedded)[0]
                try:
                    probabilities = model.predict_proba(embedded)
                except Exception:
                    probabilities = None

                st.subheader("🧾 Prediction Results")
                for i, label in enumerate(LABEL_COLS):
                    result = prediction[i] if isinstance(prediction, (list, np.ndarray)) else prediction
                    if probabilities is not None and hasattr(model, "predict_proba"):
                        try:
                            if isinstance(probabilities, list):
                                prob_value = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else probabilities[i][0][0]
                            else:
                                prob_value = probabilities[0][i] if probabilities.ndim > 1 else probabilities[i]
                            st.write(f"**{label}**: {result} _(Confidence: {float(prob_value):.2f})_")
                        except Exception:
                            st.write(f"**{label}**: {result}")
                    else:
                        st.write(f"**{label}**: {result}")
        else:
            st.warning("Please enter a comment before predicting.")

    with st.expander("📊 Compare Model Performance"):
        if "X_test" in st.session_state.test_cache and "y_test" in st.session_state.test_cache:
            X_test = st.session_state.test_cache["X_test"]
            y_test = st.session_state.test_cache["y_test"]
        else:
            X_test = joblib.load("X_test.pkl")
            y_test = joblib.load("y_test.pkl")

        y_pred_lr = lr_model.predict(X_test)
        y_pred_rf = rf_model.predict(X_test)

        f1_lr = f1_score(y_test, y_pred_lr, average=None)
        f1_rf = f1_score(y_test, y_pred_rf, average=None)

        x = np.arange(len(LABEL_COLS))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width/2, f1_lr, width, label='Logistic Regression')
        ax.bar(x + width/2, f1_rf, width, label='Random Forest')
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_COLS, rotation=45)
        ax.set_ylabel('F1 Score')
        ax.set_title('Model Comparison by Label')
        ax.legend()
        st.pyplot(fig)

        summary = pd.DataFrame({
            "Metric": ["Macro F1", "Micro F1", "Weighted F1", "Hamming Loss"],
            "Logistic Regression": [
                f1_score(y_test, y_pred_lr, average='macro'),
                f1_score(y_test, y_pred_lr, average='micro'),
                f1_score(y_test, y_pred_lr, average='weighted'),
                hamming_loss(y_test, y_pred_lr)
            ],
            "Random Forest": [
                f1_score(y_test, y_pred_rf, average='macro'),
                f1_score(y_test, y_pred_rf, average='micro'),
                f1_score(y_test, y_pred_rf, average='weighted'),
                hamming_loss(y_test, y_pred_rf)
            ]
        })
        st.dataframe(summary.style.format("{:.3f}", subset=pd.IndexSlice[:, summary.select_dtypes(include='number').columns]))

    with st.expander("ℹ️ Model Details"):
        st.markdown(f"""
        - Selected Model: {model_choice}
        - Embedding Method: FastText (average of word vectors)
        - Labels Predicted: {", ".join(LABEL_COLS)}
        """)

    with st.expander("📘 Label Definitions"):
        st.markdown("""
        - toxic: Rude, disrespectful, or hateful language.
        - severe_toxic: Intensely aggressive or harmful language.
        - obscene: Offensive or vulgar expressions.
        - threat: Implies harm or danger to others.
        - insult: Directly attacks or belittles someone.
        - identity_hate: Targets someone based on identity (e.g. race, religion).
        """)

# ---------------------------
# Sidebar Navigation & Routing
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset Preview", "Data Preprocessing", "Modeling and Evaluation", "Prediction"]
)

if page == "Home":
    homepage()
elif page == "Dataset Preview":
    dataset_preview()
elif page == "Data Preprocessing":
    page2()
elif page == "Modeling and Evaluation":
    page3()
elif page == "Prediction":
    page4()
