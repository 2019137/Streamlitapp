import streamlit as st
import pandas as pd
df = None  # Initialize df variable

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


st.title("Data Loading Application")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    st.write(df)

if "label" not in df.columns:
    st.error("The data must contain a 'label' column")
else:
    st.success("Data loaded successfully")


# Οπτικοποίηση PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.drop(columns=["label"]))
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

st.write("PCA Visualization")
fig, ax = plt.subplots()
sns.scatterplot(x='pca1', y='pca2', hue='label', data=df, ax=ax)
st.pyplot(fig)

# Οπτικοποίηση t-SNE
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(df.drop(columns=["label"]))
df['tsne1'] = tsne_result[:, 0]
df['tsne2'] = tsne_result[:, 1]

st.write("t-SNE Visualization")
fig, ax = plt.subplots()
sns.scatterplot(x='tsne1', y='tsne2', hue='label', data=df, ax=ax)
st.pyplot(fig)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Διαίρεση δεδομένων
X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# k-NN
k = st.slider("Select k for k-NN", 1, 15)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

st.write(f"k-NN accuracy: {knn_accuracy}")

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

st.write(f"Random Forest accuracy: {rf_accuracy}")




tab1, tab2, tab3, tab4 = st.tabs(["2D Visualization", "Classification", "Clustering", "Info"])

with tab1:
    st.header("2D Visualization")
    # Οπτικοποιήσεις και EDA εδώ

with tab2:
    st.header("Classification Algorithms")
    # Αλγόριθμοι κατηγοριοποίησης εδώ

with tab3:
    st.header("Clustering Algorithms")
    # Αλγόριθμοι ομαδοποίησης εδώ

with tab4:
    st.header("Πληροφορίες Σχετικά με την Εφαρμογή")
    st.write("""Αυτή η εφαρμογή έχει σχεδιαστεί για να παρέχει εργαλεία ανάλυσης                    δεδομένων και μηχανικής μάθησης. Οι χρήστες μπορούν να φορτώσουν τα                  δικά τους δεδομένα, να εκτελέσουν οπτικοποιήσεις, να χρησιμοποιήσουν                 αλγορίθμους κατηγοριοποίησης και ομαδοποίησης και να δουν τις                        επιδόσεις των μοντέλων.""")


    st.subheader("Οδηγίες χρήσης")
    st.write("""1. **Φόρτωση Δεδομένων**: Ανεβάστε ένα αρχείο CSV ή Excel με τα                      δεδομένα σας.
                2. **2D Visualization**: Χρησιμοποιήστε τις οπτικοποιήσεις PCA και                   t-SNE για να δείτε την κατανομή των δεδομένων σας.
                3. **Μηχανική Μάθηση**: Επιλέξτε μεταξύ αλγορίθμων κατηγοριοποίησης                  ή ομαδοποίησης και δείτε τα αποτελέσματα και τις επιδόσεις.""")


    st.subheader("Ομάδα Ανάπτυξης")
    st.write("Αυτή η εφαρμογή αναπτύχθηκε από την παρακάτω ομάδα:")

    st.markdown("Μέλος 1: Μιχαήλ Ζαραβέλας (Π2019137)")

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

if uploaded_file:
    with tab3:
        st.subheader("K-Means Clustering")
        k = st.slider("Select the number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        labels_kmeans = kmeans.labels_
        silhouette_kmeans = silhouette_score(X, labels_kmeans)
        st.write(f"Silhouette Score of K-Means: {silhouette_kmeans:.2f}")

        st.subheader("Agglomerative Clustering")
        agglomerative = AgglomerativeClustering(n_clusters=k)
        labels_agglomerative = agglomerative.fit_predict(X)
        silhouette_agglomerative = silhouette_score(X, labels_agglomerative)
        st.write(f"Silhouette Score of Agglomerative Clustering: {silhouette_agglomerative:.2f}")
