import os

import pandas as pd
import streamlit as st

from src.artifacts import download_and_extract_artifacts
from src.recommender import HybridMovieRecommender


GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@st.cache_resource
def load_recommender() -> HybridMovieRecommender:
    return HybridMovieRecommender()


def show_results(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("Belum ada rekomendasi yang cocok.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("score", format="%.4f"),
            "mean_rating": st.column_config.NumberColumn("mean rating", format="%.2f"),
            "rating_count": st.column_config.NumberColumn("rating count"),
        },
    )


st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("Sistem Rekomendasi Film Hybrid")
st.caption("Collaborative Filtering + Matrix Factorization + Content-Based + Cold Start")

if not HybridMovieRecommender.artifacts_exist():
    artifact_url = os.getenv("MODEL_ARTIFACT_ZIP_URL", "")
    if not artifact_url:
        try:
            artifact_url = st.secrets.get("MODEL_ARTIFACT_ZIP_URL", "")
        except Exception:
            artifact_url = ""

    if artifact_url:
        with st.spinner("Mengunduh artifact model..."):
            download_and_extract_artifacts(artifact_url)

if not HybridMovieRecommender.artifacts_exist():
    st.error("Artifact model belum ditemukan.")
    st.code(
        "python -m src.prepare_data\n"
        "python -m src.train_models\n"
        "streamlit run app.py",
        language="bash",
    )
    st.stop()

recommender = load_recommender()

mode = st.sidebar.radio(
    "Mode rekomendasi",
    ["User existing", "Cold start user baru", "Cari film mirip", "Film populer"],
)
n = st.sidebar.slider("Jumlah rekomendasi", min_value=5, max_value=30, value=10, step=5)

if mode == "User existing":
    st.subheader("Rekomendasi untuk user yang sudah punya histori rating")
    default_user = int(recommender.user_map["userId"].iloc[0])
    user_id = st.number_input("Masukkan userId MovieLens", min_value=1, value=default_user)

    if st.button("Buat rekomendasi", type="primary"):
        results = recommender.recommend_for_user_id(int(user_id), n=n)
        show_results(results)

elif mode == "Cold start user baru":
    st.subheader("Rekomendasi awal untuk user baru")
    selected_genres = st.multiselect("Pilih genre favorit", GENRES)
    free_text = st.text_input("Tambahkan preferensi bebas", placeholder="contoh: space adventure, detective, romance")

    search_query = st.text_input("Cari film contoh yang kamu sukai")
    liked_movie_ids = []
    if search_query:
        matches = recommender.search_movies(search_query, limit=25)
        options = matches["movieId"].astype(int).tolist()
        labels = {
            int(row["movieId"]): f"{row['title']} - {row['genres']}"
            for _, row in matches.iterrows()
        }
        liked_movie_ids = st.multiselect(
            "Pilih film contoh",
            options=options,
            format_func=lambda movie_id: labels.get(int(movie_id), str(movie_id)),
        )

    if st.button("Buat rekomendasi cold start", type="primary"):
        results = recommender.recommend_cold_start(
            selected_genres=selected_genres,
            liked_movie_ids=[int(movie_id) for movie_id in liked_movie_ids],
            free_text=free_text,
            n=n,
        )
        show_results(results)

elif mode == "Cari film mirip":
    st.subheader("Cari film yang mirip berdasarkan metadata konten")
    search_query = st.text_input("Judul film")

    if search_query:
        matches = recommender.search_movies(search_query, limit=25)
        if matches.empty:
            st.info("Film tidak ditemukan.")
        else:
            options = matches["movieId"].astype(int).tolist()
            labels = {
                int(row["movieId"]): f"{row['title']} - {row['genres']}"
                for _, row in matches.iterrows()
            }
            selected_movie_id = st.selectbox(
                "Pilih film",
                options=options,
                format_func=lambda movie_id: labels.get(int(movie_id), str(movie_id)),
            )

            if st.button("Cari yang mirip", type="primary"):
                results = recommender.similar_movies(int(selected_movie_id), n=n)
                show_results(results)

else:
    st.subheader("Film populer sebagai fallback")
    show_results(recommender.popular_movies(n=n))
