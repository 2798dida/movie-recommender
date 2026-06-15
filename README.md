# Sistem Rekomendasi Film Hybrid - MovieLens 32M

Project ini mengikuti rancangan pada dokumen: hybrid collaborative filtering dengan
penanganan cold start. Model dilatih offline, lalu aplikasi Streamlit hanya memuat
artifact model untuk inference.

## 1. Struktur Folder

```text
movie_recommender/
  app.py
  requirements.txt
  data/
    raw/ml-32m/
    processed/
  models/
  src/
    config.py
    prepare_data.py
    train_models.py
    recommender.py
    evaluate.py
```

## 2. Siapkan Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Jika menggunakan macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Download Dataset MovieLens 32M

Download dari GroupLens:

```text
https://files.grouplens.org/datasets/movielens/ml-32m.zip
```

Ekstrak sehingga file berikut berada di:

```text
data/raw/ml-32m/ratings.csv
data/raw/ml-32m/movies.csv
data/raw/ml-32m/tags.csv
data/raw/ml-32m/links.csv
```

## 4. Preprocessing

Untuk test cepat di laptop:

```bash
python -m src.prepare_data --max-ratings 1000000
```

Untuk dataset penuh:

```bash
python -m src.prepare_data
```

Output preprocessing:

```text
data/processed/movies.parquet
data/processed/interactions_positive.parquet
data/processed/user_map.parquet
data/processed/movie_cf_map.parquet
```

## 5. Training Model

Untuk eksperimen ringan:

```bash
python -m src.train_models --factors 50 --iterations 10 --max-features 30000
```

Untuk training lebih serius:

```bash
python -m src.train_models --factors 100 --iterations 20 --regularization 0.05 --alpha 15
```

Artifact model akan disimpan di folder:

```text
models/
```

## 6. Jalankan Streamlit

```bash
streamlit run app.py
```

Mode yang tersedia:

- User existing: rekomendasi hybrid untuk user yang sudah punya histori rating.
- Cold start user baru: rekomendasi dari genre, teks bebas, dan film contoh.
- Cari film mirip: content-based berdasarkan genre dan tag.
- Film populer: fallback jika data user belum tersedia.

## 7. Evaluasi Cepat

```bash
python -m src.evaluate --sample-users 1000 --k 10
```

File evaluasi ini hanya sanity check. Untuk laporan akademik, gunakan split
train-test/holdout yang ketat agar metrik tidak bias.

## 8. Deploy Streamlit

1. Push project ke GitHub.
2. Jangan push dataset mentah `data/raw/`.
3. Jangan push model besar jika melewati limit GitHub/Streamlit.
4. Jika artifact terlalu besar, simpan ZIP artifact di cloud storage.
5. Di Streamlit Community Cloud, pilih repo, branch, dan entrypoint `app.py`.
6. Klik Deploy.

ZIP artifact untuk deploy sebaiknya berisi:

```text
models/als_model.joblib
models/tfidf_vectorizer.joblib
models/user_item.npz
models/movie_tfidf.npz
data/processed/movies.parquet
data/processed/user_map.parquet
data/processed/movie_cf_map.parquet
```

Lalu set secret/environment variable:

```text
MODEL_ARTIFACT_ZIP_URL=https://lokasi-artifact-kamu/artifacts.zip
```

## 9. Alur Hybrid

Untuk user existing:

```text
ALS collaborative filtering
+ content similarity dari film yang pernah disukai
+ popularity fallback
= skor hybrid
```

Untuk cold start:

```text
genre favorit / teks preferensi / film contoh
+ content-based TF-IDF
+ popularity fallback
= rekomendasi awal
```
