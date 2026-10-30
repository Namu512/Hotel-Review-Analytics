# Drivers of Guest Satisfaction in European Hotels
### Sentiment classification (TF-IDF) and topic modelling (LDA) on 515,000 Booking.com reviews

This repository contains the code and the research paper for a study that identifies which aspects of the
hotel experience are most strongly associated with positive and negative guest reviews.

**Paper:** [`paper/main.pdf`](paper/main.pdf) (IEEE format, LaTeX source in `paper/main.tex`)

---

## 1. What this project does

1. **Cleans and reshapes** 515,738 reviews into labelled text segments (positive field vs. negative field).
2. **Explores** the data (score distributions, review length, countries, word clouds).
3. **Classifies sentiment** with TF-IDF features and three models: Naive Bayes, Logistic Regression, Linear SVM.
4. **Models topics** with LDA, separately for positive and negative text, choosing the number of topics by coherence.
5. **Combines** sentiment and topics: which themes are over-represented in complaints (hygiene factors) and in praise (satisfaction drivers).

## 2. Key results

| Model | Accuracy | Macro-F1 | Fit time (s) |
|---|---|---|---|
| Multinomial Naive Bayes | 0.9194 | 0.9183 | 0.26 |
| **Logistic Regression** | **0.9286** | **0.9286** | 0.80 |
| Linear SVM | 0.9261 | 0.9254 | 1.64 |

*(held-out test split, n = 30,000 segments from a 150,000-segment stratified sample)*

* LDA topics selected by c_v coherence: **k = 13** (positive), **k = 9** (negative).
* Over-represented in complaints: **Room** (30.2% of negative texts vs 10.6% of positive), **Breakfast & Food** (18.8% vs 5.4%), **Noise** (11.9%), **Check-in & Payments** (10.5%).
* Over-represented in praise: **Location** (29.8% of positive texts vs 7.4% of negative). **Staff & Service** is prominent in both (25.1% vs 14.6%).

## 3. Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                 # how to obtain the dataset (not stored in the repo)
├── notebooks/
│   ├── 01_cleaning_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_topic_modelling.ipynb
│   └── 05_combined_analysis.ipynb
├── figures/                      # charts produced by the notebooks
└── paper/
    ├── main.tex                  # IEEE-format paper (LaTeX)
    ├── main.pdf
    ├── make_figures.py
    └── figures/
```

## 4. Setup

**Requirements:** Python 3.10 or newer, about 8 GB RAM (16 GB is more comfortable for LDA on the full corpus).

```bash
# 1. clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. virtual environment (Windows)
python -m venv venv
venv\Scripts\activate
#    (macOS/Linux: source venv/bin/activate)

# 3. install dependencies
pip install -r requirements.txt

# 4. download the NLTK resources used in notebook 01
python -m nltk.downloader stopwords wordnet omw-1.4
```

## 5. Data

The dataset is **not** included (it is about 230 MB, above GitHub's 100 MB file limit, and is a third-party dataset).

1. Download **"515K Hotel Reviews Data in Europe"** from Kaggle: <https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe>
2. Save the file as `data/Hotel_Reviews.csv`.

Expected columns include `Hotel_Name`, `Hotel_Address`, `Review_Date`, `Reviewer_Nationality`, `Reviewer_Score`,
`Positive_Review`, `Negative_Review`, `lat`, `lng`.

## 6. How to run

Run the notebooks **in order**. Each notebook saves a file that the next one loads, so you never repeat slow steps.

| # | Notebook | Reads | Writes | Approx. time |
|---|---|---|---|---|
| 01 | Cleaning and preprocessing | `data/Hotel_Reviews.csv` | `hotel_reviews_cleaned.parquet/.csv` | a few minutes (lemmatisation) |
| 02 | Exploratory analysis | cleaned file | `eda_*.png` | 1-3 minutes |
| 03 | Sentiment classification | cleaned file | `final_sentiment_model.pkl`, `tfidf_vectorizer.pkl` | minutes |
| 04 | Topic modelling (LDA) | sentiment file | `lda_*.model`, `hotel_reviews_with_topics.*` | tens of minutes |
| 05 | Combined analysis | topics file | theme tables, charts, dashboard export | 1-2 minutes |

To try the pipeline quickly, take a sample at the start of notebook 01:

```python
df = pd.read_csv("data/Hotel_Reviews.csv").sample(20000, random_state=42)
```

## 7. Key analytical decisions and why

| Decision | Rationale |
|---|---|
| Use the positive and negative review **fields** as labels | Manual annotation of hundreds of thousands of texts is not feasible. Trade-off: the labels reflect the form field, not independently judged sentiment (see Limitations). |
| Remove the "No Positive" / "No Negative" placeholders | They are not text. Left in, they would dominate the vocabulary and the topic models. |
| TF-IDF with unigrams and bigrams, 10,000 features | Bigrams capture short phrases such as "very clean"; a feature cap keeps the sparse matrix fast. |
| Fit TF-IDF on the training split only | Prevents information leaking from the test set. |
| Macro-F1 next to accuracy, balanced class weights | The classes are 55.5% / 44.5%. |
| Develop on a 150,000-segment sample, retrain on all data | Makes iteration fast. |
| **Separate LDA models** for positive and negative text | Guests praise and complain about different things. One model would blend them. |
| Choose k with c_v coherence | A coherence-based choice avoids arbitrary k. A test run at k=10 produced two near-duplicate location topics. |
| Extra domain stop-words for LDA ("hotel", "room", "stay", ...) | These words appear in nearly every review and would dominate all topics. |
| Map topics of both models to shared **themes** | The two LDA models have different topics. Themes make them comparable. |
| Exclude the "No complaint" filler topic | Its words ("nothing", "everything", "great") come from guests writing "nothing to complain about". |

## 8. Limitations

* Labels come from the review form, so the classifier's accuracy overstates its accuracy on free text that mixes opinions.
* The default NLTK stop-word list removes negations ("not", "no"), which can hide the meaning of phrases like "not clean".
* The comparison uses a single train/test split and default regularisation.
* Each text receives one dominant topic; one-sided themes (e.g. Noise) partly reflect the use of two topic models.
* Data cover six countries and 2015-2017 only.

## 9. Possible extensions

Cross-validation, transformer models (BERT), aspect-based sentiment, Spark MLlib implementation, an interactive dashboard (Power BI or Streamlit).

## 10. Data and ethics

The data are public reviews from a third-party Kaggle dataset. Results are aggregated and no individual reviewer is identified. Check the dataset's licence and the source platform's terms before reusing the data.

## 11. Author

**[Your Name]** - [Master's programme, University] - [LinkedIn URL] - [email]

## 12. Citation

If you use this work, please cite the accompanying paper: *Drivers of Guest Satisfaction in European Hotels: Combining TF-IDF Sentiment Classification and LDA Topic Modelling on 515,000 Online Reviews*, [Your Name], 2026.
