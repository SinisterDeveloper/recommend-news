# Problem Definition and Objective

## Selected Project Track: Content Recommendation System

## Clear Problem Statement

Develop a Machine Learning solution to provide news article suggestions to the end-user given the article being currently read

## Real World Relevance

News Recommendation Systems are extremely useful for keeping the users hooked and also obtain relevant news which teh users are actually interested in. They help increase the time spent on the website/app and also make it convenient for users to obtain relevant information


```py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

# Data Understanding & Preparation

## Dataset source

**Kaggle: [“News Category Dataset”](https://www.kaggle.com/datasets/rmisra/news-category-dataset)** (Public)

## Data Loading and Exploration

- The dataset was downloaded from Kaggle in the `json` format
- It contains around 210k news headlines from 2012 to 2022 from HuffPost
- This is one of the biggest news datasets and can serve as a benchmark for a variety of computational linguistic tasks.

- Instead of loading the dataset dynamically, I downloaded the `json` data and uploaded it to the project source
- I open the dataset, go through the data and convert them into a Pandas DataFrame object

## Cleaning, Preprocessing, Feature Engineering

- Each record in the dataset consists of the following attributes: `category`, `headline`, `authors`, `link`, `short_description`, `date`
- I onlky required the `headline` and `short_description` fields for our recommendation system so I ignore the rest
- I handled some `JSONDecodeError`s using the `except` block and ignored that specific record

```py
import json

data = []
with open("dataset.json", "r") as f:
    for i, line in enumerate(f):
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line {i+1}: {e}")
df = pd.DataFrame(data)
```

## Handling Missing Data

Records containing missing values were dropped using Pandas' `dropna()` function

```py
df = df[['headline', 'short_description']].dropna().reset_index(drop=True)

df['text'] = df['headline'] + " " + df['short_description']
```

# Model / System Design

## AI Technique

- The AI technique being used is **Term Frequency Inverse Document Frequency (TF-IDF)**.
- The textual data from news headlines and descriptions are transformed into numerical representations
- The similarity between the news articles is determined using the cosine similarity in the **TF-IDF Vector Space**
- The articles with the highest cosine similarity is considered as most relevant, and hence are recommended

## Architecture

### Data Layer

- News articles are loaded from JSON dataset
- Only headline and short description fields are extracted & used.

### Feature Extraction Layer

- Text is preprocessed and vectorized using TF-IDF.
- The resulting sparse TF-IDF matrix is computed

### Similarity Determination Layer

- Cosine similarity is computed between the query article vector and the full TF-IDF matrix
- Similarity is calculated on demand to avoid quadratic memory usage

### Recommendation Interface

- The `recommend()` function accepts an article headline and returns the top K most similar articles.

## Justification

- The dataset chosen did not include additional data such as user interactionsm, thereby making collborative filtering not feasible
- TF-IDF technique is relatively lightweight and suitable for large text corpuses without requiring large training times or massive GPU compute
- Recommendations can be deteriministially explained in terms of shared keywords and document similarity, unlike many popular deep learning techniques which act like a black box
- This system provides a solid baseline which can be extended to include semantic embeddings, personalisation and hybrid recommendation methods

```py
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(df['text'])
```

```py
def recommend(headline, top_k=5):
    if headline not in df['headline'].values:
        return "Headline not found."

    idx = df[df['headline'] == headline].index[0]

    query_vector = tfidf_matrix[idx]
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[-top_k-1:-1][::-1]

    return df.iloc[top_indices]['headline'].tolist()

```

# Evaluation & Analysis

## Evaluation Type

A qualitative evaluation with a lightweight quantitative signal

## Justification

Since the system is content-based and unsupervised, traditional accuracy metrics like precision or RMSE are not applicable.

Therefore, evaluation focuses on system-level performance, specifically recommendation latency, which is critical for user-facing applications.

```py
import time
import numpy as np

def evaluate_recommender_latency(sample_size=50, top_k=5):
    """
    Quantitative evaluation focused on system performance.
    Measures recommendation latency for real-time usability.
    """

    sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    latencies = []

    for _, row in sample_df.iterrows():
        headline = row['headline']

        start = time.time()
        recommend(headline, top_k=top_k)
        latencies.append(time.time() - start)

    latencies = np.array(latencies)

    print("Evaluation Summary")
    print("-" * 30)
    print(f"Average Recommendation Latency: {latencies.mean()*1000:.2f} ms")
    print(f"Max Recommendation Latency: {latencies.max()*1000:.2f} ms")
    print(f"Min Recommendation Latency: {latencies.min()*1000:.2f} ms")

    print("\nInterpretation:")
    print("- Latency measures real-time suitability of the recommender.")
    print("- On-demand similarity computation avoids quadratic memory usage.")
    print("- Performance scales linearly with the number of articles.")

# Run evaluation
evaluate_recommender_latency(sample_size=50, top_k=5)
```

```py
example = df.iloc[0]['headline']
print("Input:")
print(example)
print("\nRecommendations:")
for r in recommend(example):
    print("-", r)
```

# Ethical Considerations & Responsible AI

## Bias and Representation

- The recommendation is trained on historical news data
- Hence, it may reflect ethical biases present in the media coverage
- Since recommendations are base don textual similarity, any biases present in dataset may reflect in the recommendations

## Transparency

- The AI system is deterministally interpretable
- The recommendations are made based on shared keywords and the terms' importance derived from the TF-IDF weights
- This allows it to be understood why certain arrticles are recommended
- No black box models are used

## Misinformation and Harmful Content

- The AI system does not evaluate the factual correctness of the articles it recommends
- It may recommend articles that are outdated or misleading, if such articles are present in the dataset
- Responsiblity for the content moderation lies outside the content recommender within the data accuration layer

# Conclusion

This project successfully demonstrates a content-based news recommendation system using classical Natural Language Processing techniques. News articles are represented using TF-IDF vectors derived from headlines and short descriptions, and recommendations are generated using cosine similarity.

## Possible Improvements

- **Semantic Representations**: Replacing TF-IDF with semantic text embeddings to capture deeper meaning beyond textual similarity
- **Diversity-Aware Recommendation**: Introducing mechanisms to promote topic diversity and reduce redundancy in recommendations
- **Personalization**: Using user interaction data to build user profiles & move towards more personalised recommendations
