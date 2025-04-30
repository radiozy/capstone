# NYT China Coverage Analysis

## Project Overview

This project analyzes New York Times (NYT) articles about China across different U.S. presidential eras (1981–2020) using topic modeling and sentiment analysis. The goal is to uncover thematic trends and sentiment shifts in media coverage over time, with a focus on identifying patterns and potential correlations with historical events.

The analysis is performed using two main scripts:

- **bertopic_analysis.py**: BERTopic modeling creates essential themes from articles through its analysis of each era while identifying historical topic changes between periods and resharing content according to topics.
- **sentiment_analysis.py**: The system analyzes article sentiment through DistilBERT which is a transformer-based model while it investigates how sentiment patterns change between time periods and matches these changes to historical events whenever topic data exists.

## Repository Structure

- **bertopic_analysis.py**: Script for topic modeling and cross-era topic analysis.
- **sentiment_analysis.py**: Sentiment analysis script contains both per-era and cross-era operations.
- **data/**: (Expected) Directory containing subfolders for each presidential era (e.g., `reagan_81-88`, `hwbush_89-92`, etc.), with `.txt` files of NYT articles.
- **output/**:
  - **bertopic_results/**: Stores topic modeling results, including topic assignments, legends, and visualizations.
  - **sentiment_results/**: Navigio stores the product of topic modeling that includes topic assignments with legends together with visualizations.

## Prerequisites

To run the scripts, you need Python 3.10+ and the following dependencies:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `bertopic`
- `scikit-learn`
- `tqdm`
- `nltk`
- `transformers`
- `torch`

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn bertopic scikit-learn tqdm nltk transformers torch
```

Additionally, download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Data Requirements

- **Input Data**: Place `.txt` files of NYT articles in subfolders under the `BASE_DIR` directory, named according to the `ERAS` dictionary (e.g., `reagan_81-88`, `clinton_93-00`).

- **Directory Structure**:
  
  BASE_DIR/
  ├── reagan_81-88/
  │   ├── article1.txt
  │   ├── article2.txt
  │   └── ...
  ├── hwbush_89-92/
  │   ├── article1.txt
  │   └── ...
  └── ...

- Update `BASE_DIR` and `OUTPUT_DIR` in both scripts to match your local file paths.

## Running the Analysis

1. **Topic Modeling**:
   - Run `bertopic_analysis.py` to perform topic modeling for each era, save results, and analyze cross-era trends:
     python bertopic_analysis.py
   - Outputs include:
     - Per-era topic assignments (`<era_id>_topics.csv`)
     - Topic legends (`<era_id>_topic_legend.csv`)
     - Topic distribution visualizations (`<era_label>_topic_distribution.png`)
     - Cross-era topic frequency trends (`topic_frequencies.csv`, `topic_trends.png`, `topic_heatmap.png`)
     - Topic-centric reorganized data (`<category>_<label>.csv`)

2. **Sentiment Analysis**:
   - Run `sentiment_analysis.py` to analyze sentiment for each era, combine with topic data (if available), and compare across eras:
     python sentiment_analysis.py
   - Outputs include:
     - Per-era sentiment results (`<era_id>_sentiment.csv`)
     - Sentiment distribution visualizations (`<era_label>_sentiment_distribution.png`, `<era_label>_sentiment_ratio.png`)
     - Topic-sentiment analysis (`<era_id>_topic_sentiment.csv`, `<era_label>_topic_sentiment.png`) if topic data is provided
     - Cross-era sentiment comparisons (`era_sentiment_comparison.csv`, `era_sentiment_scores.png`, `era_sentiment_ratio.png`)
     - Topic sentiment trends (`<category>_sentiment_trend.csv`, `<category>_sentiment_trend.png`) if topic-centric data is available
     - Sentiment-event correlations (`sentiment_event_correlations.csv`, `sentiment_event_correlations.txt`) if topic trends are analyzed

3. **Integration**:
   - To combine topic and sentiment analyses, ensure `topic_results`, `topic_centric_dfs`, and `category_labels` from `bertopic_analysis.py` are available to `sentiment_analysis.py`. This can be done by running `bertopic_analysis.py` first and passing the results to `sentiment_analysis.py` or modifying the scripts to share data.

## Historical Context

The scripts reference key historical events to provide context for analysis:

- Tiananmen Square Protests (1989)
- Fall of Berlin Wall (1989)
- Soviet Union Dissolution (1991)
- Hong Kong Handover (1997)
- WTO Accession of China (2001)
- Beijing Olympics (2008)
- US-China Climate Agreement (2014)
- US-China Trade War Begins (2018)
- COVID-19 Pandemic Begins (2020)

These events are used to correlate sentiment shifts in the `sentiment_analysis.py` script.

## Notes

- **Performance**: Sentiment analysis can be computationally intensive. Use a GPU if available (automatically detected by `sentiment_analysis.py`). For testing, uncomment the `max_files` parameter to limit the number of files processed.
- **Customization**:
  - Adjust `nr_topics` in `bertopic_analysis.py` to change the number of topics per era.
  - Modify the sentiment threshold (e.g., 0.2 for significant shifts) in `correlate_sentiment_with_events` to tune event correlation sensitivity.
- **Limitations**:
  - Topic mapping across eras is simplistic and may require more sophisticated methods for better accuracy.
  - Sentiment analysis relies on sentence-level tokenization, which may miss nuanced context in articles.
  - The DistilBERT model is pre-trained and may not be fine-tuned for news-specific sentiment.

## Output Files

- **Topic Modeling**:
  - `<era_id>_topics.csv`: Article-level topic assignments.
  - `<era_id>_topic_legend.csv`: Description of topics with top words and sizes.
  - `<era_label>_topic_distribution.png`: Bar chart of topic distribution.
  - `topic_frequencies.csv`: Relative topic frequencies across eras.
  - `topic_trends.png`: Line chart of topic trends.
  - `topic_heatmap.png`: Heatmap of topic frequencies.
  - `<category>_<label>.csv`: Articles reorganized by topic category.

- **Sentiment Analysis**:
  - `<era_id>_sentiment.csv`: Article-level sentiment scores.
  - `<era_label>_sentiment_distribution.png`: Histograms of positive/negative sentiment.
  - `<era_label>_sentiment_ratio.png`: Histogram of positive/negative sentence ratios.
  - `<era_id>_topic_sentiment.csv`: Sentiment aggregated by topic (if topic data is available).
  - `<era_label>_topic_sentiment.png`: Bar chart of sentiment by topic.
  - `era_sentiment_comparison.csv`: Cross-era sentiment metrics.
  - `era_sentiment_scores.png`: Bar chart of average sentiment scores across eras.
  - `era_sentiment_ratio.png`: Bar chart of positive/negative ratios across eras.
  - `<category>_sentiment_trend.csv`: Sentiment trends for each topic category.
  - `<category>_sentiment_trend.png`: Line chart of topic sentiment trends.
  - `sentiment_event_correlations.csv`: Correlations between sentiment shifts and events.
  - `sentiment_event_correlations.txt`: Text summary of correlations.
