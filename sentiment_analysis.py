# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from tqdm import tqdm
import torch
from datetime import datetime
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# %%
# Download NLTK data for sentence tokenization
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# %%
# Set paths - update these to match your directory structure
BASE_DIR = r"/content/drive/MyDrive/extracted_files"  # Directory containing the era folders
OUTPUT_DIR = r"/content/drive/MyDrive/sentiment_results"  # Directory for results
SENTIMENT_DIR = os.path.join(OUTPUT_DIR, "Sentiment")
CROSS_ERA_DIR = os.path.join(OUTPUT_DIR, "CrossEraAnalysis")

# %%
# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(CROSS_ERA_DIR, exist_ok=True)

# %%
# Define eras for analysis
ERAS = {
    "REAGAN": {"folder": "reagan_81-88", "label": "Reagan (1981-1988)", "years": (1981, 1988)},
    "HWBUSH": {"folder": "hwbush_89-92", "label": "HW Bush (1989-1992)", "years": (1989, 1992)},
    "CLINTON": {"folder": "clinton_93-00", "label": "Clinton (1993-2000)", "years": (1993, 2000)},
    "GWBUSH": {"folder": "gwbush_01-08", "label": "GW Bush (2001-2008)", "years": (2001, 2008)},
    "OBAMA": {"folder": "obama_09-16", "label": "Obama (2009-2016)", "years": (2009, 2016)},
    "TRUMP": {"folder": "trump_17-20", "label": "Trump (2017-2020)", "years": (2017, 2020)},
}


# %%
# Define historical events for context
HISTORICAL_EVENTS = [
    {"event": "Tiananmen Square Protests", "date": "1989-06-04", "era": "HWBUSH"},
    {"event": "Fall of Berlin Wall", "date": "1989-11-09", "era": "HWBUSH"},
    {"event": "Soviet Union Dissolution", "date": "1991-12-26", "era": "HWBUSH"},
    {"event": "Hong Kong Handover", "date": "1997-07-01", "era": "CLINTON"},
    {"event": "WTO Accession of China", "date": "2001-12-11", "era": "GWBUSH"},
    {"event": "Beijing Olympics", "date": "2008-08-08", "era": "GWBUSH"},
    {"event": "US-China Climate Agreement", "date": "2014-11-12", "era": "OBAMA"},
    {"event": "US-China Trade War Begins", "date": "2018-07-06", "era": "TRUMP"},
    {"event": "COVID-19 Pandemic Begins", "date": "2020-01-20", "era": "TRUMP"},
]


# %%
def analyze_article_sentiment(article_text, sentiment_pipeline):
    """Analyze sentiment of an article by breaking it into sentences."""
    sentences = sent_tokenize(article_text)
    sentiments = sentiment_pipeline(sentences)

    # Initialize counters for positive and negative scores
    pos_scores, neg_scores = [], []

    for result in sentiments:
        if result['label'] == 'POSITIVE':
            pos_scores.append(result['score'])
        else:
            neg_scores.append(result['score'])

    # Calculate average scores
    avg_positive = sum(pos_scores)/len(pos_scores) if pos_scores else None
    avg_negative = sum(neg_scores)/len(neg_scores) if neg_scores else None

    pos_count = len(pos_scores)
    neg_count = len(neg_scores)

    # Return a dictionary with aggregated sentiment data
    return {
        "avg_positive": avg_positive,
        "avg_negative": avg_negative,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "num_sentences": len(sentences)
    }

def process_folder_sentiment(folder_path, sentiment_pipeline, max_files=None):
    """Process all articles in a folder for sentiment analysis."""
    results = []

    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]

    if max_files:
        file_list = file_list[:max_files]

    for file in tqdm(file_list, desc=f"Processing sentiment for {os.path.basename(folder_path)}"):
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                article_text = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        sentiment_data = analyze_article_sentiment(article_text, sentiment_pipeline)
        results.append({
            "file": file,
            "file_path": file_path,
            **sentiment_data
        })

    return pd.DataFrame(results)

def visualize_sentiment(df_results, era_label):
    """Create visualizations of sentiment distribution."""
    plt.figure(figsize=(12, 5))

    # Plot positive sentiment distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df_results['avg_positive'].dropna(), bins=20, kde=True, color='green')
    plt.title(f"Distribution of Positive Sentiment ({era_label})")
    plt.xlabel("Average Positive Sentiment Score")
    plt.ylabel("Frequency")

    # Plot negative sentiment distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df_results['avg_negative'].dropna(), bins=20, kde=True, color='red')
    plt.title(f"Distribution of Negative Sentiment ({era_label})")
    plt.xlabel("Average Negative Sentiment Score")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(SENTIMENT_DIR, f"{era_label}_sentiment_distribution.png"))
    plt.close()
    return os.path.join(SENTIMENT_DIR, f"{era_label}_sentiment_distribution.png")

def visualize_sentiment_ratio(df_results, era_label):
    """Visualize ratio of positive to negative sentences."""
    # Calculate ratio of positive to negative sentences
    df_results['sentiment_ratio'] = df_results['positive_count'] / (df_results['negative_count'] + 0.001)  # Avoid division by zero

    plt.figure(figsize=(10, 6))
    sns.histplot(df_results['sentiment_ratio'], bins=30, kde=True)
    plt.axvline(x=1, color='red', linestyle='--')
    plt.title(f"Ratio of Positive to Negative Sentences ({era_label})")
    plt.xlabel("Positive/Negative Ratio")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(SENTIMENT_DIR, f"{era_label}_sentiment_ratio.png"))
    plt.close()
    return os.path.join(SENTIMENT_DIR, f"{era_label}_sentiment_ratio.png")

def combine_sentiment_with_topics(sentiment_df, topic_df, era_id):
    """Combine sentiment analysis with topic data."""
    # Merge sentiment and topic data
    merged_df = pd.merge(sentiment_df, topic_df, on='file')

    # Group by topic and calculate average sentiment
    topic_sentiment = merged_df.groupby('topic_cluster_id').agg({
        'avg_positive': 'mean',
        'avg_negative': 'mean',
        'positive_count': 'sum',
        'negative_count': 'sum',
        'topic_label': 'first'
    }).reset_index()

    # Calculate sentiment ratio
    topic_sentiment['sentiment_ratio'] = topic_sentiment['positive_count'] / (topic_sentiment['negative_count'] + 0.001)

    # Save results
    topic_sentiment.to_csv(os.path.join(OUTPUT_DIR, f"{era_id}_topic_sentiment.csv"), index=False)

    return topic_sentiment

def visualize_sentiment_by_topic(topic_sentiment, era_label):
    """Create visualization of sentiment by topic."""
    if topic_sentiment is None or len(topic_sentiment) == 0:
        print(f"No topic sentiment data available for {era_label}")
        return None

    # Filter out outlier topic for visualization
    filtered_sentiment = topic_sentiment[topic_sentiment['topic_cluster_id'] != -1]

    if len(filtered_sentiment) == 0:
        print(f"No topic sentiment data (excluding outliers) available for {era_label}")
        return None

    plt.figure(figsize=(12, 8))

    # Plot sentiment ratio by topic
    bars = plt.bar(filtered_sentiment['topic_cluster_id'], filtered_sentiment['sentiment_ratio'], color='purple')

    # Add horizontal line at ratio = 1 (neutral sentiment)
    plt.axhline(y=1, color='red', linestyle='--')

    # Customize plot
    plt.xlabel('Topic ID')
    plt.ylabel('Positive/Negative Ratio')
    plt.title(f'Sentiment by Topic for {era_label}')
    plt.xticks(filtered_sentiment['topic_cluster_id'])

    # Add topic labels
    for i, topic_id in enumerate(filtered_sentiment['topic_cluster_id']):
        plt.annotate(
            filtered_sentiment.iloc[i]['topic_label'],
            xy=(topic_id, filtered_sentiment.iloc[i]['sentiment_ratio']),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            rotation=45,
            fontsize=8
        )

    plt.tight_layout()
    output_path = os.path.join(SENTIMENT_DIR, f"{era_label}_topic_sentiment.png")
    plt.savefig(output_path)
    plt.close()

    return output_path

def compare_sentiment_across_eras(all_sentiment_results):
    """Compare sentiment analysis results across different eras."""
    print("Comparing sentiment across eras...")

    # Prepare data for comparison
    era_sentiments = []

    for era_id, sentiment_df in all_sentiment_results.items():
        # Calculate average sentiment metrics for this era
        avg_positive = sentiment_df['avg_positive'].mean()
        avg_negative = sentiment_df['avg_negative'].mean()

        # Calculate positive/negative ratio
        total_pos = sentiment_df['positive_count'].sum()
        total_neg = sentiment_df['negative_count'].sum()
        pos_neg_ratio = total_pos / total_neg if total_neg > 0 else float('inf')

        era_sentiments.append({
            'era_id': era_id,
            'era_label': ERAS[era_id]['label'],
            'avg_positive': avg_positive,
            'avg_negative': avg_negative,
            'pos_neg_ratio': pos_neg_ratio,
            'start_year': ERAS[era_id]['years'][0],
            'end_year': ERAS[era_id]['years'][1]
        })

    # Convert to dataframe
    df_era_sentiment = pd.DataFrame(era_sentiments)
    df_era_sentiment.sort_values('start_year', inplace=True)

    # Save to CSV
    df_era_sentiment.to_csv(os.path.join(CROSS_ERA_DIR, 'era_sentiment_comparison.csv'), index=False)

    # Visualize sentiment comparison
    visualize_sentiment_comparison(df_era_sentiment)

    return df_era_sentiment

def visualize_sentiment_comparison(df_era_sentiment):
    """Create visualizations comparing sentiment across eras."""
    # Plot average positive and negative sentiment by era
    plt.figure(figsize=(12, 6))

    x = range(len(df_era_sentiment))
    width = 0.35

    plt.bar([i-width/2 for i in x], df_era_sentiment['avg_positive'], width=width, color='green', label='Avg. Positive Score')
    plt.bar([i+width/2 for i in x], df_era_sentiment['avg_negative'], width=width, color='red', label='Avg. Negative Score')

    plt.xlabel('Presidential Era')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Scores by Presidential Era')
    plt.xticks(x, df_era_sentiment['era_label'], rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(CROSS_ERA_DIR, 'era_sentiment_scores.png'))
    plt.close()

    # Plot positive/negative ratio by era
    plt.figure(figsize=(12, 6))

    plt.bar(df_era_sentiment['era_label'], df_era_sentiment['pos_neg_ratio'], color='purple')
    plt.axhline(y=1, color='red', linestyle='--', label='Neutral (1:1 ratio)')

    plt.xlabel('Presidential Era')
    plt.ylabel('Positive/Negative Ratio')
    plt.title('Ratio of Positive to Negative Sentences by Presidential Era')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(CROSS_ERA_DIR, 'era_sentiment_ratio.png'))
    plt.close()

    return (
        os.path.join(CROSS_ERA_DIR, 'era_sentiment_scores.png'),
        os.path.join(CROSS_ERA_DIR, 'era_sentiment_ratio.png')
    )

def analyze_topic_sentiment_over_time(topic_centric_dfs, category_labels, all_sentiment_results):
    """Analyze how sentiment for each topic changes over time."""
    print("Analyzing topic sentiment over time...")

    # For each topic category, compare sentiment across eras
    topic_sentiment_trends = {}

    for category, df in topic_centric_dfs.items():
        if len(df) < 10:  # Skip categories with too few articles
            continue

        # Merge with sentiment data
        era_sentiments = []

        for era_id, sentiment_df in all_sentiment_results.items():
            # Filter for articles in this category and era
            era_category_files = set(df[df['era'] == era_id]['file'])
            era_sentiment = sentiment_df[sentiment_df['file'].isin(era_category_files)]

            if len(era_sentiment) > 0:
                avg_positive = era_sentiment['avg_positive'].mean()
                avg_negative = era_sentiment['avg_negative'].mean()

                total_pos = era_sentiment['positive_count'].sum()
                total_neg = era_sentiment['negative_count'].sum()
                pos_neg_ratio = total_pos / total_neg if total_neg > 0 else float('inf')

                era_sentiments.append({
                    'era_id': era_id,
                    'era_label': ERAS[era_id]['label'],
                    'avg_positive': avg_positive,
                    'avg_negative': avg_negative,
                    'pos_neg_ratio': pos_neg_ratio,
                    'article_count': len(era_sentiment),
                    'start_year': ERAS[era_id]['years'][0],
                    'end_year': ERAS[era_id]['years'][1]
                })

        if len(era_sentiments) > 1:  # Need at least two eras for trend analysis
            category_sentiment_df = pd.DataFrame(era_sentiments)
            category_sentiment_df.sort_values('start_year', inplace=True)

            topic_sentiment_trends[category] = category_sentiment_df

            # Save category sentiment data
            category_label = category_labels.get(category, category)
            sanitized_label = category_label.split(',')[0].strip().replace(' ', '_')
            file_path = os.path.join(CROSS_ERA_DIR, f"{sanitized_label}_sentiment_trend.csv")
            category_sentiment_df.to_csv(file_path, index=False)

            # Visualize sentiment trend for this topic
            visualize_topic_sentiment_trend(category_sentiment_df, category_label)

    return topic_sentiment_trends

def visualize_topic_sentiment_trend(df, topic_label):
    """Visualize how sentiment for a specific topic changes over time."""
    plt.figure(figsize=(12, 6))

    # Plot sentiment ratio over time
    plt.plot(df['era_label'], df['pos_neg_ratio'], marker='o', linewidth=2, color='purple')
    plt.axhline(y=1, color='red', linestyle='--', label='Neutral (1:1 ratio)')

    plt.xlabel('Presidential Era')
    plt.ylabel('Positive/Negative Ratio')
    plt.title(f'Sentiment Trend for Topic: {topic_label}')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    sanitized_label = topic_label.split(',')[0].strip().replace(' ', '_')
    plt.savefig(os.path.join(CROSS_ERA_DIR, f"{sanitized_label}_sentiment_trend.png"))
    plt.close()

    return os.path.join(CROSS_ERA_DIR, f"{sanitized_label}_sentiment_trend.png")

def correlate_sentiment_with_events(sentiment_trends, events=HISTORICAL_EVENTS):
    """Identify potential correlations between sentiment shifts and historical events."""
    print("Correlating sentiment trends with historical events...")

    # Convert events to a more usable format
    event_years = {}
    for event in events:
        year = datetime.strptime(event['date'], '%Y-%m-%d').year
        if year not in event_years:
            event_years[year] = []
        event_years[year].append(event['event'])

    # For each topic's sentiment trend, look for significant shifts near events
    correlations = []

    for category, sentiment_df in sentiment_trends.items():
        if len(sentiment_df) < 3:  # Need enough data points
            continue

        # Look for significant shifts in sentiment
        for i in range(1, len(sentiment_df)):
            prev_row = sentiment_df.iloc[i-1]
            curr_row = sentiment_df.iloc[i]

            # Calculate sentiment change
            sentiment_change = curr_row['pos_neg_ratio'] - prev_row['pos_neg_ratio']

            # Check if change is significant
            if abs(sentiment_change) > 0.2:  # 20% change threshold
                direction = "positive" if sentiment_change > 0 else "negative"

                # Year range between administrations
                year_range = range(prev_row['end_year'], curr_row['start_year'] + 1)

                # Check for events in this range
                related_events = []
                for year in year_range:
                    if year in event_years:
                        related_events.extend(event_years[year])

                if related_events:
                    correlations.append({
                        'topic': category,
                        'sentiment_shift': direction,
                        'from_era': prev_row['era_label'],
                        'to_era': curr_row['era_label'],
                        'change_magnitude': abs(sentiment_change),
                        'potential_related_events': ', '.join(related_events)
                    })

    # Save correlations to file
    if correlations:
        correlations_df = pd.DataFrame(correlations)
        correlations_df.to_csv(os.path.join(CROSS_ERA_DIR, 'sentiment_event_correlations.csv'), index=False)

        # Also create a text summary
        with open(os.path.join(CROSS_ERA_DIR, 'sentiment_event_correlations.txt'), 'w') as f:
            f.write("Potential Correlations Between Sentiment Shifts and Historical Events\n")
            f.write("="*80 + "\n\n")

            for i, corr in enumerate(correlations):
                f.write(f"{i+1}. Topic: {corr['topic']}\n")
                f.write(f"   Shift: {corr['sentiment_shift']} (magnitude: {corr['change_magnitude']:.2f})\n")
                f.write(f"   Between: {corr['from_era']} and {corr['to_era']}\n")
                f.write(f"   Potentially related events: {corr['potential_related_events']}\n\n")

    return correlations

# %%
topic_results=None
topic_centric_dfs=None
category_labels=None

# %%
"""Main execution function for sentiment analysis."""
print("Starting sentiment analysis of NYT China coverage across presidential eras...\n")

# Initialize sentiment analysis pipeline
print("Initializing sentiment analysis pipeline...")
try:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        device=device
    )
except Exception as e:
    print(f"GPU error: {e}")
    print("Defaulting to CPU.")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True
    )

# %%
# Process each era
all_sentiment_results = {}

for era_id, era_info in ERAS.items():
    era_folder = era_info["folder"]
    era_label = era_info["label"]

    print(f"\n{'='*50}")
    print(f"Processing sentiment for {era_label} era")
    print(f"{'='*50}")

    # Full path to era folder
    era_folder_path = os.path.join(BASE_DIR, era_folder)

    # Run sentiment analysis
    print(f"Running sentiment analysis for {era_label}...")

    # Limit files for processing if needed (COMMENT THIS OUT FOR FULL ANALYSIS)
    # max_files = 100  # For faster testing

    # Run sentiment analysis
    df_sentiment = process_folder_sentiment(
        era_folder_path,
        sentiment_pipeline
        # max_files=max_files  # Uncomment for testing
    )
    all_sentiment_results[era_id] = df_sentiment

    # Save sentiment results
    sentiment_file = os.path.join(SENTIMENT_DIR, f"{era_id}_sentiment.csv")
    df_sentiment.to_csv(sentiment_file, index=False)
    print(f"Saved sentiment results to {sentiment_file}")

    # Visualize sentiment
    visualize_sentiment(df_sentiment, era_label)
    visualize_sentiment_ratio(df_sentiment, era_label)

    # If topic data is available, combine with sentiment
    if topic_results and era_id in topic_results:
        topic_sentiment = combine_sentiment_with_topics(df_sentiment, topic_results[era_id], era_id)
        visualize_sentiment_by_topic(topic_sentiment, era_label)

print("\n" + "="*50)
print("Performing cross-era sentiment analysis")
print("="*50)

# %%




# Compare sentiment across eras
df_sentiment_comparison = compare_sentiment_across_eras(all_sentiment_results)

# If topic-centric data is available, analyze sentiment trends by topic
topic_sentiment_trends = None
if topic_centric_dfs and category_labels:
    topic_sentiment_trends = analyze_topic_sentiment_over_time(
        topic_centric_dfs,
        category_labels,
        all_sentiment_results
    )

    # Correlate sentiment shifts with historical events
    correlate_sentiment_with_events(topic_sentiment_trends)



# %%

print("\n" + "="*50)
print("Sentiment analysis complete!")
print("="*50)
print(f"All results saved in {OUTPUT_DIR}")


