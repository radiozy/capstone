# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from tqdm import tqdm

# %%
# Set paths - update these to match your directory structure
BASE_DIR = r"C:\Users\radio\Downloads\CAPSTONE"  # Directory containing the era folders
OUTPUT_DIR = r"C:\Users\radio\Downloads\CAPSTONE\bertopic_results"  # Directory for results
TOPICS_DIR = os.path.join(OUTPUT_DIR, "Topics")
CROSS_ERA_DIR = os.path.join(OUTPUT_DIR, "CrossEraAnalysis")

# %%
# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TOPICS_DIR, exist_ok=True)
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
def load_texts_from_folder(folder_path):
    """Load text files from a folder into a dataframe."""
    filenames = []
    texts = []

    print(f"Loading texts from {folder_path}...")
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            try:
                with open(os.path.join(folder_path, fname), 'r', encoding='latin-1') as f:
                    content = f.read().strip()
                    filenames.append(fname)
                    texts.append(content)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue

    df = pd.DataFrame({
        'file': filenames,
        'text': texts
    })
    print(f"Loaded {len(texts)} articles")
    return df



# %%
def run_bertopic_on_era(df, era_id, era_label, nr_topics=8):
    """Run BERTopic analysis on a dataframe of articles."""
    print(f"Running BERTopic analysis for {era_label} era...")
    
    # Build enhanced stopword list
    base_stopwords = text.ENGLISH_STOP_WORDS
    generic_words = {
        "china", "also", "one", "two", "say", "says", "said",
        "new", "united", "states", "government", "meet", "people",
        "percent", "million", "chinese", "president", "administration",
        "american", "year", "years", "chinas", "americas", "beijing", "peking"
    }
    short_words = {w for w in base_stopwords if len(w) <= 2}
    custom_stopwords = list(base_stopwords.union(generic_words).union(short_words))

    # Create BERTopic model with custom vectorizer
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords)
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    
    # Fit the model and transform data
    topics, probs = topic_model.fit_transform(df["text"].tolist())

    # Reduce to specific number of topics if requested
    if nr_topics is not None:
        topic_model = topic_model.reduce_topics(df["text"].tolist(), nr_topics=nr_topics)
        topics = topic_model.topics_

    # Add topic information to dataframe
    df["topic_cluster_id"] = topics
    df["topic_label"] = [
        topic_model.get_topic(t)[0][0] if t != -1 and topic_model.get_topic(t) else "Outlier"
        for t in topics
    ]
    df["era"] = era_id
    df["era_label"] = era_label

    return topic_model, df

def save_topic_results(df, era_id):
    """Save topic analysis results to CSV."""
    save_path = os.path.join(OUTPUT_DIR, f"{era_id}_topics.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved topic results to {save_path}")
    return save_path

def save_topic_legend(topic_model, era_id, top_n_words=10):
    """Save a legend/description of topics."""
    output_csv_path = os.path.join(OUTPUT_DIR, f"{era_id}_topic_legend.csv")
    
    topic_info = topic_model.get_topic_info()
    legend_rows = []
    
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            label = "Outlier"
            top_words = ""
        else:
            words = topic_model.get_topic(topic_id)
            top_words = ", ".join([w[0] for w in words[:top_n_words]])
            label = top_words

        size = topic_info[topic_info["Topic"] == topic_id]["Count"].values[0]
        legend_rows.append({
            "topic_id": topic_id,
            "top_words": top_words,
            "topic_size": size,
            "era": era_id
        })

    legend_df = pd.DataFrame(legend_rows)
    legend_df.to_csv(output_csv_path, index=False)
    print(f"Topic legend saved to {output_csv_path}")
    return legend_df

def visualize_topic_distribution(topic_legend, era_label):
    """Create visualization of topic distribution."""
    # Filter out outlier topic (-1)
    filtered_legend = topic_legend[topic_legend['topic_id'] != -1]
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(y=filtered_legend['topic_id'], width=filtered_legend['topic_size'], color='skyblue')
    
    # Truncate topic labels for readability
    topic_labels = [str(row['topic_id']) + ": " + row['top_words'].split(',')[0] for _, row in filtered_legend.iterrows()]
    
    plt.yticks(filtered_legend['topic_id'], topic_labels)
    plt.xlabel('Number of Articles')
    plt.title(f'Topic Distribution for {era_label}')
    plt.tight_layout()
    
    plt.savefig(os.path.join(TOPICS_DIR, f"{era_label}_topic_distribution.png"))
    plt.close()
    return os.path.join(TOPICS_DIR, f"{era_label}_topic_distribution.png")

def create_topic_mapping(all_topic_legends):
    """Create a mapping of similar topics across eras."""
    # This is a simplistic approach - in practice you might want more sophisticated topic matching
    all_topics = {}
    topic_to_category = {}
    category_counter = 0
    
    # Extract key words from each topic
    for era, legend_df in all_topic_legends.items():
        for _, row in legend_df.iterrows():
            if row['topic_id'] == -1:  # Skip outliers
                continue
                
            # Get top words from this topic
            top_words = [word.strip() for word in row['top_words'].split(',')[:3] if word.strip()]
            
            # Check if this topic matches any existing category
            matched = False
            for category, words in all_topics.items():
                # Check for overlap in key words
                if any(word in words for word in top_words[:2]):
                    topic_to_category[(era, row['topic_id'])] = category
                    all_topics[category].update(top_words)
                    matched = True
                    break
            
            # If no match, create new category
            if not matched:
                category = f"CATEGORY_{category_counter}"
                category_counter += 1
                topic_to_category[(era, row['topic_id'])] = category
                all_topics[category] = set(top_words)
    
    # Create a readable label for each category based on most common words
    category_labels = {}
    for category, words in all_topics.items():
        category_labels[category] = ", ".join(list(words)[:3])
    
    return topic_to_category, category_labels

def analyze_topic_trends_across_eras(all_results, all_topic_legends):
    """Analyze how topics change in frequency across different eras."""
    print("Analyzing topic trends across eras...")
    
    # Create mapping of similar topics across eras
    topic_to_category, category_labels = create_topic_mapping(all_topic_legends)
    
    # Count category occurrences in each era
    era_category_counts = {}
    for era_id in ERAS.keys():
        if era_id not in all_results:
            continue
            
        df = all_results[era_id]
        era_category_counts[era_id] = {}
        
        for _, row in df.iterrows():
            category = topic_to_category.get((era_id, row['topic_cluster_id']), None)
            if category:
                if category not in era_category_counts[era_id]:
                    era_category_counts[era_id][category] = 0
                era_category_counts[era_id][category] += 1
    
    # Calculate relative frequencies
    era_frequencies = {}
    for era_id, counts in era_category_counts.items():
        total = sum(counts.values())
        era_frequencies[era_id] = {cat: count/total for cat, count in counts.items()}
    
    # Prepare dataframe for visualization
    rows = []
    for category in set().union(*[counts.keys() for counts in era_category_counts.values()]):
        row = {'category': category, 'label': category_labels.get(category, category)}
        for era_id in ERAS.keys():
            if era_id in era_frequencies:
                row[era_id] = era_frequencies[era_id].get(category, 0)
            else:
                row[era_id] = 0
        rows.append(row)
    
    df_trends = pd.DataFrame(rows)
    
    # Identify increasing, decreasing, and consistent topics
    trend_analysis = {'increasing': [], 'decreasing': [], 'consistent': [], 'fluctuating': []}
    
    for _, row in df_trends.iterrows():
        frequencies = [row[era_id] for era_id in ERAS.keys() if era_id in row and not pd.isna(row[era_id])]
        if len(frequencies) < 3:  # Need at least 3 data points for trend analysis
            continue
            
        # Simple trend analysis
        if all(frequencies[i] <= frequencies[i+1] for i in range(len(frequencies)-1)):
            trend_analysis['increasing'].append(row['label'])
        elif all(frequencies[i] >= frequencies[i+1] for i in range(len(frequencies)-1)):
            trend_analysis['decreasing'].append(row['label'])
        elif max(frequencies) - min(frequencies) < 0.05:  # Threshold for consistency
            trend_analysis['consistent'].append(row['label'])
        else:
            trend_analysis['fluctuating'].append(row['label'])
    
    # Save trend analysis
    with open(os.path.join(CROSS_ERA_DIR, 'topic_trends.txt'), 'w') as f:
        for trend, topics in trend_analysis.items():
            f.write(f"{trend.capitalize()} topics:\n")
            for topic in topics:
                f.write(f"- {topic}\n")
            f.write("\n")
    
    # Save frequencies for further analysis
    df_trends.to_csv(os.path.join(CROSS_ERA_DIR, 'topic_frequencies.csv'), index=False)
    
    # Visualize trends
    visualize_topic_trends(df_trends, ERAS)
    
    return df_trends, trend_analysis

def visualize_topic_trends(df_trends, eras):
    """Visualize how topics change over time."""
    # Select top topics by average frequency
    era_columns = [era_id for era_id in eras.keys() if era_id in df_trends.columns]
    df_trends['avg_freq'] = df_trends[era_columns].mean(axis=1)
    top_topics = df_trends.nlargest(8, 'avg_freq')
    
    # Plot line chart of topic frequencies over time
    plt.figure(figsize=(14, 8))
    
    for _, row in top_topics.iterrows():
        frequencies = [row[era_id] if era_id in row else 0 for era_id in era_columns]
        plt.plot(era_columns, frequencies, marker='o', linewidth=2, label=row['label'])
    
    plt.xlabel('Presidential Era')
    plt.ylabel('Relative Frequency')
    plt.title('Topic Frequency Trends Across Presidential Eras')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(CROSS_ERA_DIR, 'topic_trends.png'))
    plt.close()
    
    # Plot heatmap of topic frequencies
    plt.figure(figsize=(12, 10))
    
    # Prepare data for heatmap
    heatmap_data = top_topics[era_columns].values
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        xticklabels=[eras[era_id]["label"] for era_id in era_columns],
        yticklabels=top_topics['label'],
        cbar_kws={'label': 'Relative Frequency'}
    )
    
    plt.title('Topic Frequency Heatmap Across Presidential Eras')
    plt.tight_layout()
    
    plt.savefig(os.path.join(CROSS_ERA_DIR, 'topic_heatmap.png'))
    plt.close()

def reorganize_by_topics(all_results, all_topic_legends):
    """Reorganize corpus by topics rather than by eras."""
    print("Reorganizing corpus by topics rather than eras...")
    
    # Create mapping of similar topics across eras
    topic_to_category, category_labels = create_topic_mapping(all_topic_legends)
    
    # For each era and topic, assign a universal category
    for era_id, df in all_results.items():
        df['topic_category'] = df['topic_cluster_id'].apply(
            lambda t: topic_to_category.get((era_id, t), "UNCATEGORIZED")
        )
        df['topic_category_label'] = df['topic_category'].apply(
            lambda c: category_labels.get(c, "Uncategorized")
        )
    
    # Save reorganized data
    topic_centric_dfs = {}
    
    for category, label in category_labels.items():
        category_articles = []
        
        for era_id, df in all_results.items():
            era_category_articles = df[df['topic_category'] == category].copy()
            category_articles.append(era_category_articles)
        
        if category_articles:
            topic_centric_df = pd.concat(category_articles, ignore_index=True)
            topic_centric_df.to_csv(os.path.join(OUTPUT_DIR, f"{category}_{label.split(',')[0]}.csv"), index=False)
            topic_centric_dfs[category] = topic_centric_df
    
    return topic_centric_dfs, category_labels


# %%
"""Main execution function for BERTopic analysis."""
print("Starting BERTopic analysis of NYT China coverage across presidential eras...\n")

# Process each era
all_topic_results = {}
all_topic_legends = {}

for era_id, era_info in ERAS.items():
    era_folder = era_info["folder"]
    era_label = era_info["label"]
    
    print(f"\n{'='*50}")
    print(f"Processing {era_label} era")
    print(f"{'='*50}")
    
    # Full path to era folder
    era_folder_path = os.path.join(BASE_DIR, era_folder)
    
    # Load articles
    df_era = load_texts_from_folder(era_folder_path)
    
    # Run topic modeling
    topic_model, df_topics = run_bertopic_on_era(df_era, era_id, era_label)
    all_topic_results[era_id] = df_topics
    
    # Save topic results
    save_topic_results(df_topics, era_id)
    
    # Save topic legend
    topic_legend = save_topic_legend(topic_model, era_id)
    all_topic_legends[era_id] = topic_legend
    
    # Visualize topic distribution
    visualize_topic_distribution(topic_legend, era_label)

print("\n" + "="*50)
print("Performing cross-era topic analysis")
print("="*50)





# %%
# Cross-Era Topic Analysis
df_topic_trends, trend_analysis = analyze_topic_trends_across_eras(all_topic_results, all_topic_legends)



# %%
# Topic-Centric Reorganization
topic_centric_dfs, category_labels = reorganize_by_topics(all_topic_results, all_topic_legends)


# %%
print("\n" + "="*50)
print("BERTopic analysis complete!")
print("="*50)
print(f"All results saved in {OUTPUT_DIR}")


