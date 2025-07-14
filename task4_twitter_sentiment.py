import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from collections import Counter

# Set plot style
plt.style.use("ggplot")

# Step 1: Load dataset
df = pd.read_csv("C:\\Users\\megha\\Desktop\\CODECRAFT_DS_04\\twitter_training.csv", header=None, encoding='latin1')

df.columns = ['tweet_id', 'entity', 'sentiment', 'tweet_text']
print(" Dataset loaded:", df.shape)
print(" Columns:", df.columns.tolist())

# Step 2: Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text), flags=re.MULTILINE)
    text = re.sub(r'\@[\w]*', '', text)
    text = re.sub(r'\#[\w]*', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    return text

df['cleaned_text'] = df['tweet_text'].apply(clean_text)

# Step 3: Sentiment distribution
plt.figure(figsize=(8, 5))
colors = sns.color_palette("pastel")[0:5]
sns.countplot(data=df, x='sentiment', palette=colors)
plt.title("Sentiment Distribution", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Tweet Count", fontsize=12)
plt.tight_layout()
plt.show()

# Step 4: WordCloud for Positive Sentiment
positive_words = " ".join(df[df['sentiment'] == 'Positive']['cleaned_text'].dropna())
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(positive_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Positive Tweets", fontsize=16)
plt.tight_layout()
plt.show()

# Step 5: WordCloud for Negative Sentiment
negative_words = " ".join(df[df['sentiment'] == 'Negative']['cleaned_text'].dropna())
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Negative Tweets", fontsize=16)
plt.tight_layout()
plt.show()

# Step 6: Top words per sentiment (Bar chart)
def get_top_words_by_sentiment(df, sentiment, n=10):
    text = " ".join(df[df['sentiment'] == sentiment]['cleaned_text'].dropna())
    words = text.split()
    counter = Counter(words)
    return counter.most_common(n)

sentiments = ['Positive', 'Negative', 'Neutral']
fig, axes = plt.subp
