import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset from the given path, assuming the file is named 'arxiv-metadata-oai-snapshot.json'
# For demonstration purposes, I'm loading a local copy. Replace 'local_path' with the appropriate URL or path.
local_path = 'arxiv.json'
df = pd.read_json(local_path, lines=True, chunksize=1000)
df = next(df)  # Get the first chunk (first 1000 rows)

# Display the first few rows to understand the structure
print(df.head())

# Display the columns
print(df.columns)

# Display basic information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Remove duplicates based on a subset of columns that do not contain lists
df = df.drop_duplicates(subset=['id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi', 'report-no', 'categories', 'license', 'abstract', 'update_date'])

# Drop rows with missing essential columns
df = df.dropna(subset=['title', 'abstract', 'authors', 'update_date'])

# Convert the 'update_date' column to datetime
df['update_date'] = pd.to_datetime(df['update_date'])

# Combine titles and abstracts into a single text column
df['text'] = df['title'] + ' ' + df['abstract']

# Tokenize the text
df['tokens'] = df['text'].apply(word_tokenize)

# Remove stopwords and non-alphabetic tokens, and lowercase all words
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word.lower() for word in x if word.isalpha() and word.lower() not in stop_words])

# Create a column for keywords by finding the most common words
df['keywords'] = df['tokens'].apply(lambda x: [item[0] for item in Counter(x).most_common(10)])

# Flatten the list of keywords and count the frequency of each keyword
all_keywords = [keyword for sublist in df['keywords'] for keyword in sublist]
keyword_freq = Counter(all_keywords)

# Convert to a DataFrame for visualization
keyword_df = pd.DataFrame(keyword_freq.items(), columns=['keyword', 'frequency']).sort_values(by='frequency', ascending=False).head(20)

# Plot the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='frequency', y='keyword', data=keyword_df)
plt.title('Top 20 Keywords')
plt.show()

# Extract the year from the publication date
df['year'] = df['update_date'].dt.year

# Count the number of publications per year
yearly_counts = df.groupby('year').size().reset_index(name='counts')

# Plot the line graph
plt.figure(figsize=(12, 8))
sns.lineplot(x='year', y='counts', data=yearly_counts)
plt.title('Number of Publications Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.show()

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_keywords))

# Display the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Keywords')
plt.show()
