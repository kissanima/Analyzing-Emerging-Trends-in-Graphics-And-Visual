import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


nltk.download('punkt')
nltk.download('stopwords')


local_path = 'arxiv.json'
df = pd.read_json(local_path, lines=True, chunksize=1000)
df = next(df)  # Load only the first 1000 rows so that my PC dont explode


print(df.head())


print(df.columns)


print(df.info())


print(df.describe())


print(df.isnull().sum())


df = df.drop_duplicates(subset=['id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi', 'report-no', 'categories', 'license', 'abstract', 'update_date'])


df = df.dropna(subset=['title', 'abstract', 'authors', 'update_date'])


df['update_date'] = pd.to_datetime(df['update_date'])


df['text'] = df['title'] + ' ' + df['abstract']


df['tokens'] = df['text'].apply(word_tokenize)


stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word.lower() for word in x if word.isalpha() and word.lower() not in stop_words])


df['keywords'] = df['tokens'].apply(lambda x: [item[0] for item in Counter(x).most_common(10)])


all_keywords = [keyword for sublist in df['keywords'] for keyword in sublist]
keyword_freq = Counter(all_keywords)


keyword_df = pd.DataFrame(keyword_freq.items(), columns=['keyword', 'frequency']).sort_values(by='frequency', ascending=False).head(20)


plt.figure(figsize=(12, 8))
sns.barplot(x='frequency', y='keyword', data=keyword_df)
plt.title('Top 20 Keywords')
plt.show()


df['year'] = df['update_date'].dt.year

yearly_counts = df.groupby('year').size().reset_index(name='counts')


plt.figure(figsize=(12, 8))
sns.lineplot(x='year', y='counts', data=yearly_counts)
plt.title('Number of Publications Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.show()


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_keywords))


plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Keywords')
plt.show()
