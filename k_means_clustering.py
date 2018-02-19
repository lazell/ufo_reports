
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from relative_popularity_reports import load_and_process


'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
TF-IDF Vectorizer and K-means Clustering
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''


def top_10_events():
    # Combining the top 10 events and clusering them

    df = load_and_process('data/data.pkl')
    df.drop_duplicates(inplace=True)
    df['date_'] = df['occurred_clean'].map(lambda x: str(x)[:10])
    top_10_dates = ['1999-11-16',
                    '1997-03-13',
                    '1997-07-04',
                    '1999-09-01',
                    '2015-11-07',
                    '2014-07-04',
                    '1997-11-14',
                    '1995-03-15',
                    '1978-07-01',
                    '1978-08-01']

    combined_top_10 = df[df['date_'].isin(top_10_dates)]
    # vectorizer with manually tuned parameters
    ufo_stop_words2 = [x.strip() for x in open('n_grams_stop_words.txt','r').read().split('\n')]
    vectorizer_top_10 = TfidfVectorizer(max_df=0.25,
                                   min_df=0.05,
                                   stop_words=ufo_stop_words2,
                                   max_features=12000,
                                   ngram_range=(1,4),
                                   norm='l1')

    # Cluster and output group n-grams
    top_10_model, top_10_scores = fit_kmeans(combined_top_10['report'], 4,
                                             vectorizer_top_10)




def fit_kmeans(df_data, k, vectorizer):
    # Vectorize text
    X = vectorizer.fit_transform(df_data)

    # Vector text counts
    matrix_counts = X.toarray()

    # fit kmeans model
    kmeans = KMeans(n_clusters = k).fit(X)

    # Get feature names
    fnames = np.array(vectorizer.get_feature_names())

    # Get length of labels return
    print ("No. of reports {} \n".format(kmeans.labels_.shape[0]))

    labels = [fnames[np.argsort(kmeans.cluster_centers_[i])[-50:]] for i in range(0,k)]

    for i in range(0,k):
        print ("Words: \n")
        print(i, labels[i], '\n\n')

    fname_scores = pd.DataFrame(pd.DataFrame(matrix_counts,
                                             columns=fnames).sum())
    fname_scores[0] = round(fname_scores[0]*100,0).astype(int)

    return kmeans, fname_scores.sort_values(0, ascending=False)

def assign_labels_bar_charts(df, kmeans, **kwargs):
    # Add Category labels to DataFrame and plot charts
    df['report labels'] = kmeans.labels_

    colors = ['#8DBE78', '#FAB669', '#BE78B0', '#FA6D69']

    df['Cluster Group Number'] = kmeans.labels_
    pd.crosstab(df['shape'], df['Cluster Group Number']).plot(kind='bar',
                                                              stacked=True,
                                                              figsize=(10,6),
                                                              **kwargs,
                                                              color=colors)
    pd.crosstab(df['state'], df['Cluster Group Number']).plot(kind='bar',
                                                              stacked=True,
                                                              figsize=(10,6),
                                                              **kwargs,
                                                              color=colors)

    pd.crosstab(df['date_'], df['Cluster Group Number']).plot(kind='bar',
                                                              stacked=True,
                                                              figsize=(10,6),
                                                              **kwargs,
                                                              color=colors)
    pd.crosstab(df[df['duration_clean'] < 1000]['duration_clean'],
                df['report labels']).plot(kind='bar',
                                          figsize=(10,6),
                                          **kwargs,
                                          color=colors)

if __name__ == '__main__':
    top_10_events()
