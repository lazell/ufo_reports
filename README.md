## What the UFO?
##### Exploratory analysis with Machine Learning clustering and Natural Language Processing of UFO reports for the celestially curious


![celestial](/images/ryan-lange-552065.jpg)


This project explores the data collected by The National UFO Reporting center in an attempt to understand the different types of sightings reported. I focused my efforts finding the most reported events in the United States to look at the collective coherence of their descriptions.


### The Dataset
The National UFO Reporting Center is a non-profit organization which has been collecting reports of UFO sightings since the 1970s. While the vast majority of the reports were easily explained by human and natural events, a number of cases are investigated by the organization each year. The dataset contains historical and present day reports dating back to 1762. For this project I used a United States data subset of 94,057 reports (years 1947-2017).

### Most Reported Events
I was curious to see what the most reported events were. Since the events with the highest number of reports were skewed towards the years 2014-2017, I chose to account for population growth from the by factoring in yearly estimates of online users in the United States to reduce this recent years skew. I also removed 'approximate bucket dates' e.g dates which were entered as 1st June, since the exact month, day and year were unknown or estimated in the report. This mainly impacted data prior 1995. After doing so, the following top 10 dates emerged as highly reported (1591 reports).

![top_events](/images/US_most_reported_events.png)

Notice that of the top 10, three dates have a significantly higher popularity score than the rest, one of which occurred on 4th July, in fact there are 2 years in which 4th of July were highly reported (1997 and 2014). My question is, of these top 10 events are there any distinct themes in the event descriptions?

To help answer this question, let's take a look at this subset's report content using basic natural language processing and using the similarity of language used to group the reports.

### Topic Analysis - Unsupervised Learning

The reports typically have a detailed paragraph written by the witness which describes the event in detail. To capture topics (an attempt to categorize reports based on description), I used the unsupervised machine learning method of k-means clustering.

#### K-means Clustering on Text Explained...
Feel free to skip this part, This is an overview of how the algorithm works for text analysis.

This process creates a number of matrices:

*  A term-frequency matrix which counts the number of times a certain term (i.e. word or combination of words) is mentioned in a report.

* An inverse-document-frequency matrix which is a (logarithmically scaled) inverse fraction to represent a term's frequency in the whole dataset (total number of documents in corpus divided by the number of documents containing the term)

* These two matrices are then multiplied to create a vector for each report.

Once we have our TF-IDF matrix of vectors, we can compare their similarity to each other by using k-means clustering. This algorithm randomly chooses k number of vectors to initialize the k centroids. Think of these centroids as the comparison reference vector. For each iteration the following happens:

1. Each report vector is compared to all the centroids by euclidian distance (straight line distance) and assigned (or re-assigned) to the closest centroid.

2. Once all vectors have been assigned to a centroid, the average "middle" vector of the cluster is calculated from it's cluster members, this becomes the new centroid vector for comparison in the next iteration.

The two steps are repeated many times. At early iterations it is likely the centroids will be closer to each other, they will adapt and move until the best fit is found.


#### Visualization of k-means Clustering (4 Clusters)

![k-means_demo](https://camo.githubusercontent.com/9394c353adeeed261a0fd0588e2600f5e696433e/687474703a2f2f692e696d6775722e636f6d2f755a4b714b58692e676966)

Image reference: https://github.com/vinhkhuc/VanillaML

Practically speaking scikit-learn's library uses 300 iterations as the default, it runs the clustering model 10 times (with 10 different seeds) and chooses the best output based on inertia as the model.

### Clustering Method and Results

I ran a text vectorizer on the report text after removing common and custom stop words and applied TF-IDF (Term Frequency- Inverse Document Frequency) to the dataset. For this analysis I ran a TF-IDF vectorizer using n-grams ranging from a single word up to a string of 4 words and reduced the dimensionality down to 12000 features using L1 regularization (LASSO).

Here are the top 50 n-gram results for 4 clusters

#### Cluster 1 - 4th July fireball, fireworks and flashing lights of various colors

['looking' 'show' 'quickly' 'round' 'clouds' 'behind' 'wife' 'circular'
 'watched' 'across sky' 'minute' 'fireball' 'still' '30' 'approximately'
 'line' 'night' 'slow' 'west east' 'speed' 'glowing' 'fast' 'direction'
 'watching' 'green' 'seemed' 'later' '10' 'across' 'fly' '4th' 'noticed'
 'july' 'sound' 'flying' 'moved' 'two' 'traveling' 'ball' 'slowly' 'high'
 'flashing' 'formation' 'minutes' 'disappeared' 'north' 'objects'
 'fireworks' 'red' 'orange']

#### Cluster 2 - Admin notes, blue white light and US Navy missile launch references

 ['contact' 'provides contact information' 'provides contact'
  'information pd' 'contact information pd' 'provides' 'contact information'
  'clouds' 'slowly' 'shaped' 'glowing' 'leaving' 'shape' 'looked like'
  'light sky' 'away' 'ball' 'craft' 'blue light' 'bright white'
  'disappeared' 'left' 'minutes' 'large' 'smoke' 'us' 'trail' 'bright light'
  'white light' 'note us' 'note us navy' 'us navy' 'us navy missile'
  'us navy missile launch' 'note us navy missile' 'green'
  'note navy missile launch' 'note navy missile' 'note navy' 'cloud'
  'navy missile launch pd' 'navy missile' 'navy missile launch'
  'missile launch pd' 'navy' 'launch pd' 'missile launch' 'missile' 'blue'
  'launch']

#### Cluster 3 - Moving objects in formation of various colors

['aircraft' 'shaped' 'witnessed' 'around' 'seemed' 'behind' 'slowly'
 'sound' 'direction' 'fire' 'meteor' 'red' 'area' 'said' 'three' 'fast'
 'noticed' 'flying' 'wife' 'seconds' 'moved' 'orange' 'ufo' 'craft'
 'thought' 'night' 'speed' 'large' 'tail' 'shape' 'traveling' 'fireball'
 'across' 'green' 'horizon' 'summary' 'formation' 'blue' 'fireworks'
 'north' 'two' 'objects']


#### Cluster 4 - Mostly descriptive words beginning with letters: L, F, and G!

['longer' 'long' 'little' 'line' 'light sky' 'length' 'left' 'look' 'end'
 'heard' 'headed' 'flew' 'flashing' 'first thought' 'fireworks' 'firework'
 'fireball' 'fire' 'flight' 'feet' 'far' 'faded' 'extremely' 'ever' 'event'
 'evening' 'even' 'fast' 'fly' 'flying' 'followed' 'head' 'happened' 'half'
 'ground' 'green' 'got' 'gone' 'going' 'yellow' 'go' 'glowing' 'glow' 'get'
 'front' 'friend' 'formation' 'following' 'heading' 'information']

By looking at these word and phrase groupings, the two clusters which jumped out at me are the first two which relate to 4th July - Independence Day and the US Navy missile launch. I was curious to see if the latter cluster represented a specific event, so I plotted the cluster groups the k-means model assigned each report.

![date_clusters](/images/k_means_clusters.png)

Interestingly reports in cluster 1 were mostly 4th July 2014 (and not 1997). Cluster 2 were almost entirely November 7th 2015. These indicate that the language used by witnesses are similar enough for an unsupervised model to pick up on with no training or access to the dates.

Not surprisingly, cluster 4 is spread out across all dates since they are mostly generic descriptive words which would score similar to each other given the nature of the text.

To investigate this further I decided to analyze the report text in a more targeted way, by event date. I chose the following dates for these reasons:

* 1999-7-4 for it's large percentage of reports in cluster 1 and it's high relative popularity score
* 2015-11-07 for it's large percentage of reports in cluster 2
* 1999-11-16 for it's high relative popularity score
* 1997-3-13 for a high relative popularity score

For these dates, I followed a similar technique of TF-IDF above, to get a summary of popular n-grams (mostly 2 word bi-grams) for those events and represented a selection of results in a word-cloud (Please note: these are only a selection of words from the analysis). The more frequent a bi-gram appeared in the group, the larger the lettering. The larger the letters are compared to other dates the more consistency in the event descriptions.

#### Independence Day - July 4th 1999
<insert analysis and word cloud here >


#### Midwest Fireball - November 16th 1999

An unusually bright fireball with a low altitude trajectory was seen from the midwestern states on this evening. It preceded the Leonids meter shower, which is an annual celestial event:
[news report](https://science.nasa.gov/science-news/science-at-nasa/1999/ast17nov99_1)
![Fireball](/images/1999-fireball-word-cloud.png)


#### Phoenix Lights - March 13th 1997

The event was explained by the National Guard dropping flares from aircraft and aircraft flying at high altitude in formation, although this is regarded as a controversial explanation for many eye-witnesses. Further reading: [wikipedia page](https://en.wikipedia.org/wiki/Phoenix_Lights)
![Phoenix_Lights](/images/phx-lights-word-cloud.png)


#### Strange Sightings in Los Angeles - November 7th 2015

This sighting was explained by a US Navy test missile which was launched off the Californian coast near LA: [news report](https://www.theguardian.com/us-news/2015/nov/08/navy-missile-launch-california-bright-light)
![LA_sighting](/images/blue-light-word-cloud.png)

### Sentiment Analysis on the US Dataset

I did a preliminary sentiment analysis on the data (positive emotion, negative emotion and amount of subjectivity). For this task, I used a lightweight simple NLP library; TextBlob, to explore the sentiment of the dataset.

I did not end up exploring sentiment analysis fully since the majority of sentiment was non-extreme or near neutral for both emotion and subjectivity.

Only 31 had highly positive emotion and only 4 had extremely negative emotion.

Only 20% of reports had scores over 0.5 in subjectivity, which indicates the corpus of reports are quite objective in their descriptions.

### Additional Insights on the US Dataset

According to admin notes, The National UFO Reporting Center has contacted a witness for 797 reports (0.82%)

272 (0.28%) reports contain hyperlinks to youtube.com footage

Out of all US reports, only (0.12%) mention the word 'abducted'

![top_events](/images/US_cities_reports.png)

## Conclusions

K-means clustering is a great tool for exploring latent sub-groups of data. Without the clustering, my intuition based on my initial analysis of the top 10 events would not have lead me to investigate the missile launch event of November 2015. Nor had I much reason or interest to investigate the 4th July 1999 event.

The varying strength of description similarities make it difficult for a skeptic to dismiss the validity of the event descriptions, especially events which were highly reported. In support this argument, the language of the reports were found to be, for the most part, objective in nature and rarely with high emotion attached.

The additional insights support the notion that the people reporting the events are mostly intrigued citizens, some of which NUFORC have requested further information.
