## What the UFO?
##### Exploratory analysis with Clustering and Natural Language Processing of UFO reports for the celestially curious


![celestial](/images/ryan-lange-552065.jpg)


This project explores the data collected by The National UFO Reporting center in an attempt to understand the nature and
significance of the reports the organization has received.
Due to the size of the dataset, analysis can be approached from many
angles. I decided to focus on the most heavily reported
events, to understand how coherent they were, collectively,
and to better understand what people were reporting.

I first identified the top 10 most heavily reported events in the
database. Then, focusing on reports by those dates, I ran an
unsupervised learning algorithm to cluster the types of reports
into four categories. These approaches led me to take a
closer look at five specific dates. Lastly, I ran a sentiment
analysis on the language used in the reports.


### The Dataset
The National UFO Reporting Center is a non-profit organization which has been collecting reports of UFO sightings since the 1970s. While the vast majority of the reports were easily explained by human and natural events, a number of cases are investigated by the organization each year. The dataset contains historical and present day reports dating back to 1762. For this project I used a United States data subset of 95,638 reports (years 1947-2017).

### Most Reported Events
Perhaps not surprisingly, the dataset was skewed towards
more recent times, with a greater volume of reports in 2014-2017.
I chose to account for population growth by factoring
in yearly estimates of online users in the US. I also removed
‘approximate bucket dates,’ e.g. dates that were entered as
1st June, since the exact month, day and year were unknown
or estimated in the report, something that primarily impacted
reports made before 1995. After these corrections, the
following top 10 dates emerged as highly reported (1035
reports).

![top_events](/images/US_most_reported_events.png)

Notice that of the top 10, three dates have significantly higher popularity scores than the rest. Interestingly, six are in the
1990s and two are on the Fourth of July. Oddly, the Fourth
of July in 1997 was scored as around twice as popular as
the Fourth of July in 2014.

My question is, of these top 10 events are there any distinct themes in their descriptions?


### Topic Analysis - Unsupervised Learning

I wanted to take a deeper look at this dataset, to understand
whether there were any distinct themes. The reports typically have a detailed paragraph written by the witness which describes the event in detail, I decided to use basic natural language processing (TF-IDF) and use this to group similar reports together into sub-groups or topics. To capture topics, I used the unsupervised machine learning method of k-means clustering.

#### K-means Clustering on Text Explained...
Feel free to skip this part, This is an overview of how the algorithm works for text analysis.

This process creates a number of matrices:

*  A term-frequency matrix (TF) which counts the number of times a certain term (i.e. word or combination of words) is mentioned in a report.

* An inverse-document-frequency (IDF) matrix which is a (logarithmically scaled) inverse fraction to represent a term's frequency in the whole dataset (total number of documents in corpus divided by the number of documents containing the term).

* These two matrices are then multiplied to create a vector for each report.

Once we have our TF-IDF matrix of vectors, we can compare their similarity to each other by using k-means clustering. This algorithm randomly chooses k number of vectors to initialize the k centroids. Think of these centroids as the comparison reference vector. For each iteration the following happens:

1. Each report vector is compared to all the centroids by euclidian distance (straight line distance) and assigned (or re-assigned) to the closest centroid.

2. Once all vectors have been assigned to a centroid, the average vector or "middle" of the cluster is calculated from it's cluster members, this becomes the new centroid vector for comparison in the next iteration.

The two steps are repeated many times. At early iterations it is the centroids will be closer to each other, they will adapt and move until the best fit is found by the alorithm.


#### Visualization of K-means Clustering
The image below shows how k-means clustering works for 4 clusters.

![k-means_demo](https://camo.githubusercontent.com/9394c353adeeed261a0fd0588e2600f5e696433e/687474703a2f2f692e696d6775722e636f6d2f755a4b714b58692e676966)

Image reference: https://github.com/vinhkhuc/VanillaML

For this project, I used Scikit-learn's implementation which uses 300 iterations as the default, it runs the clustering model separate 10 times (with 10 different seeds) and chooses the best output based on inertia as the model.

### Clustering Method and Results

After removing common and custom stop words, I ran a text
vectorizer on the report text and applied TF-IDF to the
dataset. For this analysis, I ran a TF-IDF vectorizer using n-
grams ranging from a single word up to a string of 4 words, then reduced the dimensionality down to 12,000 features
using L1 regularization (LASSO). I chose to cluster the data into
4 groups, after testing 3 through 8. Even though 3 clusters would have been sufficient for this analysis (clusters seemed to be more distinct and non-repeating) I chose 4 so I can demonstrate the typical case of unintentional cluster behavior of the k-means algorithm.

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

 What most jumped out at me, though, looking at these word
 and phrase groupings, were the first two clusters. Cluster 1
 references the Fourth of July, while Cluster 2 heavily
 references a US Navy missile launch. I was curious to see if
 these clusters represented specific events, particularly
 regarding the missile launch, so I plotted the cluster groups
 that the k-means model assigned to each report.

![date_clusters](/images/k_means_clusters.png)

As you can see in the chart, the words used in Cluster
1 were mostly dated the 4th July 2014, rather than the 4th July 1997 identified in the Most Reported Events
analysis above. Additionally, the words used in Cluster 2
were almost entirely from reports dated 7th November 2015.
These two results may be indicative of class imbalance, where these two dates account for 52% of the reports which make up the top 10 data subset. However, It’s worth noting, that the language used by witnesses is similar enough for an unsupervised model to pick up on these distinct events, without training on or having access to report dates.

Cluster 3 can barely be seen on the chart, the majority of clusters are from 7th November 2015, however they do not reference the missile launch or blue/white lights, this may indicate a separate, less popular event that also occurred on this date.

Cluster 4 is spread out across all the dates, which makes
sense, given that the words in that cluster are mostly generic words that would score similar to each other given the nature of the text. It is s a prime example of an unintentional clustering outcome, where instead of a topical group, the algorithm has found reports which share similarity in language structure.


### Targeted Analysis

After doing the cluster analysis, I wanted to take a deeper
look. The Most Reported Events analysis left me curious
about three dates that had particularly high relative
popularity scores:

* 13th March 13 1997
* 4th July 1997
* 16th November 1999

while the Unsupervised Learning Analysis left me
curious about the events in Clusters 1 and 2:

* 4th July 2014
* 7th November 2015

For these dates, I followed a similar technique of TF-IDF above, to get a summary of popular n-grams (mostly 2 word bi-grams) for those events and represented a selection of results in a word-cloud (Please note: these are only a selection of words from the analysis). The more frequent a bi-gram appeared in the group, the larger the lettering. The larger the letters are compared to other dates the more consistency in the event descriptions.

I also checked those dates against news reports to see if the events were or could be explained.

#### Phoenix Lights - March 13th 1997

84% of the reports submitted to NUFORC for this date were from Arizona. The event was explained by the National Guard dropping flares from aircraft in addition to aircraft flying at high altitude in formation above Phoenix, AZ, although many eyewitnesses are skeptical of this explanation. Further reading: [wikipedia page](https://en.wikipedia.org/wiki/Phoenix_Lights)
![Phoenix_Lights](/images/phx-lights-word-cloud.png)

#### Midwest Fireball - November 16th 1999

This event was explained by an astronomical event, an
unusually bright fireball with a low altitude trajectory was
seen in the mid-western states. It preceded the Leonid meteor
shower, which is occurs annually. The event was reported across many states, 76% were from Iowa, Illinois, Indiana, Kentucky, Missouri, Ohio, Pennsylvania and Wisconsin.
[news report](https://science.nasa.gov/science-news/science-at-nasa/1999/ast17nov99_1)
![Fireball](/images/1999-fireball-word-cloud.png)


#### Strange Sightings in Los Angeles - November 7th 2015

This sighting was explained by a US Navy test missile which was launched off the Californian coast near Los Angeles. 80% of the reports received originated from California and Arizona.[news report](https://www.theguardian.com/us-news/2015/nov/08/navy-missile-launch-california-bright-light),
![LA_sighting](/images/blue-light-word-cloud.png)


#### Independence Day Sightings 1997 vs 2014

The intuitive explanation of any 4th July dates in the United States would be related to fireworks displays, however comparing the word clouds of the events on 4th July 1999 verses 4th July 2014, they both were distinct from each other.

###### 4th July 1997
![4th-jul-1997](/images/blue-green-st-louis.png)

###### 4th July 2014
![4th-jul-2014](/images/red-lights-4th-2014.png)

Although the terms 'fireworks' and '4th July' occur in both sets of data, it's interesting to see that the 1997 event had St Louis and St Charles mentioned with references to blue and green light. Of the reports submitted, 74% were from Missouri and Illinois.
This [local news article](http://www.bnd.com/living/magazine/article163418718.html) mentions the event, but it had no concrete explanation. Comparing the descriptions to the midwestern fireball of 1999 this may have been a meteor or another astronomical object.

The colors orange and red were most dominant for the 2014 event, multiple objects were also referenced (e.g 'formation', 'objects appeared', 'one another', 'three lights'). The sightings were observed in 38 States and had no obvious geographic pattern. I did not find any news reports for the sightings on Independence Day 2014, however a Google search returns this interesting [youtube video](https://www.youtube.com/watch?v=Zo-o9bDN0mo) which appears to show lights moving in formation. Further digging brought my attention to the popularity of personal drones around this time. It's possible drones were filming fireworks from above, capturing the attention of those watching fireworks displays across the country. As for multiple objects flying in formation? Researches submitted their work on [autonomous coordinated flocks of drones](https://motherboard.vice.com/en_us/article/nzebqz/drones-are-now-flying-in-flocks) earlier that year. Perhaps some enthusiasts were experimenting with coordinated flight that night?

### Measuring Descriptive Coherence
I normalized each output table of the bi-grams and measured their statistical variance of the top 50 to identify which of the events had a more coherent description (larger proportion of reports using the top 50 bi-grams). The larger the variance, the more coherence. In order of coherence they are:

* Independence Day 2014 : 0.045
* Strange Sightings in LA : 0.030
* Midwestern Fireball : 0.024
* Independence Day 1997 : 0.024
* Phoenix Lights : 0.020

There is not a large enough sample size to infer observations, but it's interesting to see the most coherent reports are from 4th July 2014, which incidentally has the least amount of news coverage and, other than my own theories, a lack of consensus of what these objects were.

### Sentiment Analysis on the US Dataset

Of the most reported events, I was surprised at how
coherent the reports were with each other, I decided to do a sentiment analysis on the text, to try to understand how emotive and
subjective the language is. For this task, I used a lightweight,
simple NLP library, TextBlob, to explore sentiment of the
dataset. I chose not to explore sentiment analysis fully
since the majority of sentiment was non-extreme or near
neutral for both emotion and subjectivity.

Of the total, only 31 had highly positive emotion and only 4
had extremely negative emotion. 20% had scores over 0.5 in
subjectivity, which indicates the corpus of reports are quite
objective in their descriptions.

### Additional Insights on the US Dataset

According to admin notes, The National UFO Reporting Center has annotated of third of all reports totaling 30,969 notes. And have attempted to contact a witness for 788 reports (0.82%)

265 (0.28%) reports contain hyperlinks to youtube.com footage

Out of all US reports, only 293 (0.31%) mention the word 'abducted'

![top_events](/images/US_cities_reports.png)

## Conclusions

A number of conclusions leap from this project. On the
technical end, k-means clustering is a great tool for exploring
latent sub-groups of data. It highlighted events, such as the
November 2015 US Navy missile launch and Independence day 2014 event, that I would not have considered investigating otherwise. Additionally, it helped me to understand the nature of the reports and how relatively coherent they were.

On the more topical side, the biggest thing that popped out
was that all five of the events I looked more closely at were
mostly explainable by natural or manmade phenomena.

Additionally, I came away feeling that the reports are, collectively, fairly reliable, at least so far as the data I looked at.
Not only is the language used to describe the events
remarkably coherent across reports, but the sentiment
analysis showed that the words in the reports were largely
objective and non-emotive. Lastly, if you recall some of the
statistics from NUFORC, discussed above, regarding
surprisingly low reporting of abductions and how many reports
they reach out to investigate and annotate, it makes it a bit easier to trust the dataset and appreciate the skeptical approach the organization collecting data.

## Further Questions and Things for the Future

There are many other angles you could take in approaching
this dataset. From what I learned in this project, other
questions worth investigating include:

* Whether the remaining five most reported events can be explained.
* How significant the spikes in reporting on the Fourth of
July are for each year.
* Whether the broader dataset matches this subset in
coherence.
* Whether reports about natural phenomena are more
coherent than reports about manmade phenomena.
