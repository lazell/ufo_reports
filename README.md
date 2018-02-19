## What the UFO?
##### Exploratory analysis with clustering and NLP of UFO reports for the celestially curious


![celestial](/images/ryan-lange-552065.jpg)

### The Dataset
The National UFO Reporting Center is a non-profit organization which has been collecting reports of UFO sightings since the 1970s. While the vast majority of the reports were easily explained by human and natural events, a number of cases are investigated by the organization each year. The dataset contains 13,600+ historical and present day reports dating back to 1762.


### Topic Analysis

The reports typically have a detailed paragraph written by the witness which describes the event. I ran a text vectorizer on the report text after removing common stop words and applied TF-IDF to a United States data subset of X,0000 (years 1970-2017). To capture topics (type or category of event), I used k-means clustering.


### Most Reported Events
Roswell aside, I was curious to see the most reported events. Since the highest reports by date were skewed to the years 2014-2017. I chose to account for population growth by factoring in yearly estimates of online users in the United States and removed 'approximate bucket dates' e.g dates which were entered as 1st June since the exact day was unknown in the report. This mainly impacted data prior 1995. After doing so, the following top 10 dates emerged as highly reported (1591 reports).

![top_events](/images/US_most_reported_events.png)

The consensus descriptions varied wildly between events. Notice that of the top 10 there were 2 years in which 4th of July were highly reported. Let's take a look at report content using Natural Language Processing and cluster them into latent groups.

The reports typically have a detailed paragraph written by the witness which describes the event. After removing common and domain specific stop words from the corpus, I ran a TF-IDF vectorizer using n-grams ranging from a single word up to a string of 4 words and reduced the dimensionality to 3000 features with L1 regularization (lasso). I then fitted the reports to a K-means clustering model, which uses the vectorization to group similar report vectors together.

Here are the top 50 n-gram results when I ran K-means with 4 Clusters:

#### Cluster 1 - 4th July fireball, fireworks and flashing lights of various colors

<p style='color:#97B6C2; font-family:courier;font-size:90%;'>
['looking' 'show' 'quickly' 'round' 'clouds' 'behind' 'wife' 'circular'
 'watched' 'across sky' 'minute' 'fireball' 'still' '30' 'approximately'
 'line' 'night' 'slow' 'west east' 'speed' 'glowing' 'fast' 'direction'
 'watching' 'green' 'seemed' 'later' '10' 'across' 'fly' '4th' 'noticed'
 'july' 'sound' 'flying' 'moved' 'two' 'traveling' 'ball' 'slowly' 'high'
 'flashing' 'formation' 'minutes' 'disappeared' 'north' 'objects'
 'fireworks' 'red' 'orange']
 </p>


#### Cluster 2 - Admin notes, blue white light and US Navy missile launch references

<p style='color:#97B6C2; font-family:courier;font-size:90%;'>
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
   </p>

#### Cluster 3 - Moving objects in formation of various colors
<p style='color:#97B6C2; font-family:courier;font-size:90%;'>
 'aircraft' 'shaped' 'witnessed' 'around' 'seemed' 'behind' 'slowly'
 'sound' 'direction' 'fire' 'meteor' 'red' 'area' 'said' 'three' 'fast'
 'noticed' 'flying' 'wife' 'seconds' 'moved' 'orange' 'ufo' 'craft'
 'thought' 'night' 'speed' 'large' 'tail' 'shape' 'traveling' 'fireball'
 'across' 'green' 'horizon' 'summary' 'formation' 'blue' 'fireworks'
 'north' 'two' 'objects']
    </p>

#### Cluster 4 - Mostly descriptive words beginning with letters: L, F, and G!

['longer' 'long' 'little' 'line' 'light sky' 'length' 'left' 'look' 'end'
 'heard' 'headed' 'flew' 'flashing' 'first thought' 'fireworks' 'firework'
 'fireball' 'fire' 'flight' 'feet' 'far' 'faded' 'extremely' 'ever' 'event'
 'evening' 'even' 'fast' 'fly' 'flying' 'followed' 'head' 'happened' 'half'
 'ground' 'green' 'got' 'gone' 'going' 'yellow' 'go' 'glowing' 'glow' 'get'
 'front' 'friend' 'formation' 'following' 'heading' 'information']

The two clusters that jump out at me are the first two which relate to 4th July - independence day and the US Navy missile launch with blue lights. I was curious to see if the latter cluster represented a specific event, so I plotted the cluster groups the k-means model assigned each report.  

![date_clusters](/images/k_means_clusters.png)

Interestingly reports which were classified as cluster 1 were mostly 4th July 2014 and cluster 2 were almost entirely November 7th 2015. Not surprisingly, cluster 4 is spread out across all dates since they are mostly generic descriptive words.

For a handful of dates, I followed a similar technique of TF-IDF above, to get a summary of popular n-grams (mostly bi-grams) for those events and represented a selection of results in a word-cloud (Please note: these are only a selection of words from the analysis). The more frequent a bi-gram appeared in the group, the larger the lettering. The larger the letters are compared to other dates the more consistency in the event descriptions.

#### Midwest Fireball - November 16th 1999

![Fireball](/images/1999-fireball-word-cloud.png)
An unusually bright fireball with a low altitude trajectory was seen from the midwestern states on this evening. It preceded the Leonids meter shower, which is an annual celestial event:
[news report](https://science.nasa.gov/science-news/science-at-nasa/1999/ast17nov99_1)


#### Phoenix Lights - March 13th 1997

![Phoenix_Lights](/images/phx-lights-word-cloud.png)
The event was explained by the National Guard dropping flares from aircraft and aircraft flying at high altitude in formation, although this is regarded as a controversial explanation for many eye-witnesses. Further reading: [wikipedia page](https://en.wikipedia.org/wiki/Phoenix_Lights)


#### Strange Sightings in Los Angeles - November 7th 2015
![LA_sighting](/images/blue-light-word-cloud.png)
This sighting was explained by a US Navy test missile which was launched off the Californian coast near LA: [news report](https://www.theguardian.com/us-news/2015/nov/08/navy-missile-launch-california-bright-light)

### Additional Insights

According to admin notes, The National UFO Reporting Center has contacted a witness for 797 reports (0.82%)

272 of reports contain links to youtube footage (0.28%)

0.12% of all reports mention the word 'abducted'

![top_events](/images/US_cities_reports.png)
