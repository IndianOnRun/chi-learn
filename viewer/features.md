# Features We Want to Develop

## Our Target Audience

Because Chicago's data portal withholds the last seven days' worth of crime data, our predictions have no hope of informing the decisions of real Chicagoans. We could do lots of cool historical summaries, but that's outside the scope of this project.

Our audience is technical people who are curious about how different machine learning approaches fare on this kind of problem.  

## Features

### Overview

+ We want to let users examine the performance of our system by neighboorhood and by timespan. (e.g. How well did algorithm X perform in Edgewater over the last month?).
+ We also want to post a prediction for day x + 1 as soon as day x's data becomes available.

### Day To Day Prediction

+ There will be a page where a user can see what we predict for each neighborhood on day x + 1 once all of day x's data is available. 
+ There will be a page where the user can see how predictions for day x fared against real data once all of day x's data is available.
 
### Historical Performance

+ There will be pages where the user can see how each algorithm has performed over the last week, month, and year. 
+ There will be a page where the user can see how each algorithm has performed through history (e.g. from Jan 1 2005 to present).

### Visualization

+ Each page with prediction data will have a chloropleth of Chicago's community areas. The page will show a summary by default. The user can examine data for a particular community area by selecting it.

## Ideas on Implementation

It isn't proper to specify implementation details in a requirements doc, but whatever man, down with the patriarchy.

### Generating Predictions Each Day

+ Each day, a scheduled task will grab the new day's crimes from the data portal. We'll either need to figure out the time that data is released or poll Chicago's API. I haven't looked into it much, but Celery seems appropriate for this.
+ We'll keep an enormous pickled mapping of neighborhoods to pandas DataFrames on disk. Unlike the current way we generate our DataFrame, we might want to normalize the DF by adding an extra mapping to pan-Chicago data rather than append that to each neighborhood.
+ When we grab the new data, we'll append it to the existing DataFrame, pickle it for posterity, and then slice it up to the forms that will be required for the sequential and non-sequential algorithms. We might condsider using a "dynamic" baseline that goes on the average over the last month instead of historically.
+ Then we'll train two classifiers (sequential and non-sequential) on the data. We'll make three predictions per neighborhood: one with the sequential classifier, one with the non-sequential classifier, and one with the baseline.

### Storing The Predictions

+ I suggest we store a document (read: big-ass nested hash) for each day from the "start of history" (say, Jan 1 2005) onwards. Each document will look kinda like this:

```
March 30, 2015
  Rogers Park: 
    Predictions: 
      Sequential:
        Classification: Crime
        Probability: .67
      NonSequential: Crime
        Classification: Crime
        Probability: .73
      Baseline:
        Classification: No Crime
        Probability: .49
    Outcome: Crime
  West Ridge:
    ...
```

+ Brace yourself Joel, but I suggest we use MongoDB to store documents. It's not a performance-based decision. It just makes sense to me to store hashes and Mongo does that. The Python driver lets you feed in dicts and get dicts out.
+ In addition to a document for each day, we'll have a handful of summary documents (e.g. one for the last 7 days, one for the last 30 days, one for the last 365 days, and one for all of history).
+ The most recent document will only have predictions, no outcomes.

### Making a Website

+ We'll use Flask for the web interface.
+ We'll use D3.js for visualizations.
+ A single Jinja2 template should be able to cover most of our needs for the map pages.
+ Maybe we'll have a text-based page or 2 (like a FAQ). If we want to give really technical explanations of things, we can link to an iPython notebook.
+ Because we're both pretty shabby at frontend, let's just find a good-enough Bootstrap template for styling.

### Rendering Predictions

+ Any map-based page we render should be backed by a document in our database. There should be no processing based on user requests. All processing will be done at scheduled times and the website will just be a view onto pre-processed documents.
+ Could this be a static site??? Like we could host the backend on Digital Ocean or something and every day it pushes to a Github pages branch after doing the processing. Maybe we could cut out Flask altogether.