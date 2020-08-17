# Udacity-StarbucksRecommendationEngine

### Table of Contents

  1. [Installation](#installation)
  2. [Project Motivation](#motivation)
  3. [File Descriptions](#files)
  4. [Results](#results)
  5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no additional libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

This project will use the simulated data that mimics customer behavior on the Starbucks rewards mobile app to develop a machine learning engine for product offer recommendation. Will explore the different recommendations methodology that can help to improve the customer engagements and purchases volumes via more personalize customer product offering.

There are 3 questions that I was looking focusing on around bookings cancellation.

  1. What is the cancellation rate over the years for different hotel categories?
  2. Are we able to predict booking cancellation based on data set provided?  
  3. For those booking at risk for cancellation, what can the hotel do to mitigate the risk?


## File Descriptions <a name="files"></a>

  1. 
  Data Sets
The data is contained in three files:

portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers completed
Here is the schema and explanation of each variable in the files:

"hotel_bookings.csv" is the data file that I downloaded from Kaggle which consists of City and Resort Hotels (from Europe) booking transactions from year 2015-2017.

Hotel Booking.ipynb notebook is to showcase work related to the above questions.  I separated the notebook to multiple sections of exploratory analysis that lead to answering the questions above.  Markdown cells were used to assist in walking through the thought process for individual steps.  

  2. 

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@hong.ling.goh/random-forest-for-predicting-city-hotel-booking-cancellation-17222fe479b).


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to Jesse Mostipak for publishing and sharing the data.  Licensing for the data and other descriptive information is accessible via  the Kaggle link available [here](https://www.kaggle.com/jessemostipak/hotel-booking-demand).  Otherwise, feel free to use the code here as you would like! 

