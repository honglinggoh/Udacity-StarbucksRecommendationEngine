# Udacity-StarbucksRecommendationEngine

### Table of Contents

  1. [Installation](#installation)
  2. [Project Motivation](#motivation)
  3. [Methodology](#methodology)
  4. [File Descriptions](#files)
  5. [Results](#results)
  6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no additional libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

This project is using the simulated data that mimics customers behavior on the Starbucks rewards mobile app to explore how can we improve the customers engagement & response towards the different type of offers promoted systematically.  Along this line, hopes that it can help to increase the customers purchases, improving Starbucks revenue and market segment.

I will explore the different recommendation engines to promote personalized offers for Starbucks broad customers base.  Primary focus are centralized around 3 use cases below:
1. Offer recommendation for new customer that we don't have any of their personalized info or purchasing behavior to cross reference
2. Offer recommendation for customer based on different customers segment
3. Offer recommendation for customer based on similar customer-offer interactions

## Methodology<a name="methodology"></a>

CRISP-DM (cross-industry process for data mining) methodology is used to help provide a structure approach to work on this project.

	1. Business Understanding: 
	First I tried to study what are the objectives of Starbucks Capstone Challenge project.  Based on that, I scoped down to one of the area that I would like to explore further how can we recommend a more personalized offer for different type of Starbuck customers.

	2. Data Understanding: 
	Then I start to explore and understand the different data sets provided on profile, portfolio and transcript.  I first analyzed the individual dataset separately.  Then I combined all data sets together to perform deep dive analysis to gain more insights  

	3. Data Preparation: 
	Post data analysis with better understanding on the informations available, I started to prepare the data for subsequent tasks executions.  Below are some of the data pre-processing applied:
    		a. Fill the NaN column with some other value as appropriate per the respective column data type and context.
    		b. Categorized the customers age and income into different categories range    
    		c. Standardize some columns name across the different data sets to ease of reference and processing like id to customer id or offer id.
    		d. Map the Offer unique id to offer label to ease of processing and reference
    		e. Label & One hot encoding
    		f. Data scaling and reformat
    		g. Feature Engineering
    		h. Data merging and etc

	4. Modeling: 
	Once we have all the data needed, we need to start explore the different recommendation algorithms to use for different use cases as stated above like Rank-Based Recommendation, KMeans clustering, Content-Based Recommendations & User-item Collaborative Filtering Based Recommendations.  Once the model is developed, then we have to assess the model.

	5. Evaluation: 
	Model or solution evaluation is one of the important step to validate the effectiveness of the model/solution in addressing the business problem statement as stated.  For recommendation engine, it is challenging to validate by nature using any metrics defined immediately.  Commonly, we will run A/B testing for the evaluation.  However it will take times and effort to setup for the testing. Hence as part of this project, I layed out the testing recommendation but not actual implementation.

	6. Deployment: 
	This is out of scope for this project.  Starbuck can run the evaluation and decide on the implementation plan later. 


## File Descriptions <a name="files"></a>

	1. The data is contained in three files.  Here is the schema and explanation of each variable in the files:

		a. portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
			• id (string) - offer id
			• offer_type (string) - type of offer ie BOGO, discount, informational
			• difficulty (int) - minimum required spend to complete an offer
			• reward (int) - reward given for completing an offer
			• duration (int) - time for offer to be open, in days
			• channels (list of strings)
		b. profile.json - demographic data for each customer
			• age (int) - age of the customer
			• became_member_on (int) - date when customer created an app account
			• gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
			• id (str) - customer id
			• income (float) - customer's income
		c. transcript.json - records for transactions, offers received, offers viewed, and offers completed
			• event (str) - record description (ie transaction, offer received, offer viewed, etc.)
			• person (str) - customer id
			• time (int) - time in hours since start of test. The data begins at time t=0
			• value - (dict of strings) - either an offer id or transaction amount depending on the record

	2. Starbucks_Capstone_notebook.ipynb is to showcase work related to the above.  I separated the notebook to multiple sections of exploratory analysis & modelling that lead to answering the questions above.  Markdown cells were used to assist in walking through the thought process for individual steps.  

	3. data_analysis_model.py - the scripted file contained of all the customs functions created for data preprocessing, data analysis, graph plotting and modellings.  This way it help to make the code clean and neat to maintain as well.


## Results<a name="results"></a>

The full report of this project can be found at the post available [here](https://medium.com/@hong.ling.goh/personalize-starbucks-offer-recommendation-engines-b82506595f67).


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

	1. Credits to Starbucks for publishing and sharing the simulated data as well as their business challenges to assist with this project.  
	2. Age Group Categorization references from https://v12data.com/blog/generational-consumer-shopping-trends/    
	3. There are few improvements can be done further for this project.  Hence, feel free to use the code here as you would like to strengthen the model to next level of excellence! 

