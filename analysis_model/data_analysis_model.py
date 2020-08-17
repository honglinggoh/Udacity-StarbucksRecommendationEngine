#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import basic libraries
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt

#import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score

#import clustering libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import stats

def offer_mapper(df, source='offer_id', target='offer_label'):
    """ 
    Input:
    - df - input data frame with offer_id to map
    Output:
    - df - updated offer_id to simpler identifier
    """
    df = df.copy()
    # map the offer ids to identifier
    offer_dict = {'ae264e3637204a6fb9bb56bc8210ddfd': 'B1',
                '4d5c57ea9a6940dd891ad53e9dbe8da0': 'B2',
                '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'B3',
                'f19421c1d4aa40978ebb69ca19b0e20d': 'B4',
                '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'D1',
                '2298d6c36e964ae4a3e7e9706d1fb8c2': 'D2',
                'fafdcd668e3743c1bb461111dcafc2a4': 'D3',
                '2906b810c7d4411798c6938adc9daaa5': 'D4',
                '3f207df678b143eea3cee63160fa8bed': 'I1',
                '5a8bc65990b245e5a138643cd4eb9837': 'I2'}
    df[target] = df[source].apply(lambda x: offer_dict[x] if x else None)
    return df

def offer_mapper_reverse(df, source='offer_label', target='offer_id'):
    """ 
    Input:
    - df - input data frame with offer_id to reverse mapping
    Output:
    - df - updated offer_id to original identifier
    """
    df = df.copy()
    # map the identifier to offer_id
    offer_dict = {'B1': 'ae264e3637204a6fb9bb56bc8210ddfd',
                'B2': '4d5c57ea9a6940dd891ad53e9dbe8da0',
                'B3': '9b98b8c7a33c4b65b9aebfe6a799e6d9',
                'B4': 'f19421c1d4aa40978ebb69ca19b0e20d',
                'D1': '0b1e1539f2cc45b7b9fa7c272da2e1d7',
                'D2': '2298d6c36e964ae4a3e7e9706d1fb8c2',
                'D3': 'fafdcd668e3743c1bb461111dcafc2a4',
                'D4': '2906b810c7d4411798c6938adc9daaa5',
                'I1': '3f207df678b143eea3cee63160fa8bed',
                'I2': '5a8bc65990b245e5a138643cd4eb9837'}
    df[target] = df[source].apply(lambda x: offer_dict[x] if x else None)
    return df

def clean_portfolio(portfolio):
    '''
    INPUT 
        portfolio - raw portfolio dataframe
    OUTPUT
        df_portfolio_clean - Returns the processed portfolio dataframe after data cleansing
    '''
    # check the # of rows and columns
    #portfolio.shape

    # Create dummy columns for the channels column
    df_channel = pd.get_dummies(portfolio['channels'].apply(pd.Series).stack(),
                                prefix="channel").sum(level=0)

    # concat the dummy columns with the portfolio data frame
    df_portfolio_clean = pd.concat([portfolio, df_channel], axis=1)

    # drop the channels column which is no longer needed with the dummy columns creation
    df_portfolio_clean.drop(columns='channels', inplace=True)

    # rename the column to make it more identifiable
    df_portfolio_clean.rename(columns={'id':'offer_id'}, inplace=True)
    
    # map the offer_id to simpler identifier to ease the processing later
    df_portfolio_clean = offer_mapper(df_portfolio_clean, source='offer_id', target='offer_label')

    return df_portfolio_clean


def clean_profile(profile):
    '''
    INPUT 
        profile - raw profile dataframe
    OUTPUT
        df_profile_clean - Returns the processed profile dataframe after data cleansing
    '''
    # check the # of rows and columns
    #profile.shape
    
    # Analyze the NaN value in the profile dataset.
    # only gender & income columns has NaN value
    profile.isna().sum()
    
    # make a copy of the profile data frame
    df_profile_clean = profile.copy()
    
    # Fillna for NaN value in gender and income columns
    df_profile_clean.gender.fillna(value='N', inplace=True)
    df_profile_clean.income.fillna(value=0, inplace=True)

    # rename the column to make it more identifiable
    df_profile_clean.rename(columns={'id':'customer_id'}, inplace=True)   
    
    # change the became_member_on to valid date format
    df_profile_clean['became_member_on'] =pd.to_datetime(df_profile_clean['became_member_on'], format='%Y%m%d')

    # base on the dataset, seems like whenever age=118, gender & income data will not be available at the same time
    # hence, we can create a new column to separate customer with & without the complete profile information
    df_profile_clean['complete_profile'] = (df_profile_clean['gender'] != 'N').astype(int)
    
    # map the gender to int category
    dict_gender = {
        'N': 0,
        'F': 1,
        'M': 2,
        'O': 3 
    }
    df_profile_clean['gender_cat']=df_profile_clean['gender'].apply(lambda x: dict_gender[x])
    
    # Age Group Categorization - from https://v12data.com/blog/generational-consumer-shopping-trends/
    # 1 - GenZ (younger than 25)
    # 2 - Millennials (25-35)
    # 3 - GenX (36-54)
    # 4 - Boomers (55-75)
    # 5 - Silents(76+)
    # 99 - Invalid age
    age_ul = [15, 25, 35, 54, 75, 105, 120]
    age_cat = ['1','2','3','4','5','0']
    df_profile_clean['age_cat']=pd.cut(df_profile_clean['age'].values, age_ul , labels = age_cat)
    df_profile_clean['age_cat']=df_profile_clean['age_cat'].astype(int)

    # Check on income summary stats
    df_profile_clean['income'].describe()

    # Salary Categorization - range set based on the income summary stats above
    # 1 - 29,001 - 45,000
    # 2 - 45,001 - 60,000
    # 3 - 60,001 - 75,000
    # 4 - 75,001 - 90,000
    # 5 - 90,001 - 105,000
    # 6 - 105,001 and above
    income_ul = [-1, 29000, 45000, 60000, 75000, 90000, 105000, 125000]
    income_cat = ['0','1','2','3','4','5','6']
    df_profile_clean['income_cat']=pd.cut(df_profile_clean['income'].values, income_ul , labels = income_cat)
    df_profile_clean['income_cat']=df_profile_clean['income_cat'].astype(int)
    
    return df_profile_clean


def clean_transcript(transcript):
    '''
    INPUT 
        transcript - raw transcript dataframe
    OUTPUT
        df_transcript_clean - Returns the processed transcript dataframe after data cleansing
    '''

    # check the # of rows and columns
    #transcript.shape   
    
    # Analyze the NaN value in the transcript dataset.
    # No Nan Value found.
    transcript.isna().sum()
    
    # make a copy of the transcript data frame
    df_transcript_clean = transcript.copy()

    # rename the column to make it more identifiable
    df_transcript_clean.rename(columns={'person':'customer_id'}, inplace=True)  

    # prepare the event name for get_dummies function
    df_transcript_clean.event.unique()
    df_transcript_clean['event'] = df_transcript_clean.event.str.replace(' ','_')
    
    # Create dummy columns for the event column
    df_event = pd.get_dummies(df_transcript_clean['event'], prefix="event")

    # concat the dummy columns with the transcript data frame
    df_transcript_clean = pd.concat([df_transcript_clean, df_event], axis=1)

    # drop the event column which is no longer needed with the dummy columns creation
    df_transcript_clean.drop(columns='event', inplace=True)

    # create new column = offer_id to store the offer id value
    df_transcript_clean['offer_id'] = [[*v.values()][0]
                                    if [*v.keys()][0] in ['offer id','offer_id'] else None
                                    for v in df_transcript_clean.value]

    # create new column = amount to store the amount value
    df_transcript_clean['amount'] = [np.round([*v.values()][0], decimals=2)
                                    if [*v.keys()][0] == 'amount' else None
                                    for v in df_transcript_clean.value]

    # drop the value column which is no longer needed with the offer_id and amount columns creation
    df_transcript_clean.drop(columns='value', inplace=True)

    return df_transcript_clean

def plt_data(df=None, figsize=(14,4), subplot=111, kind='bar', ylabel='', xlabel='', title='', plt_type=None, bins=10             , color='tab:blue', tableTF=True, gridTF=True, new_fig=True):
    '''
    INPUT 
        df - data frame for the graph plot
        figsize - graph figure size
        subplot - subplot number/position
        kind - the type of graph to plot (ex: bar, line)
        ylabel - y axis label to display
        xlabel - x axis label to display
        plt_type - whether it is hist or normal plot
        bins - for hist, # of bins to display
        color - graph plot color
        tableTF - to display data table - True or False
        gridTF - to display graph grid - True or False
    OUTPUT
        display plotted graph
    '''
    
    if df is not None:
        if new_fig:
            plt.figure(figsize=figsize)
            
        if subplot != 111:
            plt.subplot(subplot)
            
        if plt_type is None:
            df.plot(kind=kind, rot=0, figsize=figsize, color=color, table=tableTF)
        else:
            if plt_type == 'hist':
                plt.hist(df, bins)
            else:
                return None
                
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(gridTF)

def get_top_offer_ids(n=3, df=None, offer_key='viewed'):
    '''
    INPUT:
    n - (int) the number of top offers id to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    offer_key - either it is received, viewed or completed
   
    OUTPUT:
    top_offer_ids - (list) A list of the top 'n' offer ids 
    
    '''
    # retrieve a list of n top offer ids
    if offer_key in ('received','viewed','completed'):
        offer_trans = 'cnt_' + offer_key
    else:
        return None
    
    top_offer_ids = list(df.groupby(['offer_label']).sum()[offer_trans].sort_values(ascending=False)[:n,].keys())
 
    return top_offer_ids # Return the top offer ids


def get_top_offers(customer_id=None, n=1, df=None, dfpf=None, dfmcot=None, dfmco=None, offer_key='viewed'):
    '''
    INPUT:
    n - (int) the number of top offers to return
    df - portfolio dataframe
    dfpf - profile data frame
    dfmcot - all data sets merged data frame
    dfmco - offer transaction data frame
    offer_key - either it is received, viewed or completed
    
    OUTPUT:
    top_offers - A dataframe of the top 'n' offers details either based on Rank-Based or Collaborative-Filtering Recommendations
    
    '''
    
    if (customer_id != None):
        # if customer id is provided, we will recommend based on collaborative filtering instead
        top_offer_ids = get_top_offer_ids_customer(customer_id=customer_id, n=n, dfpf=dfpf, dfmcot=dfmcot, dfmco=dfmco, offer_key=offer_key)
        if (top_offer_ids == None):
            # if no recommendation return due to customer_id not existed before, will use rank-based recommendation
            top_offer_ids = get_top_offer_ids(n=n, df=dfmco, offer_key=offer_key)
    else:
        # if it is new customer, will use rank-based recommendation
        top_offer_ids = get_top_offer_ids(n=n, df=dfmco, offer_key=offer_key)
            
    # retrieve the n top offer as a list object
    top_offers = df[df['offer_label'].isin(top_offer_ids)]
    
    return top_offers # Return the top offers from portfolio df

def create_cluster(dfmcot=None):
    '''
    INPUT
    dfmcot - the summarized offers and purchase activities by customer 

    OUTPUT
    cluster_list - the list of clusters for customers 
    '''
    
    df_cluster = dfmcot.copy()
    # remove the non numerics columns from the data frame
    df_cluster.drop(['customer_id','gender'], axis=1, inplace=True)
    
    #scale the data using standard scaler
    scale = StandardScaler()
    scale.fit(df_cluster[['sum_amount','avg_purchase']])
    data = scale.transform(df_cluster[['sum_amount','avg_purchase']])
    df_scaled = df_cluster.copy()
    df_scaled['sum_amount'] = data[:, 0]
    df_scaled['avg_purchase'] = data[:, 1]
    
    # use LabelEncoder to encode age_cat and income_cat from categorical value to int
    le_number = LabelEncoder()
    df_scaled['age_cat'] = le_number.fit_transform(df_scaled['age_cat'])
    df_scaled['income_cat'] = le_number.fit_transform(df_scaled['income_cat'])

    #fit KMeans for every k, append the SSE score to a list (scores)
    scores = []
    k_list = list(range(1,25))
    for k in k_list:
        kmeans = KMeans(k)
        model = kmeans.fit(df_scaled)
        scores.append(np.abs(model.score(df_scaled)))

    #plot the scores to find the elbow
    plt.figure(figsize = (10,8))
    plt.plot(k_list, scores, linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Score');
    plt.xticks(k_list)
    plt.title('SSE vs. K');

    #predicting clusters for each users based on the optimum k which is in this case I choose 11
    k = KMeans(11)
    cluster_list = k.fit_predict(df_scaled)

    #changing values in cluster_list so it starts from 1 
    cluster_list = cluster_list + 1
    
    return cluster_list    

def get_top_offer_ids_customer (customer_id=None, n=3, dfpf=None, dfmcot=None, dfmco=None, offer_key='viewed'):
        '''
        INPUT
        customer_id - customer id to recommend offer for
        n - top n rank
        dfpf - the clean profile dataframe
        dfmcot - the summarized offers and purchase activities by customer 
        dfmco - the summarized offers activities by customer 
        offer_key = could be either received, viewed, completed
    
        OUTPUT
        ranked_offers 
            - if customer profile exists:
                - look for other customers within same cluster
                - then look for all offer ids for the customers sorted by:
                    highest average purchase, # of purchase transactions, count of offer viewed & completed 
                - return the offers list 
        '''

        # check if customer_id provided
        if (customer_id == None):
            print('Customer id none')
            return None
        
        # retrieve a list of n top offer ids
        if offer_key in ('received','viewed','completed'):
            offer_trans = 'cnt_' + offer_key
        else:
            print('Offer key none')
            return None        
 

        # pull the list of customers within the same cluster if exists
        cluster_id = dfmcot[['cluster']][dfmcot['customer_id'] == customer_id].values
        
        if len(cluster_id) > 0:
            ls_sim_customers = dfmcot.loc[dfmcot.cluster == int(cluster_id),'customer_id'].values.tolist()

            if len(ls_sim_customers) > 0:
                # remove the customer_id from the similar customer list
                ls_sim_customers = list(filter((customer_id).__ne__, ls_sim_customers))

                df_sim_customers = dfmcot.loc[dfmcot['customer_id'].isin(ls_sim_customers)]

                 # sort by top avg purchase, cnt transacts(purchase), cnt viewed, cnt completed
                df_sim_customers= df_sim_customers.sort_values(by=['avg_purchase', 'cnt_transaction', 'cnt_viewed', 'cnt_completed'],                                             ascending=False)       

                k = int(df_sim_customers.shape[0]*(5/100))
                # pull the list of offers for this top 5% customers
                df_sim_cust_offer = dfmco.loc[dfmco['customer_id'].isin(list(df_sim_customers.customer_id[:k])), ['offer_label', offer_trans]]                                                   .groupby(['offer_label']).sum().sort_values(by=offer_trans, ascending=False)


                # return the top n offer ids 
                return list(df_sim_cust_offer.index[:n])
            else:
                return None
        else:
            return None

def fit(dfo=None, offer_key='viewed', latent_features=12, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization

    INPUT:
    dfo - the customer offer data frame
    offer_key = could be either received, viewed, completed
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate
    iters - (int) the number of iterations

    OUTPUT:
    user_mat - user matrix
    offer_mat - offer matrix
    user_ids_series - series of user_ids
    offer_labels_series - series of offer_labels
    '''
    
    if offer_key in ('received','viewed','completed'):
        offer_trans = 'cnt_' + offer_key
    else:
        print('Invalid offer key')
        return None, None, None, None
    
    # Create user-item matrix
    offer_type = 'cnt_' + offer_key
    usr_itm = dfo[['customer_id', 'offer_label', offer_type]]
    user_item_df = usr_itm.groupby(['customer_id','offer_label'])[offer_type].max().unstack()
    user_item_mat= np.array(user_item_df)

    # parameters
    latent_features = latent_features
    learning_rate = learning_rate
    iters = iters

    # Set up useful values to be used through the rest of the function
    n_users = user_item_mat.shape[0]
    n_offers = user_item_mat.shape[1]
    num_offer_resp = np.count_nonzero(~np.isnan(user_item_mat))
    
    user_ids_series = np.array(user_item_df.index)
    offer_labels_series = np.array(user_item_df.columns)

    # initialize the user and offer matrices with random values
    user_mat = np.random.rand(n_users, latent_features)
    offer_mat = np.random.rand(latent_features, n_offers)

    # initialize sse at 0 for first iteration
    sse_accum = 0

    # keep track of iteration and MSE
    print("Optimizaiton Statistics")
    print("Iterations | Mean Squared Error ")

    # for each iteration
    for iteration in range(iters):

        # update our sse
        old_sse = sse_accum
        sse_accum = 0

        # For each user-offer pair
        for i in range(n_users):
            for j in range(n_offers):

                # if the offers count exists
                if user_item_mat[i, j] > 0:

                    # compute the error as the actual minus the dot product of the user and movie latent features
                    diff = user_item_mat[i, j] - np.dot(user_mat[i, :], offer_mat[:, j])

                    # Keep track of the sum of squared errors for the matrix
                    sse_accum += diff**2

                    # update the values in each matrix in the direction of the gradient
                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (2*diff*offer_mat[k, j])
                        offer_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])
                        
        # print results
        print("%d \t\t %f" % (iteration+1, sse_accum / num_offer_resp))

    # SVD based fit
    return user_mat, offer_mat, user_ids_series, offer_labels_series

def predict_offer(customer_id, offer_label, uids=None, oids=None,                   user_mat=None, offer_mat=None):
    '''
    INPUT:
    customer_id - the customer_id from the profile data frame 
    offer_label - the offer_label from the portfolio data frame 

    OUTPUT:
    pred - the predicted rating for customer_id-offer_label according to FunkSVD
    '''
    try:# customer row and offer Column
        customer_row = np.where(uids == customer_id)[0][0]
        offer_col = np.where(oids == offer_label)[0][0]

        # Take dot product of that row and column in U and V to make prediction
        pred = math.floor(np.dot(user_mat[customer_row, :], offer_mat[:, offer_col]))

        print("For customer {} we predict {} transactions for the offer {}.".format(customer_id, round(pred), str(offer_label)))

        return pred

    except:
        print("I'm sorry, but a prediction cannot be made for this customer-offer pair.  It looks like one of these items does not exist in our current database.")

        return None


def make_recommendations(customer_id=None, rec_num=1, uids=None, oids=None,                          user_mat=None, offer_mat=None, dfmco=None):
    '''
    INPUT:
    rec_num - number of recommendations to return (int)

    OUTPUT:
    recs - (array) a list or numpy array of recommended offer for a customer_id given
    '''
    # if the customer id is available from the matrix factorization data,
    # will use this and # of offer responses based on the predicted values
    offer_labels = None
    if customer_id != None:
        if customer_id in uids:
            # Get the index of which row the user is in for use in U matrix
            idx = np.where(uids == customer_id)[0][0]

            # take the dot product of that row and the V matrix
            preds = np.dot(user_mat[idx,:],offer_mat)

            # pull the top offers according to the prediction
            indices = preds.argsort()[-rec_num:][::-1] #indices
            offer_labels = oids[indices]
        else:
            # if we don't have this user, give just top ratings back
            offer_labels = get_top_offers(n=rec_num, dfmco=dfmco)

    else:
        offer_labels = get_top_offers(n=rec_num, dfmco=dfmco)
    return offer_labels


# In[ ]:




