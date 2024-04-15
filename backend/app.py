import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
import re
import math
from collections.abc import Callable
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'linkedin_glassdoor_job_id.json')

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
merged_df = pd.read_json('linkedin_glassdoor_job_id.json', orient ='split', compression = 'infer')

# cmpy_reviews_docs_compressed_svd = pd.read_json('company_reviews_svd_docs.json', orient ='split', compression = 'infer')

# cmpy_reviews_terms_compressed_svd = pd.read_json('company_reviews_svd_docs.json', orient ='split', compression = 'infer')

# job_reviews_docs_compressed_svd = pd.read_json('job_reviews_svd_docs.json', orient ='split', compression = 'infer')

# job_reviews_terms_compressed_svd = pd.read_json('job_reviews_svd_words.json', orient ='split', compression = 'infer')



app = Flask(__name__)
CORS(app)
def executeQuerySearch(personalValues_query, personalExperience_query):
    def tokenize(text: str) -> List[str]:
        """
        Parameters
        ----------
        text: str
            The input text you'd like to tokenize

        Returns
        -------
        List[str]
            A list of tokens based on the inputted text
        """
        tokens = re.findall(r"\b[a-zA-Z]+\b", text)
        tokens = [token.lower() for token in tokens]

        return tokens

    def tokenize_df_fields(df: pd.DataFrame, fields_to_tokenize: List[str],
                        tokenize_method: Callable[[str], List[str]]) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            The input df that includes the fields to tokenize.

        fields_to_tokenize
            fields that you want to be tokenized

        tokenize_method:
            the method to tokenize

        Returns
        -------
        pd.DataFrame
            A new df that includes additional fields containing the tokenized original fields.
        """
        for field in fields_to_tokenize:
            if field in df.columns:
                df[field + '_tokens'] = df[field].fillna('').apply(tokenize_method)
            else:
                print(f"Field '{field}' not found in DataFrame. Skipping.")
        return df

    # set the fields you'd like to tokenize
    fields = ['job_description', 'company_description', 'job_industry']

    # create df with token fields for each element in fields list
    tokenized_df = tokenize_df_fields(merged_df, fields, tokenize)

    # The k for SVD
    k = 50
    

    def build_inverted_index(df: pd.DataFrame, token_column_name: str) -> dict:
        """
        Parameters
        ----------
        df: pd.DataFrame
            The input df that includes the tokenized fields.

        token_column_name: str
            The name of the tokenized column

        Returns
        -------
        inverted_index: dict
            An inverted index.
            Keys are the tokens.
            Values of inverted_index are a sorted list of tuples. First value in tuple is
            the original job_id (not df index). Second value in tuple is count of term in field.
        """

        inverted_index = {}

        job_ids = df.job_id.to_numpy()
        tokens_list = df[token_column_name].tolist()

        tokens_dict = dict(zip(job_ids, tokens_list))

        # for each job:
        for job_id, tokens in tokens_dict.items():
            token_counts = {}

            # iterate through the tokens and get counts
            for token in tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1

            # iterate through the token count dictionary
            for token, count in token_counts.items():
                if token in inverted_index:
                    inverted_index[token].append((job_id, count))
                else:
                    inverted_index[token] = [(job_id, count)]

        # sort list
        for token, job_dict in inverted_index.items():
            inverted_index[token] = sorted(job_dict, key=lambda x: x[1], reverse=True)

        return inverted_index

    # create inverted indexes
    job_description_inverted_index = build_inverted_index(tokenized_df, 'job_description_tokens')
    job_industry_inverted_index = build_inverted_index(tokenized_df, 'job_industry_tokens')

    def compute_idf(inv_idx, n_docs):
        """Compute term IDF values from the inverted index.

        inv_idx: an inverted index
        n_docs: int,
            The number of documents.

        Returns
        =======

        idf: dict
            For each term, the dict contains the idf value.

        """
        ans = {}
        for key in inv_idx:
            doc_list = inv_idx[key]
            val = math.log2((n_docs/(1+len(doc_list))))
            ans[key] = val
        return ans

    def compute_doc_norms(tokenized_df, inv_idx, idf, n_docs):
        """Precompute the euclidean norm of each document.

        inv_idx: the inverted index as above
        idf: dict,
            Precomputed idf values for the terms.
        n_docs: int,
            The total number of documents.

        Returns:
        norms: dict
            norms[i] = the norm of document i.
        """
        # TODO: Fix this up to be more formal in the future
        norms = {num: 0 for num in range(60000)}
        #norms = np.zeros(n_docs) # the issue here is that the keys are job_id but this will go out of bounds for number of jobs
        # pandas .index
        # tokenized_df[jobs_list['job_id'] == tup[0]].index
        for key in inv_idx:
            if idf[key] > 12:
                doc_list = inv_idx[key]
                for tup in doc_list:
                    if key in idf:
                        index = tokenized_df[tokenized_df['job_id'] == tup[0]].index[0]
                        if index in norms:
                            norms[index] = norms[index] + (tup[1]*idf[key])**2
                        else:
                            norms[index] = (tup[1]*idf[key])**2

        for key, value in norms.items():
            norms[key] = np.sqrt(value)
        #norms = np.sqrt(norms)
        return norms

    def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
        # print(cmpy_reviews_terms_compressed_svd.shape)
        """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

        Arguments
        =========

        query_word_counts: dict,
            A dictionary containing all words that appear in the query;
            Each word is mapped to a count of how many times it appears in the query.
            In other words, query_word_counts[w] = the term frequency of w in the query.
            You may safely assume all words in the dict have been already lowercased.

        index: the inverted index as above,

        idf: dict,
            Precomputed idf values for the terms.
        doc_scores: dict
            Dictionary mapping from doc ID to the final accumulated score for that doc
        """
        # TODO-7.1
        ans = {}
        for key in query_word_counts:
            if key in idf:
                doc_list = index[key]
                for (doc, tf) in doc_list:
                    if doc in ans:
                        val = ans[doc]
                        val = val + idf[key]*idf[key]*tf*query_word_counts[key]
                        ans[doc] = val
                    else:
                        ans[doc] = idf[key]*idf[key]*tf*query_word_counts[key]
        return ans
    

    # Returns top 3 matches for query
    def testing_SVD(query):
        # Tokenize query
        job_descriptions = merged_df['job_description'].astype(str).tolist()

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([query] + job_descriptions)
        query_vector = X[0].toarray()

        # Perform SVD
        svd = TruncatedSVD(n_components=2)  # Reduce to 2 dimensions for example
        svd.fit(X)
        U = svd.transform(X)
        Vt = svd.components_

        # Transform query using SVD
        query_transformed = np.dot(query_vector, Vt.T)

        # Calculate cosine similarity with each job description
        similarities = cosine_similarity(query_transformed, U[1:])

        top_matches_idx = np.argsort(similarities[0])[::-1][:3]

        # Get top 3 most similar job descriptions and their similarity scores
        top_matches = [(job_descriptions[idx], similarities[0][idx]) for idx in top_matches_idx]
        indices_with_relation_to_merged_df = [idx - 1 for idx in top_matches_idx]  # Subtract 1 for the query index

        # Get rows from merged_df corresponding to the top matches
        top_matches_df = merged_df.iloc[indices_with_relation_to_merged_df]
        # Create an empty dictionary to store formatted results
        descrips_top_matches = []

        # Iterate through each row in top_matches_df
        for index, row in top_matches_df.iterrows():
            company = row['company_name']
            # similarity_score = row['similarity_score']
            
            # Get relevant information from merged_df based on company name
            industry = merged_df.loc[merged_df['company_name'] == company, 'company_industry'].values[0]
            description = merged_df.loc[merged_df['company_name'] == company, 'company_description'].values[0]
            headline = merged_df.loc[merged_df['company_name'] == company, 'headline'].values[0]
            
            # Store the formatted information as a tuple and append to the list
            descrips_top_matches.append((company, industry, description, headline))




        print(descrips_top_matches)
        return descrips_top_matches

    
    def index_search(tokenized_df,
        query: str,
        index: dict,
        idf,
        doc_norms,
        score_func,
        tokenizer
    ,
    ) -> List[Tuple[int, int]]:
        """Search the collection of documents for the given query

        Arguments
        =========

        query: string,
            The query we are looking for.

        index: an inverted index as above

        idf: idf values precomputed as above

        doc_norms: document norms as computed above

        score_func: function,
            A function that computes the numerator term of cosine similarity (the dot product) for all documents.
            Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
            (See Q7)

        tokenizer: a TreebankWordTokenizer

        Returns
        =======

        results, list of tuples (score, doc_id)
            Sorted list of results such that the first element has
            the highest score, and `doc_id` points to the document
            with the highest score.

        Note:

        """
        # TODO-8.1
        ans = []
        query_word_counts = {}

        query = query.lower()
        query_tokens = tokenizer.tokenize(query)

        # q_p = np.dot(query_tokens.T, terms_compressed_svd)
        # svd(merged_df, query_tokens)
        svdOutputs = testing_SVD(query)



        for word in query_tokens:
            if word in query_word_counts:
                count = query_word_counts[word]
                query_word_counts[word] = count + 1
            else:
                query_word_counts[word] = 1

        # print(get_q_from_word_counts(query_word_counts))

        scores = score_func(query_word_counts, index, idf)

        query_norm = 0
        for key in query_word_counts:
            if key in idf:
                val = query_word_counts[key]
                query_norm = query_norm + (idf[key]*val)**2
        query_norm = np.sqrt(query_norm)

        for doc in scores:
            score = scores[doc] # id -> score, doc norms is index
            index = tokenized_df[tokenized_df['job_id'] == doc].index[0]
            if query_norm*doc_norms[index]>0:
                ans.append((scores[doc]/(query_norm*doc_norms[index]), doc))

        ans.sort(key = lambda x: x[0], reverse = True)
        return ans, svdOutputs


    job_idf = compute_idf(job_description_inverted_index, len(tokenized_df))

    job_description_inverted_index = {key: val for key, val in job_description_inverted_index.items() if key in job_idf}  # prune the terms left out by idf
   
    job_doc_norms = compute_doc_norms(tokenized_df, job_description_inverted_index, job_idf, len(tokenized_df)) # currently only checking if idf is > 14.5 bc there are 155270 total terms, this lets it finish in 2 min
    
    industry_results, svdOutputs = index_search(tokenized_df, personalExperience_query, job_description_inverted_index, job_idf, job_doc_norms, accumulate_dot_scores, TreebankWordTokenizer())

    job_set = set()
    top3_industries = []
    count = 0


    for tup in industry_results:
        if count >= 3:
            break
        id = tup[1]
        if (tokenized_df.loc[tokenized_df['job_id'] == id, 'job_industry'].values[0]) not in job_set:
            top3_industries.append(tokenized_df.loc[tokenized_df['job_id'] == id, 'job_industry'].values[0])
            job_set.add(tokenized_df.loc[tokenized_df['job_id'] == id, 'job_industry'].values[0])
            count +=1

    new_company_df = tokenized_df.drop(tokenized_df[~tokenized_df['company_industry'].isin(top3_industries)].index, inplace=False)

    company_description_inverted_index = build_inverted_index(new_company_df, 'company_description_tokens')

    company_idf = compute_idf(company_description_inverted_index, len(tokenized_df))

    company_description_inverted_index = {key: val for key, val in company_description_inverted_index.items() if key in company_idf}

    company_doc_norms = compute_doc_norms(new_company_df, company_description_inverted_index, company_idf, len(tokenized_df))



    company_results, svdOutputs_2 = index_search(new_company_df, personalValues_query, company_description_inverted_index, company_idf, company_doc_norms, accumulate_dot_scores, TreebankWordTokenizer())



    #Gets top 3 companies
    company_set = set()
    company_list = []
    count = 0
    index = 0

    while count < min(len(company_results), 3):
        id = company_results[index][1]
        if tokenized_df.loc[tokenized_df['job_id'] == id, 'company_name'].values[0] not in company_set:
            company_list.append(tokenized_df.loc[tokenized_df['job_id'] == id, 'company_name'].values[0])
            count+=1
            company_set.add(tokenized_df.loc[tokenized_df['job_id'] == id, 'company_name'].values[0])
        index+=1

    # gets top company descriptions for top 3 companies
    # Dict that goes from top company name to company's industry and that company's description
    #  TODO: Add review as a 3rd part of this dict
    descrips = {company: (tokenized_df.loc[tokenized_df['company_name'] == company, 'company_industry'].values[0], tokenized_df.loc[tokenized_df['company_name'] == company, 'company_description'].values[0], tokenized_df.loc[tokenized_df['company_name'] == company, 'headline'].values[0]) for company in company_list}
    
    def get_random_reviews(reviews_df: pd.DataFrame, industry: str, num_reviews: int = 3) -> pd.DataFrame:

        # filter reviews by industry
        filtered_reviews = reviews_df.loc[
        (reviews_df['company_industry'] == industry) & 
        (reviews_df['headline'].notna())
        ][['company_name', 'headline']].drop_duplicates(subset=['headline'], keep='first')

        # if no reviews for industry, return empty df
        if filtered_reviews.empty:
            return []
        
        if len(filtered_reviews) <= num_reviews:
            return filtered_reviews.apply(lambda x: [x['company_name'], x['headline']], axis=1).tolist()
        
        # randomly select reviews and return
        random_reviews = filtered_reviews.sample(n=num_reviews)
        return random_reviews.apply(lambda x: [x['company_name'], x['headline']], axis=1).tolist()
    
    # get 3 reviews for each industry, returns dict with list of 3 review headlines
    industry_reviews = {}
    for industry in top3_industries:
        reviews = get_random_reviews(tokenized_df, industry)
        industry_reviews[industry] = reviews
    

# for company in company_list:
#   company_description = tokenized_df.loc[tokenized_df['company_name'] == company, 'company_description'].values
#   descrips.append(company_description[0]) #I AM NOT SURE WHICH INDEX OF THIS LIST WE WANT TO PRINT-- I CANT PRINT RN SO CAN'T SEE WHAT THIS GIVES US


# Total Output to send back to front end
#  industry_list, descrips, industry_reviews
    return [top3_industries, descrips, industry_reviews, svdOutputs]

# Sample search using json with pandas
def json_search(personalValues, personalSkills):
    rslts = executeQuerySearch(personalValues, personalSkills)
    return rslts

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/searchPaths")
def episodes_search():
    personalSkills = request.args.get("personalSkills")
    personalValues = request.args.get("personalValues")
    print(personalSkills, personalValues)
    return json_search(personalValues, personalSkills)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)