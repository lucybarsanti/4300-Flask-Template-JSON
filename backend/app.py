from collections import defaultdict
import json
import os
from flask import Flask, render_template, request, jsonify
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
json_file_path = os.path.join(current_directory, 'linkedin_glassdoor_final.json')

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
merged_df = pd.read_json('linkedin_glassdoor_final.json', orient ='split', compression = 'infer')

# cmpy_reviews_docs_compressed_svd = pd.read_json('company_reviews_svd_docs.json', orient ='split', compression = 'infer')

# cmpy_reviews_terms_compressed_svd = pd.read_json('company_reviews_svd_docs.json', orient ='split', compression = 'infer')

# job_reviews_docs_compressed_svd = pd.read_json('job_reviews_svd_docs.json', orient ='split', compression = 'infer')

# job_reviews_terms_compressed_svd = pd.read_json('job_reviews_svd_words.json', orient ='split', compression = 'infer')



app = Flask(__name__)
CORS(app)
def executeQuerySearch(personalValues_query, personalExperience_query, num_return_industries=5, num_return_companies_per_industry=3):
    try:
    

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
        fields = ['job_industry']

        # create df with token fields for each element in fields list
        tokenized_df = tokenize_df_fields(merged_df, fields, tokenize)


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

    
        # Returns top n matches for query in the form (company, industry, description, simScore)[] or (industry, simScore)[]
        # Based on the col_title provided
        def SVD_search_jobs(query, col_title, name_col, n = 3, k=50):
            # Tokenize query
            # print("STARTING SVD")
            description = merged_df[col_title].astype(str).tolist()
            # print(merged_df)
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([query] + description)
            query_vector = X[0].toarray()
            # Perform SVD
            svd = TruncatedSVD(n_components=k)  # Reduce to k dimensions for example
            svd.fit(X)
            U = svd.transform(X)
            Vt = svd.components_

            # Transform query using SVD
            query_transformed = np.dot(query_vector, Vt.T)

            # Calculate cosine similarity with each description
            similarities = cosine_similarity(query_transformed, U[1:])

            top_matches_idx = np.argsort(similarities[0])[::-1][:n]

            # Get top 3 most similar descriptions and their similarity scores
            top_matches = [(description[idx], similarities[0][idx]) for idx in top_matches_idx]
            indices_with_relation_to_merged_df = [idx - 1 for idx in top_matches_idx]  # Subtract 1 for the query index

            # Get rows from merged_df corresponding to the top matches
            top_matches_df = merged_df.iloc[indices_with_relation_to_merged_df]
            # Create an empty dictionary to store formatted results
            descrips_top_matches = []
            indx = 0
            # Iterate through each row in top_matches_df
            for index, row in top_matches_df.iterrows():
                company_or_industry = row[name_col]
                # Retrieve the similarity score for this particular match
                # print("HERE ARE TOP MATCHES LENGTH", len(top_matches),index)
                simScore = top_matches[indx][1]
                # similarity_score = row['similarity_score']
                # Get relevant information from merged_df based on company name
                if(col_title=='company_description'):
                    industry = merged_df.loc[merged_df[name_col] == company_or_industry, 'company_industry'].values[0]
                    description = merged_df.loc[merged_df[name_col] == company_or_industry, 'company_description'].values[0]
                    # Store the formatted information as a tuple and append to the list
                    descrips_top_matches.append((company_or_industry, industry, description, simScore))
                else:
                    # Store the formatted information as a tuple and append to the list
                    descrips_top_matches.append((company_or_industry,  simScore))
                
                
                indx+=1





            # print(descrips_top_matches)
            return descrips_top_matches

        
           # print("COMPUTE IDF")
        industry_results = SVD_search_jobs(personalExperience_query, 'job_industry_tokens', 'job_industry', 3000)
        industry_results = list(set(industry_results))


        company_results = SVD_search_jobs(personalValues_query, 'company_description',  'company_name', 1000)
        company_results = list(set(company_results))
       
        # print("industry Results", industry_results)
        # print('\n')
        # print('company results', company_results)

        # Idea now is we have scores for industry and company.
        # We want to find the maximum overlap of the two. We can group companies into industry categories
        # We then take the top 3 companies of each industry scores and add it with the score of that industry (with the scaling of a weight for how much we should value each)
        # We then take the top 3 industry+company sim score groups.
        companies_grouped_by_industry_results = defaultdict(list)
        for company, industry, description, simScore in company_results:
            companies_grouped_by_industry_results[industry].append((company, description, simScore))

        # Sort companies within each industry group by simScore
        for industry, companies in companies_grouped_by_industry_results.items():
            companies_grouped_by_industry_results[industry] = sorted(companies, key=lambda x: x[2], reverse=True)


        # # Convert defaultdict to regular dictionary if needed
        # companies_grouped_by_industry_results = dict(companies_grouped_by_industry_results)

        combinedScores = []
        company_weight = 1
        industry_weight = 1
        # Print the grouped results
        for  industry, industry_simScore in industry_results:
            companies = companies_grouped_by_industry_results.get(industry)
            if companies:
                # Take the average of top num_return_companies_per_industry companies' simScores
                top_companies = companies[:num_return_companies_per_industry] if len(companies) >= num_return_companies_per_industry else companies
                avg_company_score = sum(company[2] for company in top_companies) / len(top_companies)
                # Calculate weighted score
                weighted_score = (avg_company_score * company_weight + industry_simScore * industry_weight)/(company_weight+industry_weight)
                combinedScores.append((industry, weighted_score, top_companies))
        combinedScores.sort(key=lambda x: x[1], reverse=True)

        # Step 1, comb through company_results, group into industry (taking only the top 3 score companies for each industry)

        
        # gets top company descriptions for top num_return_companies_per_industry companies
        # Dict that goes from top company name to company's industry and that company's description
        #  TODO: Add review as a 3rd part of this dict
        # descrips = {company: (tokenized_df.loc[tokenized_df['company_name'] == company, 'company_industry'].values[0], tokenized_df.loc[tokenized_df['company_name'] == company, 'company_description'].values[0], tokenized_df.loc[tokenized_df['company_name'] == company, 'headline'].values[0]) for company in company_list}
        
        def get_random_reviews(reviews_df: pd.DataFrame, industry: str, num_reviews: int = 3) -> pd.DataFrame:

            # filter reviews by industry
            filtered_reviews = reviews_df.loc[
            (reviews_df['company_industry'] == industry) & 
            (reviews_df['headline'].notna())
            ][['company_name', 'headline', 'overall_rating']].drop_duplicates(subset=['headline'], keep='first')

            # if no reviews for industry, return empty df
            if filtered_reviews.empty:
                return []
            
            if len(filtered_reviews) <= num_reviews:
                return filtered_reviews.apply(lambda x: [x['company_name'], x['headline'], x['overall_rating']], axis=1).tolist()
            
            # randomly select reviews and return
            random_reviews = filtered_reviews.sample(n=num_reviews)
            return random_reviews.apply(lambda x: [x['company_name'], x['headline'], x['overall_rating']], axis=1).tolist()
        
        # get 3 reviews for each industry, returns dict with list of 3 review headlines and overall rating
        industry_reviews = {}
        for industry, _, _ in combinedScores[:num_return_industries]:
            reviews = get_random_reviews(tokenized_df, industry)
            industry_reviews[industry] = reviews



    # Total Output to send back to front end
    #  industry_list, descrips, industry_reviews
    # scores is of type (industry, weighted_score, top_companies)[]
    # top_companies is of type (company, description, simScore)[]
        return {'scores': combinedScores[:num_return_industries], 'reviews': industry_reviews}

    except Exception as e:
        # Log the exception or handle it in an appropriate way
        print("An error occurred during query search:", e)
        # Return an error message or empty results
        raise e


# Sample search using json with pandas
def json_search(personalValues, personalSkills):
    try:
        rslts = executeQuerySearch(personalValues, personalSkills)
        return rslts
    except Exception as e:
        # Log the exception or handle it in an appropriate way
        print("An error occurred during query search:", e)
        # Return an error message or empty results
        raise e

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/searchPaths")
def episodes_search():
    try:
        personalSkills = request.args.get("personalSkills")
        personalValues = request.args.get("personalValues")
        # print(personalSkills, personalValues)
        return json_search(personalValues, personalSkills)
    except Exception as e:
        # Log the exception or handle it in an appropriate way
        print("An error occurred during search:", e)
        # Return an error response to the client
        return jsonify(error=str(e)), 500

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)