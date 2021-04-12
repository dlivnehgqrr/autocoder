#@title Load the Universal Sentence Encoder's TF Hub module
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import string
from textblob import TextBlob
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from fuzzywuzzy import fuzz
import re
import nltk
from datetime import date
from statistics import mean
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def main():
	
	# Get filenames and column names from the input user
	response_file = input("Please enter filename of the csv with the responses. For example 'responses.csv': ")
	response_column_names = input("Please enter the name of columns (separated by columns) with the responses that you wish to run the program on. For example 'O_DESCRIBE, O_DISLIKE, O_HAPPY': ")
	columns = response_column_names.split(",")
	id_column = input("Please enter nameo of the column of the unique response ID. For example 'RESPONDENT_ID': ")
	response = input("Please make sure you have codeframes saved in this directory. They should be named [COLUMN_NAME_OF_RESPONSES]_codeframe.csv, and there must be a separate codeframe for every column you entered. Once confirmed, press enter ")
	

	# where the results will be save [YYYMMDD_responsefile_output]
	todays_date = date.today().strftime("%Y%m%d")
	output_folder = todays_date + "_" + response_file.split(".csv")[0] + "_output"

	# Make the output directory
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)


	print("Loading Tensorflow please hold!")
	# load tensor flow, this takes about 2 minutes
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" # options are: ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
	model = hub.load(module_url)
	print ("Done! Module %s loaded" % module_url)

	# define function to get the tensor flow embeddings
	def embed(input):
		return model(input)


	# Do the thing!
	for c in columns:
	    print("NOW RUNNING: " + c)
	    # The code assumes a codeframe file in this directory called "[columnnameofverbtaims]_codeframe.csv"
	    codeframe_file = c + "_codeframe.csv"
	    column = c
	    # output file will be named YYYYMMDD_[columnnameofverbtaims]_coded.csv
	    output_file = output_folder + "_" + todays_date + "_" + c + "_coded.csv"
	    
	    df = pd.read_csv(response_file, encoding = "ISO-8859-1")
	    df = df.replace(np.nan, '', regex=True)
	    df = df.astype('str')
	    responses = df[column].tolist()
	    df_response = df[column]
	    
	    respondent_id = df[id_column]
	    
	    
	    code_frame_df = pd.read_csv(codeframe_file, header = None)
	    code_frame_df.index += 1
	    code_frame = code_frame_df.iloc[:,0].tolist()
	    split_code_frame = [c.split(",") for c in code_frame]
	    strip_and_split_code_frame = []
	    for code_list in split_code_frame:
	        strip = [c.strip() for c in code_list]
	        strip.append("".join(code_list))
	        strip_and_split_code_frame.append(strip)
	    code_frame_embeddings = []
	    for cf in strip_and_split_code_frame:
	        code_frame_embeddings.append(embed(cf))
	    
	    dk_na = ["Nothing", "I don't know"]
	    exclusive_code_frame_options = embed(dk_na)
	    
	    
	    count = 0
	    results = []
	    for response in responses:
	        count = count + 1
	        print(count)
	        best_responses = []
	        best_responses_scores = []
	        
	        # want to split up the responses into clauses....to make it easier to split
	        # we replace all punctuation that indicates "clause" with a period so we can
	        # just split on period later
	        response = response.replace(') ', '.')
	        response = response.replace(',', '.')
	        response = response.replace(';', '.')
	        response = response.replace('! ', '.')
	        response = response.replace('...', '.')
	        response = response.replace('- ', '.')
	        response = response.replace('/', '.')
	        response = response.replace('&', '.')
	        response = response.replace(' and ', '.')
	        
	        for res in response.split("."):
	            res = res.strip()
	            response_embeddings = embed([res])
	            if (not res) | res.isspace():
	                continue
	            else:
	                max_scores = []
	                # loop through the codeframe embeddings
	                for i in range(0, len(code_frame)):
	                    code_frame_scores = []
	                    # get the embedding for the ith codeframe
	                    code_frame_embedding = code_frame_embeddings[i]
	                    # get the cosine similiary between the response emedding and every codeframe emedding 
	                    # cosines is a list of the cosine similatiries where the index corresponds to the
	                    # the codeframe indexes
	                    # so cosines[0] is cosine similiatry of respnose and code_frame[0]
	                    cosines = tf.keras.losses.cosine_similarity(code_frame_embedding, response_embeddings,axis=-1).numpy().tolist()
	                    
	                    # I'm waiting the cosine score and the fuzzy token sort scores here...honestly
	                    # it's a little arbitrary i chose 130 and .70 aka putting more weight on the cosine than the fuzzy sort ratio
	                    scores = [(c * -130 +  fuzz.token_sort_ratio(code_frame[i], res) * .70) for c in cosines]
	                    
	                    # take the max score append to the list
	                    max_scores.append(max(scores))
	                
	                # get the best score's index. -1 could be changed to -2 or whatever 
	                # to get the top n scores
	                max_best_scores = np.argsort(max_scores)[-1:].tolist()
	                
	                # loop through top n scores (in the case its just the one score)
	                for best_score in max_best_scores:
	                    # basically only take the top if it's greater than 75...again
	                    # an arbitrary threshold for what's considered "good enough"
	                    if max_scores[best_score] > 75:
	                        # since code_frame is not zero indexes need to add one
	                        code_frame_number = best_score + 1
	                        # check to make sure we don't already have that codeframe variable
	                        if code_frame_number not in best_responses:
	                            best_responses.append(code_frame_number)
	                            best_responses_scores.append(max_scores[best_score])
	        
	        # take top four best responses
	        top_best_responses = np.argsort(best_responses_scores)[-4:].tolist()
	        response_result = [best_responses[i] for i in top_best_responses]
	        
	        # basically if the response ins't blank but we couldn't find any matches with the codeframe
	        if not response_result and not(not response):
	            # well try with the exclusive codeframe options (Which in this case is "none" and "I don't know"
	            cosines = tf.keras.losses.cosine_similarity(exclusive_code_frame_options, response_embeddings,axis=-1).numpy().tolist()
	            # weighting is a little different here since the chance of it being an exact word match are more likely
	            scores = [(c * -122 +  fuzz.token_sort_ratio(dk_na[cosines.index(c)], response)) for c in cosines]
	            max_exclusive_score = max(scores)
	            #if none of the exclusive options produce a sufficient score, we use "97" which stands for "other"
	            if max_exclusive_score < 70:
	                response_result = [97]
	            else: 
	                # otherwise its 98 (none) or 99 (I don't know)
	                # none is index at 0 and i don't know is indexed 1 so I just add 98 
	                response_result = [scores.index(max_exclusive_score) + 98]
	        results.append(response_result)
	    
	    # create a dataframe of the responses to the results
	    d = {'Response':  responses,'Code': results}
	    df_results = pd.DataFrame(d)
	    output_columns = []
	    # create columns for the codes
	    for i in range(1, max([len(x) for x in results]) + 1):
	        output_columns.append("Code" + str(i))
	    new_df = pd.DataFrame(df_results["Code"].to_list(), columns=output_columns)
	    new_df['Response'] = df_results.Response
	    # add respondent ID 
	    new_df.insert(0, 'RESPONDENT_ID', respondent_id)
	    # output csv
	    new_df.to_csv(output_file)
	    print("DONE WITH: " + c + " Saved to: "  + output_file + "\n\n")     
        
	print ("COMPLETE!! All results saved to: " + output_folder)

if __name__ == "__main__":
	main()