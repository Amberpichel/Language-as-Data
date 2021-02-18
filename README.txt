README
Authors: Merel de Groot, Amber Pichel
Research question: How are the differences in attitude on the topic of 'gay marriage' expressed in tweets between the languages English and Dutch?
Link to the blogpost: https://wordpress.com/post/languageasdata.wordpress.com/252

get_all_documents.py:
This will provide you with tweets about gay marriage in both Dutch and English. Please change the API-KEY, API-secret, Access-token, Access-secret before running the code.
When running from the commandline an outdirectory can be specified, if not the results will be put in a tweets-folder in the current directory. 
Be aware that running this code without change will overwrite the older results. Furthermore, because of the workings of the API the current set timestamps are ignored.

evaluate_annotations.py:
Calculates the agreement percentage, cohen's kappa, and confusion matrix and prints the results and writes it to a txt file in the directory of the annotation sheets.

run_all_analyses.py:
Runs the general statistics of the users and tweets, sentiment analysis, babelnet word queries, term frequency plots and prints them in the command line, writes them to a txt file, and saves plots in the directories.

Additional information:
Models should be put in the models file