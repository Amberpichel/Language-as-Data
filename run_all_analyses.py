from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import stanza
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.manifold import TSNE
import requests
from gensim.models import KeyedVectors
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from polyglot.text import Text
import seaborn as sns
import networkx as nx

#Add Babelnet key here
babelnet_key = "0c8982b7-1e89-4955-b111-493cd009af33"
wordurl = "https://babelnet.io/v5/getSynsetIds?"
synseturl = "https://babelnet.io/v5/getSynset?"
relations_url = 'https://babelnet.io/v5/getOutgoingEdges?'

def get_general_tweet_stats(dir, language):
	stanza.download(language)
	twitter_content = pd.read_csv(f'{dir}/gaymarriage_tweets.tsv', delimiter='\t', encoding='utf-8')
	twitter_content['Created at'] = pd.to_datetime(twitter_content['Created at'])
	with open(f'{dir}/tweet_statistics.txt', 'w', encoding='utf8') as outfile:
		user_stats(twitter_content,outfile)
		tweet_stats(twitter_content, language,outfile)

def user_stats(twitter_content, outfile):
	"""
	Produces all the statistics about all the twitter data
	:param twitter_content: dataframe containing the twitter content
	:param outfile: outfile the results will be written to
	:type twitter_content: dataframe
	:type outfile: file
	"""
	locs=twitter_content['User Location'].fillna("Unknown")
	#Print all statistics
	print("Number of unique users:" + str(twitter_content['User'].nunique()))
	print("Number of verified accounts:" + str(twitter_content['Verified'].value_counts('True')))
	print("Languages present in dataset: " + str(twitter_content["Language"].unique()))
	print()
	print("Most frequent user locations:" + str(sorted(Counter(locs).items(), key=lambda item: item[0])))
	print()
	print("Earliest timestamp:" + str(twitter_content['Created at'].min()))
	print("Latest timestamp:" + str(twitter_content['Created at'].max()))
	print("Timespan collection:" + str(twitter_content['Created at'].max() - twitter_content['Created at'].min()))
	print()
	print(twitter_content.describe())
	#write all statistic to the txt file
	outfile.write("Number of unique users:" + str(twitter_content['User'].nunique()))
	outfile.write("\nNumber of verified accounts:" + str(twitter_content['Verified'].value_counts()))
	outfile.write("\nLanguages present in dataset: " + str(twitter_content["Language"].unique()))
	outfile.write("\n\nMost frequent user locations:" + str(sorted(Counter(locs).items(), key=lambda item: item[0])))
	outfile.write("\n\nEarliest timestamp:" + str(twitter_content['Created at'].min()))
	outfile.write("\nLatest timestamp:" + str(twitter_content['Created at'].max()))
	outfile.write("\nTimespan collection:" + str(twitter_content['Created at'].max() - twitter_content['Created at'].min()))

def calculate_stats(twitter_content, language):
	"""
	Calculates the general statics of the twitter data, such as pos-tag frequencies, token frequencies, max sentence length, etc.
	:param twitter_content: dataframe containing the the tweets
	:param language: indicates the language used
	:type twitter_content: dataframe
	:type language: string
	:return: 4 counter objects with frequencies and 5 ints with sentence data
	"""
	nlp = stanza.Pipeline(language)
	#initialize variables
	token_without_frequencies = Counter()
	token_frequencies = Counter()
	upos_frequencies = Counter()
	ner_frequencies = Counter()
	num_sentences = 0
	max_sentence = 0
	min_sentence = 1000
	max_tweet = 0
	min_tweet = 1000
	#get the right stopwords
	if language == 'en':
		stop_lan='english'
	elif language == 'nl':
		stop_lan='dutch'
	else:
		raise NotImplementedError

	stop_and_punct = stopwords.words(stop_lan)
	for i in string.punctuation:
		stop_and_punct.append(i)
	for i in range(len(twitter_content['Text'])):
		current_article = twitter_content['Text'][i]
		# Skip empty articles
		if current_article != '':
			# Process the article with the stanza pipeline
			processed_article = nlp(current_article)
			sentences = processed_article.sentences
			tokens_per_tweet = 0

			# Iterate through all sentences of the article
			for sentence in sentences:
				num_sentences += 1
				#should remove stopwords and punctuation form the string
				all_tokens_without = [token.text.lower() for token in sentence.tokens if
									  token.text.lower() not in stop_and_punct]
				all_tokens = [token.text.lower() for token in sentence.tokens]
				tokens_per_tweet += len(all_tokens)
				if len(all_tokens) > max_sentence:
					max_sentence = len(all_tokens)
				if len(all_tokens) < min_sentence:
					min_sentence = len(all_tokens)
				all_upos = [word.pos for word in sentence.words]
				all_ner = [token.ner for token in sentence.tokens]
				token_frequencies.update(all_tokens)
				token_without_frequencies.update(all_tokens_without)
				upos_frequencies.update(all_upos)
				ner_frequencies.update(all_ner)
			# Add the tokens to a counter
			if tokens_per_tweet > max_tweet:
				max_tweet = tokens_per_tweet
			if tokens_per_tweet < min_tweet:
				min_tweet = tokens_per_tweet
	return 	token_without_frequencies, token_frequencies, upos_frequencies, ner_frequencies, num_sentences, max_sentence, min_sentence, max_tweet, min_tweet

def tweet_stats(twitter_content, language, outfile):
	"""
	Collects the general statistics of the tweets, prints them, writes them the the outfile, and makes plots of the ner and pos frequencies
	:param twitter_content: dataframe containing the twitter data
	:param language: indicates the language that is being used
	:param outfile: outfile to write the results in
	:type twitter_content: dataframe
	:type language: string
	:type outfile: txt file
	"""
	token_without_frequencies, token_frequencies, upos_frequencies, ner_frequencies, num_sentences, max_sentence, min_sentence, max_tweet, min_tweet = calculate_stats(twitter_content, language)
	print("Number of types:" + str(len(token_frequencies.keys())))
	print("Number of tokens:" + str(sum(token_frequencies.values())))
	print("Type/token ratio:" + str((len(token_frequencies.keys()) / sum(token_frequencies.values()))))
	print()
	print("Average number of tokens per sentence:" + str(sum(token_frequencies.values()) / num_sentences))
	print("Highest number of tokens in a sentence:" + str(max_sentence))
	print("Lowest number of tokens in a sentence:" + str(min_sentence))
	print()
	print("Average number of tokens per tweet:" + str(sum(token_frequencies.values()) / len(twitter_content['Text'])))
	print("Highest number of tokens in a tweet:" + str(max_tweet))
	print("Lowest number of tokens in a tweet:" + str(min_tweet))
	print()
	print("Number of types without stopwords and punctuation:" + str(len(token_without_frequencies.keys())))
	print("Number of tokens without stopwords and punctuation:" + str(sum(token_without_frequencies.values())))
	print("Type/token ratio without stopwords and punctuation:" + str(
		(len(token_without_frequencies.keys()) / sum(token_frequencies.values()))))
	print("50 most common tokens without stopwords and punctuation:" + str(token_without_frequencies.most_common(50)))
	print()
	print("Most common pos-tags:" + str(upos_frequencies.most_common()))
	print()
	print("Most common named entity tags:" + str(ner_frequencies.most_common()))

	#writes the results to the outfile
	outfile.write("\n\nNumber of types:" + str(len(token_frequencies.keys())))
	outfile.write("\nNumber of tokens:" + str(sum(token_frequencies.values())))
	outfile.write("\nType/token ratio:" + str((len(token_frequencies.keys()) / sum(token_frequencies.values()))))
	outfile.write("\n\nAverage number of tokens per sentence:" + str(sum(token_frequencies.values()) / num_sentences))
	outfile.write("\nHighest number of tokens in a sentence:" + str(max_sentence))
	outfile.write("\nLowest number of tokens in a sentence:" + str(min_sentence))
	outfile.write("\n\nAverage number of tokens per tweet:" + str(sum(token_frequencies.values()) / len(twitter_content['Text'])))
	outfile.write("\nHighest number of tokens in a tweet:" + str(max_tweet))
	outfile.write("\nLowest number of tokens in a tweet:" + str(min_tweet))
	outfile.write("\n\nNumber of types without stopwords and punctuation:" + str(len(token_without_frequencies.keys())))
	outfile.write("\nNumber of tokens without stopwords and punctuation:" + str(sum(token_without_frequencies.values())))
	outfile.write("\nType/token ratio without stopwords and punctuation:" + str(
		(len(token_without_frequencies.keys()) / sum(token_frequencies.values()))))
	outfile.write("\n50 most common tokens without stopwords and punctuation:" + str(token_without_frequencies.most_common(50)))
	outfile.write("\n\nMost common pos-tags:" + str(upos_frequencies.most_common()))
	outfile.write("\nMost common named entity tags:" + str(ner_frequencies.most_common()))
	plot_general_freqs(upos_frequencies, 'Part of Speech', language)
	plot_general_freqs(ner_frequencies, 'Named Entity', language)

def plot_general_freqs(freqs, type_tags, language):
	"""
	Makes a plot of the distributions of the frequencies
	:param freqs: Counter object containing the frequencies
	:param type_tags: type of analysis done
	:param language: which language is being used
	:type freqs: Counter
	:type type_tags: string
	:type language: string
	"""
	D = dict(freqs)
	if 'O' in D.keys():
		D.pop('O')
	plt.figure(figsize=(20, 10))
	plt.bar(range(len(D)), list(D.values()), align='center')
	plt.xticks(range(len(D)), list(D.keys()))
	plt.title(f"Distribution of {type_tags} Tags")
	plt.xlabel("Tags")
	plt.ylabel("Ocurrances")
	plt.savefig(f"Dist_{type_tags}_{language}.jpg")
	plt.show()


def annotation_stats_and_analysis(dir, language):
	"""
	Performs all the analysis steps for the annotation, plots most fequent terms, the named enities, and does the sentiment analysis
	:param dir: path to directory
	:param language: indicates which language is being used
	:type dir: string
	:type language: string
	"""
	annotations=get_annotations(dir, language)
	with open(f'{dir}/analysis_results.txt', 'w', encoding='utf8') as outfile:
		terms_a1_neutral, terms_a2_neutral, terms_a1_positive, terms_a2_positive, terms_a1_negative, terms_a2_negative, vocab, model=get_annotation_stats(annotations, 'en')
		plot_most_frequent_terms(terms_a1_neutral, terms_a1_positive, terms_a1_negative, 1, language, vocab, model)
		plot_most_frequent_terms(terms_a2_neutral, terms_a2_positive, terms_a2_negative, 2, language, vocab, model)
		full_ner(annotations, language, 1, outfile)
		full_ner(annotations, language, 2, outfile)
		analyse_sentiment(annotations, outfile)


def get_annotations(dir, language):
	"""
	Collects the annotation sheets in the directory and merges them
	:param dir: path to the directory were the annotations are in
	:param language: indicates the language that is used
	:type dir: string
	:type language: string
	:return: merged dataframe
	"""
	df1=pd.read_csv(f"{dir}/{language}_annotationsheet_a1.csv", delimiter=';')
	df2=pd.read_csv(f"{dir}/{language}_annotationsheet_a2.csv", delimiter=';')
	df1['Annotation 2'] = df2['Annotation']
	return df1


def get_annotation_stats(df, language):
	"""
	Gets most frequent annotations that there are in the vocabulary, determines the model and the vocabulary
	:param df: dataframe containing the
	:param language: indicates the languages we're working with
	:type df: dataframe
	:type language: string
	:return: 6 lists of the most words for each annotator and classification, the model, and the vocabulary
	"""
	nlp = stanza.Pipeline(language)
	a1_freq_neutral = Counter()
	a2_freq_neutral = Counter()
	a1_freq_positive = Counter()
	a2_freq_positive = Counter()
	a1_freq_negative = Counter()
	a2_freq_negative = Counter()

	num_sentences=0
	#checks which model and stopwords to use
	if language == 'en':
		stop_lan = 'english'
		fasttext_model = KeyedVectors.load_word2vec_format("models/wiki-news-300d-1M.vec")
	elif language == 'nl':
		stop_lan = 'dutch'
		fasttext_model = KeyedVectors.load_word2vec_format("models/cc.nl.300.vec")
	else:
		raise NotImplementedError

	stop_and_punct = stopwords.words(stop_lan)
	for i in string.punctuation:
		stop_and_punct.append(i)

	for i in range(len(df['Instance'])):
		current_article = df['Instance'][i]
		# Skip empty articles
		if current_article != '':
			# Process the article with the stanza pipeline
			processed_article = nlp(current_article)
			sentences = processed_article.sentences

			# Iterate through all sentences of the article
			for sentence in sentences:
				num_sentences += 1
				all_tokens_without = [token.text.lower() for token in sentence.tokens if token.text.lower() not in stop_and_punct]
				if df['Annotation'][i] == 'Positive':
					a1_freq_positive.update(all_tokens_without)
				elif df['Annotation'][i] == 'Negative':
					a1_freq_negative.update(all_tokens_without)
				elif df['Annotation'][i] == 'Neutral':
					a1_freq_neutral.update(all_tokens_without)
				if df['Annotation 2'][i] == 'Positive':
					a2_freq_positive.update(all_tokens_without)
				elif df['Annotation 2'][i] == 'Negative':
					a2_freq_negative.update(all_tokens_without)
				elif df['Annotation 2'][i] == 'Neutral':
					a2_freq_neutral.update(all_tokens_without)
	model = fasttext_model
	smaller_vocab = {k: model.vocab[k] for k in list(model.vocab.keys())[0:5000]}
	vocab = smaller_vocab
	vocab_list = list(vocab.keys())
	#Extraxt the most frequent terms for each annotator and classification
	terms_a1_neutral=extract_most_freq_terms(a1_freq_neutral, vocab_list)
	terms_a2_neutral=extract_most_freq_terms(a2_freq_neutral, vocab_list)
	terms_a1_positive=extract_most_freq_terms(a1_freq_positive, vocab_list)
	terms_a2_positive=extract_most_freq_terms(a2_freq_positive, vocab_list)
	terms_a1_negative=extract_most_freq_terms(a1_freq_negative, vocab_list)
	terms_a2_negative=extract_most_freq_terms(a2_freq_negative, vocab_list)
	return terms_a1_neutral, terms_a2_neutral, terms_a1_positive, terms_a2_positive, terms_a1_negative, terms_a2_negative, vocab, model


def extract_most_freq_terms(counter, vocab_list):
	"""
	Extracts the 50 most frequent terms from the counter and adds them to the wordlist if they appear in the vocab
	:param counter: counter of the most frequents words
	:param vocab_list: list of the words
	:type counter: Counter
	:type vocab_list: list
	:return: list of the most frequent words that are in the vocabulary
	"""
	word_list=[]
	list_of_counts=counter.most_common(50)
	for i in range(len(list_of_counts)):
		if list_of_counts[i][0] in vocab_list:
			word_list.append(list_of_counts[i][0])
	return word_list

def plot_most_frequent_terms(terms_neutral, terms_positive, terms_negative, annotator, language, vocab, model):
	"""
	Plots the most frequent terms, which positions are based on the embedding model
	:param terms_neutral: list of the most frequent neutral terms
	:param terms_positive: list of the most frequent positive terms
	:param terms_negative: list of the most frequent
	:param annotator: indicates which annotations are used
	:param language: indicates the language used
	:param vocab: dictionary containing the embeddings
	:param model: the loaded fasttext model
	:type terms_positive: list
	:type terms_negative: list
	:type terms_neutral: list
	:type annotator: int
	:type vocab: dict
	:type model: gensim wordtovec model
	:return:
	"""
	# Apply dimensionality reduction with T-SNE
	high_dimensional = model[vocab]
	reduction_technique = TSNE(n_components=2)
	vocab_list = list(vocab.keys())

	two_dimensional = reduction_technique.fit_transform(high_dimensional)

	term_indices_neutral = [vocab_list.index(term) for term in terms_neutral if term in vocab_list]
	term_indices_positive = [vocab_list.index(term) for term in terms_positive if term in vocab_list]
	term_indices_negative = [vocab_list.index(term) for term in terms_negative if term in vocab_list]

	x_pos = [two_dimensional[index, 0] for index in term_indices_positive]
	y_pos = [two_dimensional[index, 1] for index in term_indices_positive]
	x_neg = [two_dimensional[index, 0] for index in term_indices_negative]
	y_neg = [two_dimensional[index, 1] for index in term_indices_negative]
	x_neut = [two_dimensional[index, 0] for index in term_indices_neutral]
	y_neut = [two_dimensional[index, 1] for index in term_indices_neutral]

	fig, ax = plt.subplots(1, 1, figsize=(20, 10))

	for x, y in zip(x_pos, y_pos):
		positive, = ax.plot(x, y, 'o', markersize=10, color="blue", label='Positive')
	for x, y in zip(x_neut, y_neut):
		neutral, = ax.plot(x, y, 's', markersize=10, color="yellow", label='Neutral')
	for x, y in zip(x_neg, y_neg):
		negative, = ax.plot(x, y, '^', markersize=10, color="red", label='Negative')

	# Add title and description
	ax.set_title(f'Terms spread for annotator {annotator}', fontsize=20)
	description = f"The word vectors for the 50 most common words by annotation class in the scraped tweets based {language} fasttext model reduced to two dimensions using the t-sne algorithm. "
	fig.text(.51, .05, description, ha="center", fontsize=14)

	# Hide the ticks
	ax.set_yticks([])
	ax.set_xticks([])

	# Annotate the terms in the plot
	for i, word in enumerate(terms_neutral):
		plt.annotate(word, xy=(x_neut[i], y_neut[i]), fontsize=14)
	for i, word in enumerate(terms_negative):
		plt.annotate(word, xy=(x_neg[i], y_neg[i]), fontsize=14)
	for i, word in enumerate(terms_positive):
		plt.annotate(word, xy=(x_pos[i], y_pos[i]), fontsize=14)

	plt.legend([positive, neutral, negative], ["Positive", "Neutral", "Negative"])
	plt.savefig(f"{language}_A{annotator}_similarity.jpg")
	plt.show()




def annotator_stats(df, annotator, outfile):
	"""
	Calculates the agreement percentage, cohen's kappa and confusion matrix, and prints these and writes them to the outfile

	:param df: dataframe containing the annotations made by both polyglot and the
	:param annotator: indicates which annotator we are working with
	:param outfile: path to outfile
	:type df: dataframe
	:type annotator: int
	:type outfile: string
	"""
	categories = ["Positive", "Negative","Neutral"]
	if annotator == 1:
		column='Annotation'
	else:
		column=f'Annotation {annotator}'
	agreement=np.where(df[column]==df['Annotation Polyglot'], True, False)
	percentage = sum(agreement) / len(agreement)
	# calculate cohen's kappa
	kappa = cohen_kappa_score(df[column], df['Annotation Polyglot'], labels=categories)
	# provide the confusion matrix
	confusions = confusion_matrix(df[column], df['Annotation Polyglot'], labels=categories)
	matrix = pd.DataFrame(confusions, index=categories, columns=categories)
	print("Percentage Agreement: %.2f" % percentage)
	print("Cohen's Kappa: %.2f" % kappa)
	print(matrix)
	#write the results to an outfile
	outfile.write("Percentage Agreement: %.2f" %percentage)
	outfile.write("Cohen's Kappa: %.2f" %kappa)
	outfile.write(matrix.to_markdown())

def sentiment(tweet, language):
	"""
	Collects the total polarity of all words in the tweet

	:param tweet: row of dataframe containing the tweet text
	:param language: indicates the language of the tweets
	:type tweet: string
	:type language: string
	:returns: int with the total polarity score
	"""
	sentiment=0
	text=Text(tweet, hint_language_code = language)
	for w in text.words:
		sentiment+=w.polarity
	return sentiment

def analyse_sentiment(df, outfile):
	'''
			Performs the full sentiment analysis (e.g collecting the polarities, changing them into annotations, and printing and writing the stats to an outfile)

			:param df: dataframe with the tweet data
			:param outfile: indicates the path of the outfile
			:type df: dataframe
			:type outfile: string
		'''
	df['polarity'] = df.apply(lambda row: sentiment(row['Instance'], 'en'), axis=1)
	df['Annotation Polyglot'] = np.where(df['polarity'] >= 1, 'Positive',
									   np.where(df['polarity'] <= -1, 'Negative', 'Neutral'))
	annotator_stats(df,1, outfile)
	annotator_stats(df,2, outfile)

def plot_babel_net_query(word, language):
	'''
			performs a babelnet query and plots it's most closely related entities in a graph

			:param word: the word the babelnet query is done for
			:param language: indicates for which language the plot is made (e.g. nl/en)
			:type word: string
			:type language: string
			'''
	params = dict(lemma=word, searchLang=language, key=babelnet_key)
	resp = requests.get(url=wordurl, params=params)
	word_data = resp.json()
	id=word_data[0]['id']
	relations_params = dict(id=id, key=babelnet_key)
	resp = requests.get(url=relations_url, params=relations_params)

	relations_data = resp.json()
	# Create a graph structure
	relations_graph = nx.Graph()
	relations_graph.add_node(word)
	for relation in relations_data[0:20]:
		#collect the lemma's for the relations
		synset_params = dict(id=relation["target"], key=babelnet_key, targetLang=language.upper())
		resp = requests.get(url=synseturl, params=synset_params)
		synsetdata = resp.json()
		relations_graph.add_edge(word, synsetdata['senses'][0]['properties']['simpleLemma'],
								 title=relation["pointer"]["name"])
	edge_labels = nx.get_edge_attributes(relations_graph, 'title')
	# Map edge labels to colors
	color_palette = sns.color_palette("Dark2")
	unique_labels = set(list(edge_labels.values()))
	labels2color = {label: color_palette[i] for i, label in enumerate(unique_labels)}
	edge_colors = [labels2color[label] for label in edge_labels.values()]

	# Create a figure
	fig, ax = plt.subplots(1, 1, figsize=(20, 15))
	pos = nx.spring_layout(relations_graph)
	# Draw the nodes and edges with colors
	node_colors = ["yellow" for node in relations_graph.nodes]
	# The first node should be red
	node_colors[0] = "red"
	nx.draw_networkx_nodes(relations_graph, pos, node_color=node_colors, node_size=3000, ax=ax)
	nx.draw_networkx_labels(relations_graph, pos, ax=ax)
	nx.draw_networkx_edges(relations_graph, pos, arrows=True, edge_color=edge_colors, width=4, ax=ax)
	nx.draw_networkx_edge_labels(relations_graph, pos, edge_labels=edge_labels)
	ax.set_title(
		f'Graph of lexical enitities related to "{word}" in {language} according to Babelnet',
		fontsize=20)
	description = f"This graph display entities related to {word} and the way they are related based on the Babelnet API."
	fig.text(.51, .05, description, ha="center", fontsize=14)
	fig.savefig(f"{language}_homosexuality_plot.png")

	fig.show()

def full_ner(df, language, annotator, outfile):
	'''
			Performs the full named entity recognition analysis (e.g aggregating the dataframe, collecting and plotting the ner-tags)

			:param df: dataframe with the tweet data
			:param language: indicates for which language the plot is made
			:param annotator: indicates which annotations we are based on
			:param outfile
			:type df: dataframe
			:type language: string
			:type annotator: int
			:type outfile: txt file
			'''
	nlp = stanza.Pipeline(language, processors='tokenize,pos,lemma,ner')
	df_new=aggregated_df(df, annotator)
	df_new['ner_tokens']=df_new.apply(lambda row: collect_ner_tokens(nlp(row['Instance'])), axis=1)
	#if annotator == 1:
		#df_new.apply(lambda row: outfile.write(f"\n{row['Annotation']}: {row['ner_tokens']}"))
	#else:
		#df_new.apply(lambda row: outfile.write(f"\n{row[f'Annotation {annotator}']} :{row['ner_tokens']}"))
	plot_ner(df_new, language, annotator)


def collect_ner_tokens(row):
	'''
			Colllects all ner tokens label frequencies and prints them and all the full ner tokens

			:param row: dataframe row with the tweet text
			:type row: text
			:returns: counterobject with the named entity tokens and their occurences
			'''
	sentences = row.sentences
	ner_list = []
	cats = Counter()

	for sentence in sentences:
		for token in sentence.tokens:
			if not token.ner == "O":
				position, category = token.ner.split("-")

				# we only update the list if they are the final formm for that named enity
				if (position == "S"):
					cats.update({f'{category}': 1})
					ner_list.append(token.text)
				if (position == "B"):
					current_token = token.text
				if (position == "I"):
					current_token = current_token + " " + token.text
				if (position == "E"):
					current_token = current_token + " " + token.text
					ner_list.append(current_token)
					cats.update({f'{category}': 1})
	print(set(ner_list))
	print(cats)
	return cats


def plot_ner(df, language, annotator):
	'''
		Plots the bar graph of the named entity labels

		:param df: dataframe with the tweet data
		:param language: indicates for which language the plot is made
		:param annotator: indicates which annotations we are based on
		:type df: dataframe
		:type language: string
		:type annotator: int
		'''

	if annotator ==1:
		columnname='Annotation'
	else:
		columnname=f'Annotation {annotator}'
	#Turn the df columns into dictionaries
	A = list(df['ner_tokens'][df[columnname] == 'Positive'])[0]
	C = list(df['ner_tokens'][df[columnname] == 'Neutral'])[0]
	B = list(df['ner_tokens'][df[columnname] == 'Negative'])[0]
	fig = plt.figure()
	# Combine the dictionaries into a dataframe and plot that
	ax = pd.DataFrame({'Positive': A, 'Negative': B, 'Neutral': C}).plot.bar(title=f'{language}: Annotator {annotator}')
	ax.set(xlabel="Named entity labels", ylabel="Counts")
	fig = ax.get_figure()
	# This makes sure the X-axis is fully saved, if not the labels might be dropped from the picture
	plt.tight_layout()
	fig.savefig(f"{language}_A{annotator}_Ner.jpg")

def aggregated_df(df, annotator):
	'''
		Aggregates the rows of a dataframe into 1 row per annotation label and returns it

		:param df: dataframe with the tweet data
		:param annotator: indicates which annotations we are based on
		:type df: dataframe
		:type annotator: int
		:returns: aggregated dataframe
		'''
	if annotator==1:
		column='Annotation'
	else:
		column=f'Annotation {annotator}'
	df_aggregation = df.groupby([column], as_index=False).agg({'Instance': ' '.join})
	return df_aggregation

def main():
	get_general_tweet_stats('tweets/eng', 'en')
	get_general_tweet_stats('tweets/nld', 'nl')
	plot_babel_net_query('homosexuality', 'en')
	plot_babel_net_query('homoseksualiteit', 'nl')
	annotation_stats_and_analysis('annotations/eng', 'en')
	annotation_stats_and_analysis('annotations/nld', 'nl')


if __name__ == '__main__':
	main()