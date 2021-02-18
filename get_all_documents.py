import sys
import tweepy
import os
'''
Code based on Language as Data Lab1.5-Processing_tweets
'''

#Setting up the twitter API. Please adjust for own access information
API_KEY = '548hQqzIk8gRiAVKQylVgAhjZ'
API_SECRET = '7JHiT2UvvOAF2oE7ULJGQj1tQycZzarNMOdgF1GVW1QNEY9Pj8'
ACCESS_TOKEN = '1318545959324913664-OQNfKWo3m6A0fqTyCXOaxo9sZytedn'
ACCESS_SECRET = '2GOueNXvh3mqcDx8qfClB66jPRbX4XxCt2c8WgazbdW0s'
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


def get_tweets(language):
	'''
	Collects 100 tweets in english or dutch about gay marriage between november 6th 2020 and november 7th 2020

	:param language: language code for the twitter api
	:type language: string
	:returns list of tweets
	'''
	
	#Set keywords based on language, if the language is not implemented error
	if language=='nl':
		keywords="(homohuwelijk) AND (homo OR huwelijk OR queer OR nicht) OR (homo OR rechten) OR (homorechten) OR (homoseksualiteit) OR (reformatorisch OR religieus OR afwijzen OR christenen OR moslims OR islam)"
	elif language =='en':
		keywords="((gay OR fag OR queer OR homosexual) AND (rights OR marriage)) OR ((gaymarriage) OR (gayrights) OR (queerrights) OR (queermarriage) OR (fagrights) OR (fagmarriage)) OR (religious OR christianity OR christian OR islam OR muslim OR rejection))"
	else:
		raise NotImplementedError()
	filter = "-filter:retweets" #Set filter to ignore retweets
	query = keywords + filter 
	start_date = "2020-11-06"
	end_date = "2020-11-07"
	nr_tweets = 100
	tweet_iterator = tweepy.Cursor(api.search, q=query, since=start_date, till=end_date, lang=language, tweet_mode='extended').items(nr_tweets)
	tweets = list(tweet_iterator) 
	return tweets  

def write_to_dir(outdir, tweets, language):
	'''
	Write the found tweets in csv file in the given outdirectory split based on language

	:param outdir: path to the oirectory the tweets will be placed in
	:param tweets: list of tweets collected from the twitter api
	:param language: language code for the twitter api
	:type outdir: string
	:type tweets: list
	:type language: string
	'''
	
	#Check if the language is Dutch or English, if neither Error
	if language=='nl':
		iso='nld'
	elif language =='en':
		iso='eng'
	else:
		raise NotImplementedError()

	#Check if the final directory exists, if not make the directory
	if not os.path.exists(f'{outdir}\\{iso}'):
		os.makedirs(f'{outdir}\\{iso}')

	tweets_as_text =[]

	# Iterate over the tweets and append the information
	for tweet in tweets:
		text = tweet.full_text.replace("\n", " ")
		keep = str(tweet.created_at) + "\t" + tweet.user.screen_name + "\t" + str(tweet.user.verified)+"\t"+tweet.user.location + "\t" + str(tweet.user.followers_count)+ "\t" + str(tweet.user.friends_count)+"\t" + text+ "\t" + str(tweet.retweet_count)+"\t" + str(tweet.favorite_count) +"\t" + tweet.lang
		tweets_as_text.append(keep)

    #Write the tweets in a csv file in the given outdirectory
	with open(f'{outdir}\\{iso}\\gaymarriage_tweets.tsv', 'w', encoding='utf8') as outfile:
		csv_header = "Created at\tUser\tVerified\tUser Location\tFollowers\tFollowing\tText\tRetweets\tFavorites\tLanguage\n"
		outfile.write(csv_header)
		outfile.write("\n".join(tweets_as_text))


def main():
	try:
		outdir=sys.argv[1]
	except IndexError:
		outdir='tweets'
	nl_tweets=get_tweets('nl') #Collect tweets in dutch
	en_tweets=get_tweets('en') #Collect tweets in english
	write_to_dir(outdir,nl_tweets,'nl') #Store all the dutch tweets
	write_to_dir(outdir,en_tweets,'en') #Store all the english tweets


if __name__ == '__main__':
	main()