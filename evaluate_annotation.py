import pandas as pd
import glob
import os.path
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix
categories = ["Positive", "Negative", "Neutral"]

''''Code from Language as Data lab 3.2 Evaluate Annotations'''

def collect_files(dir):
    '''
        	Collect all the annotation files in the specified directory
        	and put them in a dictionary with each annotator and their annotations.

        	:param dir: path to the directory the annotationsheets are placed in
        	:type dir: string

        	:returns: dictionary with annotations per annotator
        	'''
    annotations = {}
    for sheet in glob.glob(f"{dir}/*_annotationsheet_a*.csv"):
        filename, extension = os.path.basename(sheet).split(".")
        lang,prefix, annotator= filename.split("_")

        # Read in annotations
        annotation_data = pd.read_csv(sheet, sep=";", header=0, keep_default_na=False)
        annotations[annotator] = annotation_data["Annotation"]
    return annotations

def get_stats_and_confusion(annotations,dir):
    '''
        	Calculates the agreement and cohen's kappa for the annotator combinations and the confusion matrix
        	Prints the results, and writes them to a txt in the specified directory.
        	:param annotations: dictionary containing the annotations per annotator
        	:param dir: path to the directory the annotationsheets are placed in
        	:type annotations: dict
        	:type dir: string

        	:returns: dictionary with annotations per annotator
        	'''
    for annotator_a, annotator_b in combinations(annotations.keys(), len(annotations.keys())):
        # calculate the agreement percentage
        agreement = [anno1 == anno2 for anno1, anno2 in  zip(annotations[annotator_a], annotations[annotator_b])]
        percentage = sum(agreement)/len(agreement)
        print("Percentage Agreement: %.2f" %percentage)
        #calculate cohen's kappa
        kappa = cohen_kappa_score(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print("Cohen's Kappa: %.2f" %kappa)
        #provide the confusion matrix
        confusions = confusion_matrix(annotations[annotator_a], annotations[annotator_b], labels=categories)
        matrix= pd.DataFrame(confusions, index=categories, columns=categories)
        print(matrix)
        #write results to a txt
        with open(f'{dir}/annotation_evaluation.txt', 'w', encoding='utf8') as outfile:
            outfile.write("Percentage Agreement: %.2f\n" %percentage)
            outfile.write("Cohen's Kappa: %.2f\n" %kappa)
            outfile.write(matrix.to_markdown())

def full_evaluation(dir):
    '''
    	Collect all the annotation file in the specified directory and calculate the agreement and cohen's kappa, provide the confusion matrix. It will print the results, and write them to a txt file in the specified directory.
    	:param dir: path to the directory the annotationsheets are placed in
    	:type dir: string
    	'''
    annotations=collect_files(dir)
    get_stats_and_confusion(annotations,dir)

def main():
    print('English:')
    full_evaluation('annotations/eng')
    print('\nDutch:')
    full_evaluation('annotations/nld')


if __name__ == '__main__':
    main()