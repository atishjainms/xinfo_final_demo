from pathlib import Path
from gensim.models.ldamodel import  LdaModel
# from gensim.models.ldamulticore import LdaMulticore
import gensim.corpora as corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import csv
import numpy as np
import spacy
#import en_core_web_md
import argparse
import json
#import sqlite3
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score

# Constants
STOPWORDS = {
    'say', 'not', 'like', 'go', "be", "have", "s"
}


def predict_bias(model, topic_vecs, labels):
    """

    :param model:
    :param topic_vecs:
    :param labels:
    :return:
    """

    pred_labels = model.predict(topic_vecs)

    # get accuracy from the training data, just to look at whether this even seems feasible...
    # 0.3 f1 score on the training, using 12123 documents. not great results for now.
    print("accuracy on training data: ",
          f1_score(labels, pred_labels, average='weighted'))

    return


def train_model(documents, onehot_enc, labels):
    """

    :param documents:
    :param onehot_enc:
    :param labels:
    :return:
    """

    print('number of documents: ', len(documents))

    id2word = corpora.Dictionary(documents)

    corpus = [id2word.doc2bow(doc) for doc in documents]

    onehot_labels = onehot_enc.transform(labels)

    print("starting LDA model")
    # plug into LDA model.
    # this can take a while with larger number of documents
    lda = LdaModel(num_topics=20,
                   id2word=id2word,
                   corpus=corpus,
                   passes=50,
                   eval_every=1)
    print("topics:")
    for topic in lda.show_topics(num_topics=20,
                                 num_words=20):  # print_topics():
        print(topic)

    # print("getting topics for testing document")
    # topic_prediction = lda.get_document_topics(bow=corpus[0])

    # print(testing_text_raw)
    # print(topic_prediction)

    print("")
    print(
        "starting setup to train a classifier based on LDA topics for each document")

    topic_vecs = []

    # get topic matches and put them into vectors
    for i in range(len(documents)):
        top_topics = lda.get_document_topics(corpus[i],
                                             minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(20)]
        topic_vecs.append(topic_vec)

    # train basic logistic regression
    model = LogisticRegression(class_weight='balanced').fit(topic_vecs, labels)

    return model, topic_vecs


def load_articles(articles_dir, mbfc_labels):
    """

    :param articles_dir:
    :return:
    """
    # initialize return values.
    documents = []
    labels = []
    # store raw text / processed text to use later to look at result example
    testing_text_raw = []

    # load our spacy model
    nlp = spacy.load('en_core_web_md')

    # use 10 days as a prototype
    dates = [f for f in Path(articles_dir).iterdir() if f.is_dir()]

    with nlp.disable_pipes("ner"):
        for date in dates[:2]:  # parsing documents can take a while.
            smalltest_dir = (articles_dir / date).resolve()

            publishers = [f for f in smalltest_dir.iterdir() if f.is_dir()]

            for pub_articles in publishers:
                articles = [f for f in pub_articles.iterdir() if f.is_file()]
                if articles:
                    this_publisher = str(
                        articles[0].parent.relative_to(smalltest_dir))
                    # print(this_publisher)
                    # skip if no label for publisher
                    if str(this_publisher) not in mbfc_labels.keys():
                        continue
                    else:
                        text = articles[0].read_text()

                        # save raw text of first document, just for looking at results later
                        if not testing_text_raw:
                            testing_text_raw = text

                        text = text.replace("\n", " ")

                        # preprocessing text
                        lem_text = [token.lemma_.lower() for token in nlp(text)
                                    if not token.is_stop
                                    and not token.is_punct
                                    and not token.is_space
                                    and not token.lemma_.lower() in STOPWORDS
                                    and not token.pos_ == 'SYM'
                                    and not token.pos_ == 'NUM']
                        # print(lem_text)
                        documents.append(lem_text)
                        labels.append(mbfc_labels[this_publisher])
    return documents, labels


def load_labels(path):
    """

    :param path:
    :return:
    """

    # get data from labels csv into a couple dictionaries
    labels = dict()
    # publisher data keys should match the folder names for each publisher for a given day
    publisher_data = dict()
    #with (path / 'labels.csv').resolve().open() as f:
    with open(path + '\labels.csv') as f:
        labelreader = csv.reader(f, delimiter=',')
        firstrow = True
        for row in labelreader:
            if firstrow:
                for ind in range(len(row)):
                    labels[row[ind]] = ind - 1
                firstrow = False
                continue
            publisher_data[row[0]] = row[1:]


    label_ind = labels['Media Bias / Fact Check, label']
    mbfc_labels = dict()
    for pub in publisher_data.keys():
        bias_label = publisher_data[pub][label_ind]
        if bias_label != '':
            # print(pub, ':', bias_label)
            mbfc_labels[pub] = bias_label
    # print(publisher_data.keys())
    print(set(mbfc_labels.values()))
    print(len(set(mbfc_labels.values())))
    onehot_enc = LabelBinarizer().fit(list(mbfc_labels.values()))

    return mbfc_labels, onehot_enc


def load_data(data_dir, article_dir):
    """
    This method will load in our biased news dataset, either as a json blob
    or as a sqlite database.

    :param path:
    :param mode:
    :return:
    """
    raw_data = None

    # load our news source labels
    mbfc_labels, onehot_enc = load_labels(data_dir)

    # Load the articles into memory
    documents, labels = load_articles(article_dir, mbfc_labels)

    return documents, labels, onehot_enc


def main():
    # Using the argument parser library to handle command line inputs
    # Note: this will also handle validating the input types
    program_desc = ("This program, given a directory full of images, will "
                    "compare them to each other, using two methods, and print "
                    "the two most similar and the two least similar images "
                    "for each method.")
    parser = argparse.ArgumentParser(description=program_desc)

    parser.add_argument("data_dir",
                        type=str,
                        help=("A text file containing our bias dataset."))

    parser.add_argument("article_dir",
                        type=str,
                        help=("Directory the articles live in."))

    #parser.add_argument("comment_symbol",
    #                    type=str,
    #                    help=("Comment symbol used in input file."))

    #parser.add_argument("mode",
    #                    type=str,
    #                    help=("Comment symbol used in input file."))

    inputs = parser.parse_args()

    # load our label data, form of a tuple of (lables, publisher_data)
    documents, labels, onehot_enc = load_data(inputs.data_dir, inputs.article_dir)

    # Train our model
    model, topic_vector = train_model(documents, onehot_enc, labels)

    # Predict
    predict_bias(model, topic_vector, labels)


    # Do the tests
    #TEST_SET()

    return


if __name__ == "__main__":
    main()
