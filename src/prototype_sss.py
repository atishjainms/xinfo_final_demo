from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

# expecting labels.csv in the ../data folder, and articles to be in ../data/articles/articles/ etc.
PROJ_ROOT = Path(__file__).parent.parent
DATA_DIR = (PROJ_ROOT / 'data').resolve()
ARTICLES_DIR = (DATA_DIR / 'articles' / 'articles').resolve()


# get data from labels csv into a couple dictionaries
labels = dict()
# publisher data keys should match the folder names for each publisher for a given day
publisher_data = dict()
with (DATA_DIR / 'labels.csv').resolve().open() as f:
    labelreader = csv.reader(f, delimiter=',')
    firstrow = True
    for row in labelreader:
        if firstrow:
            for ind in range(len(row)):
                labels[row[ind]] = ind-1
            firstrow = False
            continue
        publisher_data[row[0]] = row[1:]

# for lab in labels.keys():
#     print(lab)
print("===========")
# has labels like "left_bias", "right_center_bias", "questionable_source", "conspiracy_pseudoscience"
label_ind = labels['Media Bias / Fact Check, label']
mbfc_labels = dict()
for pub in publisher_data.keys():
    bias_label = publisher_data[pub][label_ind]
    if bias_label != '':
        # print(pub, ':', bias_label)
        mbfc_labels[pub] = bias_label
# print(publisher_data.keys())

dates = [f for f in ARTICLES_DIR.iterdir() if f.is_dir()]
# use a single date as a prototype
smalltest_dir = (ARTICLES_DIR / dates[0]).resolve()
publishers = [f for f in smalltest_dir.iterdir() if f.is_dir()]
print('number of publishers for this date: ', len(publishers))

documents = []
for pub_articles in publishers:
    articles = [f for f in pub_articles.iterdir() if f.is_file()]
    this_publisher = str(articles[0].parent.relative_to(smalltest_dir))
    # print(this_publisher)
    # skip if no label for publisher
    if str(this_publisher) not in mbfc_labels.keys():
        continue
    else:
        # print(this_publisher)
        # print(mbfc_labels[this_publisher])
        # print(articles[0].read_text())
        documents.append(articles[0].read_text())

print('number of documents: ', len(documents))

tfidf_vec = TfidfVectorizer(stop_words='english')
X = tfidf_vec.fit_transform(documents)
id_to_word = tfidf_vec.get_feature_names()

# just plugging stuff into LDA and printing results. not doing anything interesting yet.
lda = LatentDirichletAllocation(n_components=5)
lda.fit(X)

for topic_id, topic in enumerate(lda.components_):
    print('topic id: ', topic_id, ': top 10 words: ', [", ".join([id_to_word[i] for i in topic.argsort()[:10]])])
