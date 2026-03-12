from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_classifier(category_keywords):

    texts = []
    labels = []

    for category, keywords in category_keywords.items():

        for word in keywords:

            texts.append(word)
            labels.append(category)

    classifier = make_pipeline(
        CountVectorizer(),
        MultinomialNB()
    )

    classifier.fit(texts, labels)

    return classifier