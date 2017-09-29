from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

seed_num = 42
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = load_files('20news-bydate-train', categories = categories, random_state=seed_num)
# debug
print(b"\n".join(twenty_train.data[0].split(b'\n')[:3]).decode('utf-8'), '\n')

count_vect = CountVectorizer(decode_error='ignore')
x_train_counts = count_vect.fit_transform(twenty_train.data)

tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
x_train_tf = tf_transformer.transform(x_train_counts)

clf = MultinomialNB().fit(x_train_tf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
x_new_counts = count_vect.transform(docs_new)
x_new_tf = tf_transformer.transform(x_new_counts)

predicted = clf.predict(x_new_tf)

for doc, category in zip(docs_new, predicted):
    print(f"{doc} => {twenty_train.target_names[category]}")