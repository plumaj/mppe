from gensim.models import KeyedVectors
import umap, matplotlib.pyplot as plt

kv = KeyedVectors.load("models/lb_de_fr.kv")
words = ["Lëtzebuerg", "Luxembourg", "Berlin", "Paräis", "Zug"]  # LB/DE/FR mix
vecs  = kv[words]
embedding = umap.UMAP().fit_transform(vecs)
plt.scatter(embedding[:,0], embedding[:,1])
for w, (x,y) in zip(words, embedding): plt.text(x,y,w)
plt.show()
