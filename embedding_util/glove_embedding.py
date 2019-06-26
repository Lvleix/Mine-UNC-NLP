import gensim
import numpy as np
import nltk.tokenize as tk

class gloveModel:
	def __init__(self, path):
		print("Now load in glove model.")
		#self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
		self.vocab_size, self.embedding_size, self.model = self.embedding_in(path)
		print("Model loaded!")
		self.tokenizer = tk.WordPunctTokenizer()

	def wordsEmbedding(self, words):
		res= []
		for word in words:
			if word in self.model:
				res.append(self.model[word])
		#return np.array(res)
		return res

	def doc_retr_embed(self, data):
		words = self.tokenizer.tokenize(data)
		#words[0] = words[0].lower()
		return self.wordsEmbedding(words)

	def sen_sele_embed(self, data):
		words = self.tokenizer.tokenize(data)
		#words[0] = words[0].lower()
		return self.wordsEmbedding(words)

	def cla_veri_embed(self, data):
		words = self.tokenizer.tokenize(data)
		#words[0] = words[0].lower()
		return self.wordsEmbedding(words)

	def embedding_in(self, path):
		with open(path, 'r', encoding='utf8') as f:
			# vocab_size, embedding_size = tuple(f.readline().split())
			w2v = {line.split()[0]: np.array([float(entry) for entry in line.split()[1:]], dtype=np.float32) for line in
				   f}

		return len(w2v), len(w2v['the']), w2v

if __name__ == "__main__":
	test = ["I","love","apples"]
	s = 'No Escape was released in 2015.'
	model = gloveModel("D:/大三下/自然语言处理/lab2/glove.6B/glove.6B.50d.txt")
	print(model.model["the"])
	res = model.wordsEmbedding(test)
	print(res)
