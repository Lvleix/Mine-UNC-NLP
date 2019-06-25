import gensim
import numpy as np

class gloveModel:
	def __init__(self, path):
		print("Now load in glove model.")
		self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
		print("Model loaded!")
		pass

	def wordsEmbedding(self, words):
		res= []
		for word in words:
			if word in self.model:
				res.append(self.model[word])
		return np.array(res)

if __name__ == "__main__":
	test = ["I","love","apples"]
	model = gloveModel("D:\\PoLanDocument\\2019春夏\\自然语言处理\\glove.6B.50d.txt")
	res = model.wordsEmbedding(test)
	print(res)
	pass
