from nltk.corpus import wordnet as wn

def is_lemma(a, b):
	for synSet in wn.synsets(a):
		lemma_set = synSet.lemma_names()
		if b in lemma_set:
			return 1
	return 0

def is_antonym(a, b):
	for synSet in wn.synsets(a):
		for lemma in synSet.lemmas():
			lemma_set = [lemma1.name() for lemma1 in lemma.antonyms()]
			if b in lemma_set:
				return 1
	return 0

def path_length(a, b):
	pass

if __name__ == "__main__":
	print(is_lemma("water","H2O"))
	print(is_antonym("walk","ride"))
