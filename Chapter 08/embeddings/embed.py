from sentence_transformers import SentenceTransformer, util

sentences = ["Dreaming of blue skies", "I am looking for a painting with clouds", "Dreaming of clouds"]
print("Sentences:\n" + str(sentences))

general_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
general_embeddings = general_model.encode(sentences)
print("General Embeddings:\n" + str(general_embeddings))

qa_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

qa_embeddings = qa_model.encode(sentences)
print("QA Embeddings:\n" + str(qa_embeddings))

print("Similarity between phrase 1 and 2 (General):", util.dot_score(general_embeddings[0], general_embeddings[1]))
print("Similarity between phrase 1 and 2 (QA):", util.dot_score(qa_embeddings[0], qa_embeddings[1]))

print("Similarity between phrase 1 and 3 (General):", util.dot_score(general_embeddings[0], general_embeddings[2]))
print("Similarity between phrase 1 and 3 (QA):", util.dot_score(qa_embeddings[0], qa_embeddings[2]))