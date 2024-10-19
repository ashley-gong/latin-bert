import argparse, sys, re
import numpy as np
from numpy import linalg as LA
from gen_berts import LatinBERT
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import matplotlib.pyplot as plt
import collections


PINK = '\033[95m'
ENDC = '\033[0m'

def proc(filenames):
	matrix_all=[]
	sents_all=[]
	sent_ids_all=[]
	toks_all=[]
	position_in_sent_all=[]
	doc_ids=[]

	num_parallel_processes = 10
	vals=Parallel(n_jobs=num_parallel_processes)(
			delayed(proc_doc)(f) for f in tqdm(filenames))

	for matrix, sents, sent_ids, toks, position_in_sent, filename in vals:
		matrix_all.append(matrix)
		sents_all.append(sents)
		sent_ids_all.append(sent_ids)
		toks_all.append(toks)
		position_in_sent_all.append(position_in_sent)
		doc_ids.append(filename)

	# matrix_all contains all bert arrays
	return matrix_all, sents_all, sent_ids_all, toks_all, position_in_sent_all, doc_ids

def proc_doc(filename):
	berts=[]
	toks=[]
	sent_ids=[]
	sentid=0
	position_in_sent=[]
	p=0
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			if len(cols) == 2:
				word=cols[0]
				bert=np.array([float(x) for x in cols[1].split(" ")])
				bert=bert/LA.norm(bert)
				toks.append(word)
				berts.append(bert)
				sent_ids.append(sentid)
				position_in_sent.append(p)
				p+=1
			else:
				sentid+=1
				p=0

	sents=[]
	lastid=0
	current_sent=[]
	for s, t in zip(sent_ids, toks):
		if s != lastid:
			sents.append(current_sent)
			current_sent=[]
		lastid=s
		current_sent.append(t)

	matrix=np.asarray(berts)
	
	# matrix is berts
	return matrix, sents, sent_ids, toks, position_in_sent, filename

# increasing window means more words around
def get_window(pos, sentence, window):
	start=pos - window if pos - window >= 0 else 0
	end=pos + window + 1 if pos + window + 1 < len(sentence) else len(sentence)
	return "%s %s%s%s %s" % (' '.join(sentence[start:pos]), PINK, sentence[pos], ENDC, ' '.join(sentence[pos+1:end]))

def get_window_no_color(pos, sentence, window):
	start=pos - window if pos - window >= 0 else 0
	end=pos + window + 1 if pos + window + 1 < len(sentence) else len(sentence)
	return "%s" % (' '.join(sentence[start:end]))

def compare_one(idx, matrix_all, sents_all, sent_ids_all, toks_all, position_in_sent_all, doc_ids, target_bert):
	c_matrix=matrix_all[idx]
	c_sents=sents_all[idx]
	c_sent_ids=sent_ids_all[idx]
	c_toks=toks_all[idx]
	c_pos=position_in_sent_all[idx]
	doc_id=doc_ids[idx]
	similarity=np.dot(c_matrix,target_bert)
	argsort=np.argsort(-similarity)
	len_s,=similarity.shape
	vals5=[]
	vals10=[]
	for i in range(min(100,len_s)):
		tid=argsort[i]
		if tid < len(c_sent_ids) and tid < len(c_pos) and c_sent_ids[tid] < len(c_sents):
			wind10=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 10)
			wind5=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 5)
			vals5.append((similarity[tid], wind5, doc_id ))
			vals10.append((similarity[tid], wind10, doc_id ))

	return vals5, vals10

def compare(berts, target_bert, outputDir, query, sent):

	vals=[]
	outfile="%s/%s_%s" % (outputDir, query, re.sub(" ", "_", sent))
	out=open(outfile, "w", encoding="utf-8")

	matrix_all, sents_all, sent_ids_all, toks_all, position_in_sent_all, doc_ids=berts

	for idx in range(len(doc_ids)):
		c_matrix=matrix_all[idx]
		c_sents=sents_all[idx]
		c_sent_ids=sent_ids_all[idx]
		c_toks=toks_all[idx]
		c_pos=position_in_sent_all[idx]
		doc_id=doc_ids[idx]

		similarity=np.dot(c_matrix,target_bert)
		argsort=np.argsort(-similarity)
		len_s,=similarity.shape
		for i in range(min(100,len_s)):
			tid=argsort[i]
			if tid < len(c_sent_ids) and tid < len(c_pos) and c_sent_ids[tid] < len(c_sents):
				wind10=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 10)
				wind5=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 5)
				out.write("%s\t%s\t%s\n" % (similarity[tid], wind10, doc_id))
				vals.append((similarity[tid], wind5, doc_id))

	vals=sorted(vals, key=lambda x: x[0])		# all of the values
	for a, b, doc in vals[-25:]:
		print("%.3f\t%s\t%s" % (a, b, doc))

	out.close()

# Only taking top 50 sentences
def comparison_preprocessing(berts, target_bert, outputDir, query, sent):
	vals=[]
	outfile="%s/%s_%s" % (outputDir, query, re.sub(" ", "_", sent))
	out=open(outfile, "w", encoding="utf-8")

	matrix_all, sents_all, sent_ids_all, toks_all, position_in_sent_all, doc_ids=berts

	# each row of matrix_all is a vector compared to target_bert

	for idx in range(len(doc_ids)):
		c_matrix=matrix_all[idx]
		c_sents=sents_all[idx]
		c_sent_ids=sent_ids_all[idx]
		c_toks=toks_all[idx]
		c_pos=position_in_sent_all[idx]
		doc_id=doc_ids[idx]

		similarity=np.dot(c_matrix,target_bert)
		argsort=np.argsort(-similarity)
		len_s,=similarity.shape
		for i in range(min(100,len_s)):
			tid=argsort[i]
			if tid < len(c_sent_ids) and tid < len(c_pos) and c_sent_ids[tid] < len(c_sents):
				wind=get_window_no_color(c_pos[tid], c_sents[c_sent_ids[tid]], 5)
				out.write("%s\t%s\t%s\n" % (similarity[tid], wind, doc_id))
				vals.append((similarity[tid], wind, doc_id, c_matrix[tid]))

	vals=sorted(vals, key=lambda x: x[0])		# all of the values
	out.close()
	return vals


# K Means
def compare_cluster_kmeans(berts, target_bert, target_sent, outputDir, query, sent, num_clusters):
	vals = comparison_preprocessing(berts, target_bert, outputDir, query, sent)
	embeddings = [tuple[3] for tuple in vals]

	kmeans = KMeans(n_clusters=num_clusters, random_state=0)
	kmeans.fit(embeddings)

	labels = kmeans.labels_
	pca = PCA(n_components=2)
	reduced_embeddings = pca.fit_transform(embeddings)

	target = np.argmax([tuple[0] for tuple in vals])

	df = pd.DataFrame({
    'PCA1': reduced_embeddings[:, 0],
    'PCA2': reduced_embeddings[:, 1],
    'Sentence': [tuple[1] for tuple in vals],
		'Document': [tuple[2] for tuple in vals],
    'Cosine Similarity': [tuple[0] for tuple in vals],
    'Cluster': labels,
		'Size': [15 if i == target else 3 for i in range(len(vals))],
    'Color': ['Closest' if i == target else f'Cluster {labels[i]}' for i in range(len(vals))]
	})

	fig = px.scatter(
    df, x='PCA1', y='PCA2', color='Color', size='Size',
    hover_data={
        'Sentence': True,
				'Document': True,
        'Cosine Similarity': ':.2f',  # Show cosine similarity with 2 decimal places
        'PCA1': False,  # Hide PCA components in hover
        'PCA2': False   # Hide PCA components in hover
    },
    title=f"Sentence Clusters for Target: {target_sent}"
	)

	fig.show()

# Try Agglomerative Hierarchy clustering
def compare_cluster_agglom(berts, target_bert, target_sent, outputDir, query, sent, num_clusters):
	vals = comparison_preprocessing(berts, target_bert, outputDir, query, sent)[-50:]
	embeddings = [tuple[3] for tuple in vals]
	agglom = AgglomerativeClustering(n_clusters=num_clusters)	# Euclidean distance, Ward linkage
	agglom.fit(embeddings)

	target = np.argmax([tuple[0] for tuple in vals])
	labels = agglom.labels_
	pca = PCA(n_components=2)
	reduced_embeddings = pca.fit_transform(embeddings)

	target = np.argmax([tuple[0] for tuple in vals])

	df = pd.DataFrame({
    'PCA1': reduced_embeddings[:, 0],
    'PCA2': reduced_embeddings[:, 1],
    'Sentence': [tuple[1] for tuple in vals],
		'Document': [tuple[2] for tuple in vals],
    'Cosine Similarity': [tuple[0] for tuple in vals],
    'Cluster': labels,
		'Size': [15 if i == target else 3 for i in range(len(vals))],
    'Color': ['Closest' if i == target else f'Cluster {labels[i]}' for i in range(len(vals))]
	})

	fig = px.scatter(
    df, x='PCA1', y='PCA2', color='Color', size='Size',
    hover_data={
        'Sentence': True,
				'Document': True,
        'Cosine Similarity': ':.2f',  # Show cosine similarity with 2 decimal places
        'PCA1': False,  # Hide PCA components in hover
        'PCA2': False   # Hide PCA components in hover
    },
    title=f"Agglomerative Hierarchical Sentence Clusters for Target: {target_sent}"
	)

	# Dendrogram - Plotly dimensions not working but Matplotlib does
	# Z = linkage(reduced_embeddings, method='ward')
	# sentence_counts = collections.Counter(sentences)
	# unique_sentences = []
	
	# for sentence in sentences:
	# 		if sentence_counts[sentence] > 1:
	# 				idx = sentence_counts[sentence]
	# 				unique_sentences.append(f"{sentence[:30]} ({idx})")  # Shorten labels and add uniqueness
	# 				sentence_counts[sentence] -= 1
	# 		else:
	# 				unique_sentences.append(sentence[:30])

	# plt.figure(figsize=(10, 7))
	# dendrogram(Z, labels=unique_sentences, orientation='left')
	# plt.title(f"Dendrogram of Agglomerative Hierarchical Clustering (PCA-Reduced) for Target: {target_sent}")
	# plt.xlabel("Distance")
	# plt.ylabel("Sentences")
	# plt.show()

	fig.show()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--bertPath', help='path to pre-trained BERT', required=True)
	parser.add_argument('-t', '--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)
	parser.add_argument('-i', '--inputDir', help='input files to search', required=True)
	parser.add_argument('-o', '--outputDir', help='output directory to write results to', required=True)
	
	args = vars(parser.parse_args())

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]			
	inputDir=args["inputDir"]			
	outputDir=args["outputDir"]			

	onlyfiles = [f for f in listdir(inputDir) if isfile(join(inputDir, f))]
	target_files=[]
	for filename in onlyfiles:

		target_files.append("%s/%s" % (inputDir, filename))

	bert=LatinBERT(tokenizerPath=tokenizerPath, bertPath=bertPath)

	berts=proc(target_files)

	print ("> ",)
	line = sys.stdin.readline()
	while line:
		word=line.rstrip()
		toks=line.rstrip().split(" ")
		target_word=toks[0]
		sents=[' '.join(toks[1:])]

		bert_sents=bert.get_berts(sents)[0]
		toks=[]
		target_bert=None
		seen=False
		# tok = token, b = bert embedding
		for idx, (tok, b) in enumerate(bert_sents):
			toks.append(tok)
			if tok == target_word:
				if seen:
					print("more than one instance of %s" % target_word)
					sys.exit(1)
				else:
					target_bert=b
					seen=True
		target_sent = ' '.join(toks)
		print(target_sent)
		print("target: %s" % target_word)

		if target_bert is not None:

			target_bert=target_bert/LA.norm(target_bert)
			compare(berts, target_bert, outputDir, target_word, sents[0])
			compare_cluster_agglom(berts, target_bert, target_sent, outputDir, target_word, sents[0], 3)
			compare_cluster_kmeans(berts, target_bert, target_sent, outputDir, target_word, sents[0], 3)

		print ("> ",)
		line = sys.stdin.readline()

