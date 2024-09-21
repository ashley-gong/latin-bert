import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def show_graph_with_labels(dataframe):
  df_long = dataframe.stack().reset_index()
  df_long.columns = ['Poem1', 'Poem2', 'Similarity']

  threshold = 0.8
  filtered_sim = df_long[(df_long['Poem1'] != df_long['Poem2']) & 
                                      (df_long['Similarity'] >= threshold)]
  
  G = nx.Graph()
  G.add_nodes_from(dataframe.columns)
  edges = list(zip(filtered_sim['Poem1'], filtered_sim['Poem2'], filtered_sim['Similarity']))
  G.add_weighted_edges_from(edges)

  pos = nx.spring_layout(G, weight='weight', k=0.6, scale=1)  
  nx.draw_networkx_nodes(G, pos, node_size=250)

  edges = nx.draw_networkx_edges(G, pos, width=0.5)
  edge_labels = nx.get_edge_attributes(G, "weight")
  edge_labels = {key: f'{value:.2f}' for key, value in edge_labels.items()}
  nx.draw_networkx_edge_labels(G, pos, font_size=8, edge_labels=edge_labels)

  nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

  plt.title('Poem Similarity Network Graph')
  plt.show()



if __name__ == '__main__':
  similarity_df = pd.read_csv('cat_line1_similarity.csv', index_col=0)
  show_graph_with_labels(similarity_df)