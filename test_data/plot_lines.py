import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
  edge_labels = nx.get_edge_attributes(G, 'weight')
  edge_labels = {key: f'{value:.2f}' for key, value in edge_labels.items()}
  nx.draw_networkx_edge_labels(G, pos, font_size=8, edge_labels=edge_labels)

  nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

  plt.title('Poem Similarity Network Graph')
  plt.show()


def visualize_dynamic_network(df, threshold=0.8):
    
    df_long = df.stack().reset_index()
    df_long.columns = ['Poem1', 'Poem2', 'Similarity']
    filtered_sim = df_long[(df_long['Poem1'] != df_long['Poem2']) & (df_long['Similarity'] >= threshold)]

    G = nx.Graph()
    G.add_nodes_from(df.columns)

    edges = list(zip(filtered_sim['Poem1'], filtered_sim['Poem2'], filtered_sim['Similarity']))
    G.add_weighted_edges_from(edges)

    pos = nx.spring_layout(G, weight='weight')

    edge_x = []
    edge_y = []
    edge_text = []
    xtext = []
    ytext = []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        xtext.append((x0+x1)/2)
        ytext.append((y0+y1)/2)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # Separate the edges with None
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_text.append(f'Similarity: {edge[2]:.2f}')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        mode='lines',
    )

    eweights_trace = go.Scatter(
      x=xtext,y= ytext, mode='markers',
      marker_size=0.5,
      text=edge_text,
      hoverinfo='text',
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace, eweights_trace],
                    layout=go.Layout(
                        title='Poem Similarity Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    )
                   )

    fig.show()

if __name__ == '__main__':
  similarity_df = pd.read_csv('cat_line1_similarity.csv', index_col=0)
  # show_graph_with_labels(similarity_df)
  visualize_dynamic_network(similarity_df)