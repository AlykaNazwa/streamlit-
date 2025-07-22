import streamlit as st
import pandas as pd
import unicodedata
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from itertools import combinations
from community import community_louvain
import matplotlib.cm as cm

st.set_page_config(layout="wide")
st.title("Analisis Jaringan Kolaborasi Musisi Spotify")

# ===================== Upload Dua File =====================
st.header("1. Upload Dataset")
mentah = st.file_uploader("Unggah Dataset Awal (Belum Bersih)", type="csv", key="file1")
bersih = st.file_uploader("Unggah Dataset Bersih (Hasil Preprocessing)", type="csv", key="file2")

if mentah and bersih:
    # ========== PREPROCESSING ========== 
    df = pd.read_csv(mentah, encoding='latin1')
    df = df[['track_name', 'artist(s)_name', 'artist_count', 'released_year']].copy()
    df = df.drop_duplicates(subset=['track_name', 'artist(s)_name', 'released_year'])
    df['track_name'] = df['track_name'].str.lower().str.strip()
    df['artist(s)_name'] = df['artist(s)_name'].str.lower().str.strip()

    remix_keywords = ['remix', 'version', 'remastered']
    df['is_remix'] = df['track_name'].apply(lambda x: any(k in x for k in remix_keywords))
    df = df[~((df['is_remix'] == True) & (df['artist_count'] == 1))]
    df = df.drop(columns=['is_remix'])

    df_kolaborasi = df[df['artist_count'] >= 2].copy()

    def bersihkan_nama_final(artist):
        artist = artist.strip().replace('"', '').replace("'", '')
        artist = unicodedata.normalize('NFKD', artist)
        artist = ''.join([c for c in artist if not unicodedata.combining(c)])
        artist = artist.encode("ascii", "ignore").decode("ascii")
        artist = re.sub(r'[^a-zA-Z0-9\s]', '', artist)
        return artist.lower().strip()

    def valid_nama_musisi(nama):
        if len(nama) < 3:
            return False
        if not any(c.isalpha() for c in nama):
            return False
        if re.match(r'^[a-z]*[0-9]+[a-z0-9]*$', nama):
            return False
        if 'aa12' in nama:
            return False
        return True

    df_kolaborasi['artist_list'] = df_kolaborasi['artist(s)_name'].str.split(',')
    df_kolaborasi['artist_list'] = df_kolaborasi['artist_list'].apply(
        lambda artists: [bersihkan_nama_final(a) for a in artists if valid_nama_musisi(bersihkan_nama_final(a))]
    )

    jumlah_data_bersih = len(df_kolaborasi)
    st.success(f"Jumlah data kolaborasi setelah dibersihkan: {jumlah_data_bersih} baris")
    st.dataframe(df_kolaborasi.head())

    # ========== GRAF ==========
    df_bersih = pd.read_csv(bersih)
    df_bersih['artist_list'] = df_bersih['artist_list'].apply(ast.literal_eval)

    G = nx.Graph()
    for artists in df_bersih['artist_list']:
        for a, b in combinations(sorted(set(artists)), 2):
            if G.has_edge(a, b):
                G[a][b]['weight'] += 1
            else:
                G.add_edge(a, b, weight=1)

    st.subheader("2. Ringkasan Struktur Jaringan")
    st.markdown(f"**Jumlah Musisi (Nodes):** {G.number_of_nodes()}")
    st.markdown(f"**Jumlah Kolaborasi (Edges):** {G.number_of_edges()}")
    st.markdown(f"**Jumlah Komponen Terhubung:** {nx.number_connected_components(G)}")

    # ========== VISUALISASI SPRING ==========
    pos = nx.spring_layout(G, k=0.5, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=5)
    plt.title("Visualisasi Jaringan Kolaborasi Musisi (Semua Node)")
    plt.axis("off")
    st.pyplot(plt)

    # ========== KOMPONEN TERBESAR ==========
    G_sub = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos_sub = nx.spring_layout(G_sub, seed=42)
    weights_sub = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G_sub, pos_sub, node_size=30, node_color='orange', alpha=0.8)
    nx.draw_networkx_edges(G_sub, pos_sub, width=[w*0.2 for w in weights_sub], edge_color='black', alpha=0.5)
    plt.title("Visualisasi Komponen Terbesar")
    plt.axis('off')
    st.pyplot(plt)

    # ========== MATRKS KETETANGGAAN & HEATMAP ==========
    adj_matrix = nx.to_pandas_adjacency(G, weight='weight')
    st.write("Ukuran matriks:", adj_matrix.shape)
    st.dataframe(adj_matrix.iloc[:10, :10])

    subset = adj_matrix.iloc[:100, :100]
    plt.figure(figsize=(14, 12))
    sns.heatmap(subset, cmap='YlOrBr', linewidths=0.5, linecolor='gray')
    plt.title("Heatmap Matriks Ketetanggaan Kolaborasi Musisi")
    st.pyplot(plt)

    # ========== CENTRALITY ==========
    deg = nx.degree_centrality(G)
    clo = nx.closeness_centrality(G)
    bet = nx.betweenness_centrality(G, weight='weight')

    st.subheader("3. Centrality")
    deg_df = pd.DataFrame.from_dict(deg, orient='index', columns=['Degree'])
    clo_df = pd.DataFrame.from_dict(clo, orient='index', columns=['Closeness'])
    bet_df = pd.DataFrame.from_dict(bet, orient='index', columns=['Betweenness'])

    st.markdown("**Top 5 Degree Centrality**")
    st.dataframe(deg_df.sort_values(by='Degree', ascending=False).head())
    st.markdown("**Top 5 Closeness Centrality**")
    st.dataframe(clo_df.sort_values(by='Closeness', ascending=False).head())
    st.markdown("**Top 5 Betweenness Centrality**")
    st.dataframe(bet_df.sort_values(by='Betweenness', ascending=False).head())

    # ========== KOMUNITAS LOUVAIN ==========
    st.subheader("4. Deteksi Komunitas Louvain")
    partition = community_louvain.best_partition(G, weight='weight')
    komunitas_df = pd.DataFrame(list(partition.items()), columns=['artist', 'community'])
    modularity = community_louvain.modularity(partition, G, weight='weight')
    st.markdown(f"**Modularitas Jaringan:** {modularity:.4f}")

    jumlah_komunitas = len(set(partition.values()))
    st.markdown(f"**Jumlah Komunitas:** {jumlah_komunitas}")
    st.dataframe(komunitas_df.head())

    # ========== VISUALISASI KOMUNITAS ==========
    pos = nx.kamada_kawai_layout(G)
    num_communities = len(set(partition.values()))
    colors = [partition[node] for node in G.nodes()]
    cmap = cm.get_cmap('nipy_spectral', num_communities)

    plt.figure(figsize=(20, 16))
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cmap, node_size=35, alpha=0.85)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
    plt.title("Visualisasi Komunitas Louvain dalam Jaringan Kolaborasi Musisi")
    plt.axis('off')
    st.pyplot(plt)

    # ========== KOMUNITAS TERBESAR ==========
    st.subheader("5. Komunitas Terbesar")
    jumlah_per_komunitas = komunitas_df['community'].value_counts()
    komunitas_terbesar_id = jumlah_per_komunitas.idxmax()
    anggota_terbesar = komunitas_df[komunitas_df['community'] == komunitas_terbesar_id]['artist'].sort_values()
    st.markdown(f"**Komunitas {komunitas_terbesar_id} (Jumlah: {len(anggota_terbesar)})**")
    st.dataframe(anggota_terbesar.reset_index(drop=True))

    # ========== 5 KOMUNITAS TERBESAR ==========
    st.subheader("6. Visualisasi 5 Komunitas Terbesar")
    top5 = Counter(partition.values()).most_common(5)
    for com_id, _ in top5:
        anggota = [node for node, com in partition.items() if com == com_id]
        subg = G.subgraph(anggota)
        pos_sub = nx.spring_layout(subg, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw(subg, pos_sub, with_labels=True, node_size=50, node_color='lightgreen', font_size=6, edge_color='gray')
        plt.title(f"Komunitas {com_id} (Top 5 Terbesar)")
        plt.axis('off')
        st.pyplot(plt)
