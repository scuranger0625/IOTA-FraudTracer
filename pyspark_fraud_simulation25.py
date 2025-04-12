import os, sys, time, random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# 初始化 Spark
spark = SparkSession.builder \
    .appName("IOTA DAG with Graph Algorithms") \
    .master("local[*]") \
    .getOrCreate()

print(f"[INFO] Spark 執行緒數: {spark.sparkContext.defaultParallelism}")

# 模擬參數
FRAUD_COUNT = 50
NUM_TRANSACTIONS = 20000
NUM_ACCOUNTS = 10000
accounts = [f"U{i}" for i in range(1, NUM_ACCOUNTS + 1)]
fraud_senders = random.sample(accounts, k=FRAUD_COUNT)

# 分配銀行
bank_names = [f"BANK_{i+1}" for i in range(5)]
account_bank = {acc: random.choice(bank_names) for acc in accounts}

# 生成交易資料
data = []
for _ in range(NUM_TRANSACTIONS):
    sender = random.choice(accounts)
    is_fraud = sender in fraud_senders
    cross_bank_prob = 0.8 if is_fraud else 0.5

    if random.random() < cross_bank_prob:
        receiver = random.choice([
            acc for acc in accounts if account_bank[acc] != account_bank[sender] and acc != sender
        ])
    else:
        same_bank_accounts = [acc for acc in accounts if account_bank[acc] == account_bank[sender] and acc != sender]
        receiver = random.choice(same_bank_accounts) if same_bank_accounts else random.choice(accounts)

    transactions = random.randint(1, 50)
    balance = random.randint(0, 1000)
    risk_score = random.random()
    timestamp = random.randint(1000, 100000)
    label = 1 if sender in fraud_senders else 0
    data.append((sender, receiver, transactions, balance, risk_score, timestamp, label))

columns = ["sender", "receiver", "transactions", "balance", "risk_score", "timestamp", "label"]
df = spark.createDataFrame(data, columns)

assembler = VectorAssembler(inputCols=["transactions", "balance", "risk_score"], outputCol="features")
df = assembler.transform(df)
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
model = rf.fit(df)
predictions = model.transform(df).select("sender", "receiver", "label", "transactions", "timestamp")
pdf = predictions.toPandas().sort_values("timestamp")

# 建構 DAG
G = nx.DiGraph()
for _, row in pdf.iterrows():
    s, r = row["sender"], row["receiver"]
    if s != r:
        if r in G.nodes and s in G.nodes:
            if nx.has_path(G, r, s):
                continue
        weight = row["transactions"] / (1 + row["timestamp"])  # Mana 模擬
        G.add_edge(s, r, fraud=row["label"], weight=weight)

true_positives = set(fraud_senders)

# Union-Find
parent = {}
def find(x):
    if x not in parent: parent[x] = x
    if parent[x] != x: parent[x] = find(parent[x]) # <- 這裡使用了路徑壓縮
    return parent[x]
def union(x, y): parent[find(y)] = find(x)

start_uf = time.perf_counter()
for u, v in G.edges(): union(u, v)
uf_build = time.perf_counter() - start_uf
start_uf_q = time.perf_counter()
fraud_roots = set(find(f) for f in fraud_senders)
uf_found = {n for n in G.nodes() if find(n) in fraud_roots}
uf_query = time.perf_counter() - start_uf_q

# DFS
start_dfs = time.perf_counter()
dfs_visited = set()
def dfs(v):
    for n in G.successors(v):
        if n not in dfs_visited:
            dfs_visited.add(n)
            dfs(n)
for v in G.nodes(): dfs(v)
dfs_time = time.perf_counter() - start_dfs

# BFS
start_bfs = time.perf_counter()
bfs_visited = set()
for v in G.nodes():
    queue = [v]
    while queue:
        n = queue.pop(0)
        if n not in bfs_visited:
            bfs_visited.add(n)
            queue.extend([x for x in G.successors(n) if x not in bfs_visited])
bfs_time = time.perf_counter() - start_bfs

# Dijkstra
start_dijkstra = time.perf_counter()
dijkstra_found = set()
for n in G.nodes():
    dijkstra_found.update(nx.single_source_dijkstra_path_length(G, n, weight="weight").keys())
dijkstra_time = time.perf_counter() - start_dijkstra

# Bellman-Ford
start_bf = time.perf_counter()
bellman_found = set()
for node in G.nodes():
    try:
        lengths = nx.single_source_bellman_ford_path_length(G, node, weight="weight")
        bellman_found.update(lengths.keys())
    except:
        continue
bf_time = time.perf_counter() - start_bf

# Kruskal（最小生成樹）
start_kruskal = time.perf_counter()
UG = G.to_undirected()
kruskal_edges = list(nx.minimum_spanning_edges(UG, data=True))
kruskal_nodes = set()
for u, v, _ in kruskal_edges:
    kruskal_nodes.update([u, v])
kruskal_time = time.perf_counter() - start_kruskal

# 評估函式
def evaluate(found):
    tp = len(found & true_positives)
    fp = len(found - true_positives)
    fn = len(true_positives - found)
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return tp, len(found), recall, precision, f1

# 整合資料
methods = {
    "Union-Find": uf_found,
    "DFS": dfs_visited,
    "BFS": bfs_visited,
    "Dijkstra": dijkstra_found,
    "Bellman-Ford": bellman_found,
    "Kruskal": kruskal_nodes
}

runtime_dict = {
    "Union-Find": uf_build + uf_query,
    "DFS": dfs_time,
    "BFS": bfs_time,
    "Dijkstra": dijkstra_time,
    "Bellman-Ford": bf_time,
    "Kruskal": kruskal_time
}

metrics_df = pd.DataFrame([
    {
        "method": m,
        "found": len(s),
        "TP": len(s & true_positives),
        "recall": evaluate(s)[2],
        "precision": evaluate(s)[3],
        "F1": evaluate(s)[4],
        "runtime_sec": runtime_dict.get(m, 0)
    } for m, s in methods.items()
])
metrics_df["rank_F1"] = metrics_df["F1"].rank(ascending=False, method="min").astype(int)
metrics_df["rank_runtime"] = metrics_df["runtime_sec"].rank(ascending=True, method="min").astype(int)
metrics_df = metrics_df.sort_values(by="rank_F1").reset_index(drop=True)
metrics_df.index += 1

runtime_df = pd.DataFrame([
    {"method": "Union-Find (build)", "runtime_sec": uf_build},
    {"method": "Union-Find (query)", "runtime_sec": uf_query},
    {"method": "DFS", "runtime_sec": dfs_time},
    {"method": "BFS", "runtime_sec": bfs_time},
    {"method": "Dijkstra", "runtime_sec": dijkstra_time},
    {"method": "Bellman-Ford", "runtime_sec": bf_time},
    {"method": "Kruskal", "runtime_sec": kruskal_time}
]).sort_values(by="runtime_sec").reset_index(drop=True)
runtime_df.index += 1

print("\n[INFO] 評估指標：")
print(tabulate(metrics_df, headers="keys", tablefmt="grid"))
print("\n[INFO] 運行時間（秒）：")
print(tabulate(runtime_df, headers="keys", tablefmt="grid"))

# DAG 繪圖（依 timestamp 與銀行層級分布，改善時間感）
try:
    layer_index = {bank: i for i, bank in enumerate(bank_names)}
    pos = {}
    for node in G.nodes():
        # 找出該節點所有相關的最早 timestamp（不只 sender）
        timestamps = pdf[(pdf["sender"] == node) | (pdf["receiver"] == node)]["timestamp"]
        ts = timestamps.min() if not timestamps.empty else 0
        bank = account_bank.get(node, "BANK_1")
        layer = layer_index.get(bank, 0)
        pos[node] = (ts, -layer)

    # node & edge 顏色（fraud_senders 有些可能沒出現在 G.nodes）
    node_colors = ['orange' if n in fraud_senders else 'skyblue' for n in G.nodes()]
    edge_colors = ['red' if G[u][v]['fraud'] == 1 else 'gray' for u, v in G.edges()]

    # 繪圖設定
    plt.figure(figsize=(60, 30))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=8,
        node_color=node_colors,
        edge_color=edge_colors,
        arrows=True
    )

    legend = [
        mpatches.Patch(color='orange', label='Fraud Account'),
        mpatches.Patch(color='skyblue', label='Normal Account'),
        mpatches.Patch(color='red', label='Fraud Transaction'),
        mpatches.Patch(color='gray', label='Normal Transaction')
    ]
    plt.legend(handles=legend, fontsize=10)
    plt.title("IOTA-Inspired DAG (Layered by Bank, Ordered by Time)", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("dag_with_kruskal.png", dpi=300)
    if sys.platform.startswith("win"):
        os.startfile("dag_with_kruskal.png")
except nx.NetworkXUnfeasible:
    print("[WARN] Graph contains cycles. Skipped visualization.")

spark.stop()
