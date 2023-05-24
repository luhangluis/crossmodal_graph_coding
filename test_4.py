import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加节点，节点值未知数据结构
G.add_node(1, value='value1', pos=(1.0, 3.4), label="1")
G.add_node(2, value=42, pos=(1.6, 5.6), label="2")
G.add_node(3, value=[1, 2, 3], pos=(2.4, 7.8), label="3")

# 遍历所有节点
for node in G.nodes():
    # 获取节点的当前值
    current_value = G.nodes[node]['value']

    # 在此处进行特定操作，例如更新值、打印等
    # 这里只是打印当前值的示例
    print(f"Node {node}: {current_value}")
