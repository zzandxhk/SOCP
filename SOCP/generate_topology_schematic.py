# This script reproduces the topology schematic from manually specified branch sets.
# It is a schematic drawing script, not the GA/ADMM planning implementation.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Reset plot scaling to keep text at a readable size.
plt.rcdefaults()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# ==============================================================================
# IEEE 33-bus network topology comparison.
# ==============================================================================
# Build the 33-bus radial topology.
G = nx.Graph()
# Main feeder.
main_branch = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18)]
# Lateral branches.
branch_1 = [(2,19),(19,20),(20,21),(21,22)]
branch_2 = [(3,23),(23,24),(24,25)]
branch_3 = [(6,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33)]
base_edges = main_branch + branch_1 + branch_2 + branch_3
G.add_edges_from(base_edges)

# Approximate node positions for a recognizable IEEE 33-bus schematic.
pos = {1:(0,0), 2:(1,0), 3:(2,0), 4:(3,0), 5:(4,0), 6:(5,0), 7:(6,0), 8:(7,0), 9:(8,0), 10:(9,0),
       11:(10,0), 12:(11,0), 13:(12,0), 14:(13,0), 15:(14,0), 16:(15,0), 17:(16,0), 18:(17,0),
       19:(1,-1), 20:(1,-2), 21:(1,-3), 22:(1,-4),
       23:(2,1), 24:(2,2), 25:(2,3),
       26:(5,-1), 27:(5,-2), 28:(5,-3), 29:(5,-4), 30:(5,-5), 31:(5,-6), 32:(5,-7), 33:(5,-8)}

# Tie switches at candidate SOP locations.
tie_lines = [(12,22), (18,33), (25,29)]

# Draw a side-by-side comparison.
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6.5))

# Panel (a): conventional plan without SOPs and with extensive line upgrades.
ax1 = axes1[0]
nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=150, node_color='lightgray', edgecolors='black')
nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)
# Draw existing lines.
nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=base_edges, width=1.5, edge_color='gray')
# Highlight upgraded lines in the conventional plan.
upgraded_edges_trad = [(2,3), (3,4), (4,5), (5,6), (6,7), (2,19), (19,20), (6,26), (26,27)]
nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=upgraded_edges_trad, width=4.0, edge_color='red', label='Upgraded Line (Heavy Invest)')
# Draw normally open tie switches as dashed lines.
nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=tie_lines, width=2.0, edge_color='blue', style='dashed', label='Normally Open Tie-switch')

ax1.set_title('(a) Traditional Centralized Planning \n(Extensive Line Upgrades, No SOPs)', fontsize=14, pad=10)
ax1.axis('off')
ax1.legend(loc='lower left', fontsize=10)

# Panel (b): proposed plan with SOPs and fewer line upgrades.
ax2 = axes1[1]
nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=150, node_color='lightgray', edgecolors='black')
nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=base_edges, width=1.5, edge_color='gray')
# Highlight the two upgraded lines in the proposed plan.
upgraded_edges_ours = [(2,3), (6,26)]
nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=upgraded_edges_ours, width=4.0, edge_color='red', label='Upgraded Line (Light Invest)')
# Show the installed SOPs at selected locations.
nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=[(12,22), (25,29)], width=4.0, edge_color='green', label='Installed SOP (Active Routing)')
# Draw the remaining normally open tie switch.
nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=[(18,33)], width=2.0, edge_color='blue', style='dashed')

# Highlight nodes connected to installed SOPs.
sop_nodes = [12, 22, 25, 29]
nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=sop_nodes, node_size=250, node_color='gold', edgecolors='black', node_shape='s')

ax2.set_title('(b) Proposed Privacy-Aware Bi-level Planning \n(SOP Integration, Minimal Upgrades)', fontsize=14, pad=10)
ax2.axis('off')
ax2.legend(loc='lower left', fontsize=10)

plt.tight_layout()
output = Path(__file__).resolve().parents[1] / '\u8bba\u6587' / 'fig' / 'Fig_4_4_Topology.png'
plt.savefig(output, dpi=300, bbox_inches='tight')
