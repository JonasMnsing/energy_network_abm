import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union

################################################################################################################################################

def setup_city(N : int, x_length : float, y_length : float) -> np.array:
    """
    Init a city of x_length and y_length with N Agents randomly placed
    """

    city = np.zeros(shape=(N,3))

    for i in range(N):

        x_pos   = np.random.uniform(low=0,high=x_length)
        y_pos   = np.random.uniform(low=0,high=y_length)
        
        city[i][0]  = x_pos
        city[i][1]  = y_pos

    return city

def setup_producer(city : np.array, p_producer : float) -> np.array:
    """
    Make some agents in city to be producers based on probability p_producer
    """

    for i in range(len(city)):

        rand_var = np.random.rand()

        if rand_var <= p_producer:

            city[i][2] = 1.

    return city

def setup_graph(city : np.array) -> nx.Graph:
    """
    Init a networkx graph object based on city with a producer attribute
    """

    # Init a producer feature dictonary
    dic_p = {}

    # Init a Graph Object
    G = nx.Graph()

    # Cover each agent
    for i in range(len(city)):

        # And add node with node-name = agent-number and node-pos = (x,y)-position
        G.add_node(i, pos=(city[i][0],city[i][1]))

        # Generate key value pair with key = agent-number and value = producer bool
        dic_p[i] = city[i,2]

    # Add a producer attribute to grapg G based on producer dictonary
    nx.set_node_attributes(G=G, values=dic_p, name="producer")

    return G

def setup_graph_power_plant(city : np.array) -> nx.Graph:
    """
    Init a networkx graph object based on city for power plant graph
    """

    # Init a Graph Object
    G = nx.Graph()

    # Add power plant node with name=-1 and pos=(x,y)=(0,0)
    G.add_node(-1, pos=(0,0))

    # Cover each agent
    for i in range(len(city)):

        # And add node with node-name = agent-number and node-pos = (x,y)-position 
        G.add_node(i, pos=(city[i][0],city[i][1]))

    return G

def connect_nodes(G : nx.Graph, max_distance : float) -> nx.Graph:
    """
    Connect nodes with each other in networkx graph obejct G based on maximum allowed distance max_distance
    """

    # Move through each node
    for i in range(len(G.nodes)):
            
        # get first position
        pos_i = G.nodes[i]['pos']

        # move through each node
        for j in range(len(G.nodes)):

            # Dont cover the same node and one node should be producer
            if (i != j) and (G.nodes[i]['producer'] == 1 or G.nodes[j]['producer'] == 1):

                # Get second position
                pos_j       = G.nodes[j]['pos']

                # Measure distance
                distance    = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)

                # If in maximum range
                if distance <= max_distance:

                    # Connect i and j by adding an edge
                    G.add_edge(i,j)
    
    return G

def connect_power_plant(G : nx.Graph) -> nx.Graph:
    """
    Connect all nodes in G to power plant
    """

    for i in range(len(G.nodes)-1):
            
        G.add_edge(-1,i)
    
    return G


def init_graphs(N : int, x_length : float, y_length : float, p_producer : float, max_distance : float) -> Union[nx.Graph,nx.Graph]:
    """
    Call all functions to init Network Graph and Power Plant Graph based on city and connection parameter
    """
    city    = setup_city(N=N,x_length=x_length,y_length=y_length)
    city    = setup_producer(city=city, p_producer=p_producer)
    G1      = setup_graph(city=city)
    G1      = connect_nodes(G=G1, max_distance=max_distance)
    G2      = setup_graph_power_plant(city=city)
    G2      = connect_power_plant(G=G2)

    return G1, G2

################################################################################################################################################

def add_demand_profile(G : nx.Graph, demand_profile : list) -> nx.graph:
    """
    Add demand profile attribute to graph
    """

    demand_dict = {i:demand_profile.copy() for i in G.nodes}

    nx.set_node_attributes(G=G, values=demand_dict, name="demand profile") 

    return G

def add_production_profile(G : nx.Graph, production_profile : list) -> nx.graph:
    """
    Add production profile attribute to graph
    """
    prod_dict = {}

    for node in G.nodes:

        # If node is producer add profile otherwise list of zeros
        if G.nodes[node]['producer'] == 1:
            prod_dict[node] = production_profile.copy()
        else:
            prod_dict[node] = [0 for j in range(len(production_profile.copy()))]

    nx.set_node_attributes(G=G, values=prod_dict, name="production profile") 

    return G

def satisfy_own_demand(G : nx.Graph, time_step : int) -> nx.graph:
    """
    For each agent check if producer and compansate demand as much as possible
    """

    for i in range(len(G.nodes)):

        if ((G.nodes[i]['producer'] == 1)
            and (G.nodes[i]['production profile'][time_step] > 0)
            and (G.nodes[i]['demand profile'][time_step] > 0)):

            difference = G.nodes[i]['production profile'][time_step] - G.nodes[i]['demand profile'][time_step]

            if difference > 0:

                G.nodes[i]['demand profile'][time_step]     = 0
                G.nodes[i]['production profile'][time_step] = difference
            
            else:

                G.nodes[i]['demand profile'][time_step]     -= G.nodes[i]['production profile'][time_step]
                G.nodes[i]['production profile'][time_step] = 0

    return G 

def satisfy_demand(G : nx.Graph, time_step : int):
    """
    For each agent if demand left, ask adjacent producers for compensation
    """
    for i in range(len(G.nodes)):

        if (G.nodes[i]['demand profile'][time_step] > 0):

            # Get each neighbor of i
            for neighbor in G.neighbors(i):

                if ((G.nodes[neighbor]['producer'] == 1) and (G.nodes[neighbor]['production profile'][time_step] > 0)):

                    difference = G.nodes[neighbor]['production profile'][time_step] - G.nodes[i]['demand profile'][time_step]

                    if difference > 0:

                        G.nodes[i]['demand profile'][time_step]             = 0
                        G.nodes[neighbor]['production profile'][time_step]  = difference
                    
                    else:

                        G.nodes[i]['demand profile'][time_step]             -= G.nodes[neighbor]['production profile'][time_step]
                        G.nodes[neighbor]['production profile'][time_step]  = 0

                if G.nodes[i]['demand profile'][time_step] == 0:

                    break
         
    return G

################################################################################################################################################

def draw_graph_obj(G : nx.Graph, path : str, node_size=100, figsize=(6,6), transparent=True) -> None:
    """
    Draw networkx graph G and save to path
    """

    # Get position and producer attribute dict
    pos         = nx.get_node_attributes(G=G, name='pos')
    producers   = nx.get_node_attributes(G=G, name='producer')

    # Init Matplotlib figure object
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot()

    # Generate a list of two colors representing producer -> 1/0
    colors = []

    for key in producers.keys():

        if producers[key] == 1.:
            colors.append('#A22A10')
        else:
            colors.append('#004d99')

    # Call draw function and save to path
    nx.draw(G=G, pos=pos,ax=ax, node_size=node_size, node_color=np.array(colors))
    fig.savefig(path, transparent=transparent, bbox_inches='tight')
    plt.close(fig)


def draw_power_plant_graph(G : nx.Graph, path : str, node_size=100, figsize=(6,6), transparent=True) -> None:
    """
    Draw power plant networkx graph G and save to path
    """

    pos = nx.get_node_attributes(G=G, name='pos')
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot()
    
    colors = []

    for i in range(len(G.nodes)):

        if i == 0:
            colors.append('#A22A10')
        else:
            colors.append('#004d99')
    
    nx.draw(G=G, pos=pos,ax=ax, node_size=node_size, node_color=np.array(colors))
    fig.savefig(path, transparent=transparent, bbox_inches='tight')
    plt.close(fig)  

def say_hello():
    print("Hello!")