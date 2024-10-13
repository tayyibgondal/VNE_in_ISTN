import heapq
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from itertools import islice
from config_for_integrated_network import *
import os
'''-----------------------------------------------------PLOTTING FUNCTIONS----------------------------------------------------'''
def plot_saved_disruptions():
    global disruptions_saved
    fig = plt.figure()
    x_values = [(i+1) for i in range(num_of_iter)]
    for key, value in disruptions_saved.items():
        y_values = value[1:]
        plt.plot(x_values, y_values, label=f"{key} requests")
    

    # Set labels and title
    plt.xlabel('Topology Changes')
    plt.ylabel('Percent of Total Requests Saved')
    plt.title('Requests Saved from Unnecessary Disruptions')

    # Add legend
    plt.legend(loc='upper right')
    pass

def plot_spectrum_occupancy():
    # plot overall spectrum occupancy(for all links)
    global spectrum_occupancy
    fig = plt.figure()
    print('spectrum_occupancy', spectrum_occupancy)
    for key, value in spectrum_occupancy.items():
        print(key, value)
        y_values, x_values = value
        # x_values = [x for x, _ in x_values_tuples]  # Unpack x values from tuples
        plt.plot(x_values, y_values, label=f"{key} requests")  # Use f-string for label


    # Set labels and title
    plt.xlabel('Time (min)')
    plt.ylabel('Occupied Spectrum (MBps)')
    plt.title('Spectrum Occupancy')
    # plt.ylim(0, 1)

    # Add legend
    plt.legend(loc='upper right')

    print('total spectrum occupancy', spectrum_occupancy)

def plot_ground_spectrum_occupancy():
    # plot ground spectrum occupancy
    global ground_spectrum_occupancy
    fig = plt.figure()
    
    for key, value in ground_spectrum_occupancy.items():
        y_values, x_values = value
        # x_values = [x for x, _ in x_values_tuples]  # Unpack x values from tuples
        plt.plot(x_values, y_values, label=f"{key} requests")  # Use f-string for label


    # Set labels and title
    plt.xlabel('Time (min)')
    plt.ylabel('Occupied Spectrum (MBps)')
    plt.title('Ground Spectrum Occupancy')
    # plt.ylim(0, 1)

    # Add legend
    plt.legend(loc='upper right')

    print('ground_spectrum_occupancy', ground_spectrum_occupancy)   

def plot_sat_spectrum_occupancy():
    # plot satellite spectrum occupancy
    global sat_spectrum_occupancy
    fig = plt.figure()
    
    for key, value in sat_spectrum_occupancy.items():
        y_values, x_values_tuples = value
        x_values = x_values_tuples # Unpack x values from tuples
        plt.plot(x_values, y_values, label=f"{key} requests")  # Use f-string for label


    # Set labels and title
    plt.xlabel('Time (min)')
    plt.ylabel('Occupied Spectrum (MBps)')
    plt.title('Satellite Spectrum Occupancy')
    # plt.ylim(0, 1)

    # Add legend
    plt.legend(loc='upper right')

    print('sat_spectrum_occupancy', sat_spectrum_occupancy) 

def plot_gs_spectrum_occupancy():
    global gs_spectrum_occupancy
    fig = plt.figure()

    for key, value in gs_spectrum_occupancy.items():
        y_values, x_values = value
        # x_values = [x for x, _ in x_values_tuples]  # Unpack x values from tuples
        plt.plot(x_values, y_values, label=f"{key} requests")  # Use f-string for label


    # Set labels and title
    plt.xlabel('Time (min)')
    plt.ylabel('Occupied Spectrum (MBps)')
    plt.title('Gateway-Satellite Spectrum Occupancy')
    # plt.ylim(0, 1)

    # Add legend
    plt.legend(loc='upper right')

    print('gs_spectrum_occupancy', gs_spectrum_occupancy) 

'''-----------------------------------------------------HELPER FUNCTIONS FOR RUNNING THE SIMULATION------------------------------------'''

def calculate_ellipse_points(cx, cy, a, b, num_points): # (center_x_coord, center_y_coord, major axis, minor axis, num of points)
    # determine the points along the satellites' elliptical path
    points = []
    angle_increment = 2 * math.pi / num_points
    for i in range(num_points):
        theta = i * angle_increment
        x = cx + a * math.cos(theta)
        y = cy + b * math.sin(theta)
        points.append((x, y))
    return points

def path_exists(topology, path, source, destination):
     # returns true if the path provided exists in the network topology at some future point in time
    # print("all simple paths starting(path exists fucntion)")
    paths = nx.all_simple_paths(topology, source, destination) # Find all simple paths between the source and destination nodes
    # print("all simple paths ending(path exists fucntion)")
    if path in paths: # check if the path chosen will exist in this version of the topology
        return True
    return False

def path_viable(graph, path, source, destination):
    og_topology_copy = copy.deepcopy(graph)
    move_satellites(og_topology_copy)
    if path_exists(og_topology_copy, path, source, destination) == False:
        return False
    else:
        return True
    
# Check if the path exists in the graph
def path_exists_in_graph(graph, path):
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True

def store_network_changes(graph):
    global num_of_iter
    topologies = []
    og_topology_copy = copy.deepcopy(graph)
    for i in range(num_of_iter):
        move_satellites(og_topology_copy)
        topologies.append(og_topology_copy)
    return topologies

def set_gateway_positions(gateway_positions, num_of_gs):
    city_data = [
    ("Anchorage, Alaska", 61.2173, -149.8631),
    ("Baxley, Georgia", 31.7642, -82.3519),
    ("Beekmantown, New York", 44.7289, -73.5383),
    ("Bellingham, Washington", 48.7519, -122.4787),
    ("Blountsville, Alabama", 34.0817, -86.5917),
    ("Boca Chica, Texas", 25.9969, -97.1632),
    ("Brewster, Washington", 48.093, -119.7934),
    ("Broadview, Illinois", 41.8584, -87.8361),
    ("Butte, Montana", 45.9412, -112.5868),
    ("Cass County, North Dakota", 47.0294, -97.2035),
    ("Charleston, Oregon", 43.3726, -124.2365),
    ("Colburn, Idaho", 48.4218, -116.7359),
    ("Conrad, Montana", 48.1718, -111.935),
    ("Dumas, Texas", 35.8614, -101.9634),
    ("Elbert, Colorado", 39.2423, -104.528),
    ("Evanston, Wyoming", 41.2683, -110.9632),
    ("Fairbanks, Alaska", 64.8378, -147.7164),
    ("Fort Lauderdale, Florida", 26.1224, -80.1373),
    ("Frederick, Maryland", 39.4142, -77.4105),
    ("Gaffney, South Carolina", 35.0729, -81.6487),
    # ... Add more cities here ...
    ]

    for i, (city, latitude, longitude) in enumerate(city_data[:num_of_gs], start=1):
        node_name = f'Node{i}'
        gateway_positions[node_name] = (latitude, longitude)

def calculate_distance(node_1, node_2):
    '''
    calculates distance between two nodes.
    inputs: node1, node2
    output: distance
    '''
    position1 = node_positions[node_1]
    position2 = node_positions[node_2]
    # Calculate the Euclidean distance between the two positions
    distance = math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
    return distance

def calculate_propogation_delay_between_two_nodes(node1, node2):
    '''
    calculates propagation delay between two nodes.
    inputs: node1, node2
    output: propagation delay
    '''
    distance = calculate_distance(node1, node2)
    delay = distance / 10
    if node1.startswith('Node') and node2.startswith('Node'):
        delay += delay / 10
    elif (node1.startswith('Node') and node2.startswith('Satellite')) or (node1.startswith('Satellite') and node2.startswith('Node')):
        delay += delay / 5
    else:
        delay -= delay / 10

    return delay

def calaculate_propagation_delay_for_path(path):
    '''
    calculates propagation delay for complete path
    inputs: paths list of form ['Node1', 'Node2', 'Satellite1']
    output: propagation delay
    '''
    total_delay = 0
    for i in range(len(path) - 1):
        total_delay += calculate_propogation_delay_between_two_nodes(path[i], path[i + 1])
        
    return total_delay

def update_links():
    global communication_range, total_gr_sat_capacity, total_capacity
    # Iterate over all pairs of satellite and terrestrial nodes (sat-gs links)
    for node1 in G.nodes():
        if node1.startswith('Satellite'):
            for node2 in nodes:
                if (node2.startswith('Node')):
                    distance = calculate_distance(node1, node2)
                    # print("distance bw",node1," and ",node2," : ",distance)
                    # Check if the distance is within the communication range
                    if distance <= communication_range:
                        # If the edge does not exist, add it
                        if not G.has_edge(node1, node2):
                            # print("adding a ground-air link")
                            G.add_edge(node1, node2)
                            G[node1][node2]['capacity'] = S2S_AND_G2S_CAPACITY
                            total_gr_sat_capacity += S2S_AND_G2S_CAPACITY
                            total_capacity += S2S_AND_G2S_CAPACITY
                    else:
                        # If the edge exists, remove it
                        if G.has_edge(node1, node2):
                            G.remove_edge(node1, node2)
                            total_capacity -= S2S_AND_G2S_CAPACITY
                            total_gr_sat_capacity -= S2S_AND_G2S_CAPACITY
                            
def generate_requests(graph, num_of_requests):
    global terrestrial_nodes, satellite_nodes
    requests = {}
    id = 1
    # print('num_of_requests',num_of_requests)
    while len(requests) < num_of_requests:
        source = random.choice(nodes)
        dest = random.choice(nodes)

        while source == dest:
            dest = random.choice(nodes)

        req = (source, dest, id)
        requests[req] = {'required_rate': random.randint(REQUIRED_RATE_MINIMUM, REQUIRED_RATE_MAXIMUM)}
        id += 1
    return requests  # this dict is of the form: {(node1, node2): {id:1, required_rate: 100}}
                     # keys are tuples, values are dictionaries

def find_alternate_path(graph, source, destination, required_capacity):
    # Dijkstra's algorithm with capacity as the priority
    visited = set()
    priority_queue = [(0, source, [])]  # (cumulative_capacity, current_node, path_so_far)
    while priority_queue:
        cum_capacity, current_node, path = heapq.heappop(priority_queue)
        if current_node == destination:
            return path + [destination]
        if current_node in visited:
            continue
        visited.add(current_node)
        for neighbor in graph.neighbors(current_node):
            edge_capacity = graph[current_node][neighbor]['capacity']
            if edge_capacity < required_capacity:
                continue
            new_cum_capacity = min(cum_capacity, edge_capacity)
            new_path = path + [current_node]
            heapq.heappush(priority_queue, (new_cum_capacity, neighbor, new_path))
    return None

'''----------------------------------------------------------------VIRTUAL NETWORK EMBEDDING--------------------------------------------------'''

def embed_virtual_network(graph, requests, no_of_req, iteration):
    # print("in embed fxn")
    global num_points_along_orbit, total_delay_till_now, total_no_of_requests, all_request_statuses
    time_elapsed_in_one_step_along_orbit = LEO_period/num_points_along_orbit# period is LEO_period min, for, say, 20 steps along the orbit, covering 1/20th of the orbit takes LEO_period/20 min

    successful_in_this_iteration = 0
    failed_in_this_iteration = 0
    for req in requests:
        source, destination, id = req
        required = requests[req]['required_rate']
        embedded = False
        graph_for_embeddings = graph
        for _ in range(30):
            try:
                path = nx.shortest_path(graph_for_embeddings, source, destination)
            except:
                break
            request_propagation_delay = calaculate_propagation_delay_for_path(path)
            if all(graph.edges[e]['capacity'] >= required for e in zip(path, path[1:])) and request_propagation_delay <= max_allowed_latency:
                            # print("path viable")
                            # Deduct the capacity from the edges in the path
                            for e in zip(path, path[1:]):
                                graph.edges[e]['capacity'] -= required
                            all_request_statuses[req] = (True, path)
                            successful_in_this_iteration += 1
                            request_propagation_delay = calaculate_propagation_delay_for_path(path)
                            # print(f'==============================Request propagation delay for this path is: {request_propagation_delay}===================================')
                            total_delay_till_now += request_propagation_delay
                            total_no_of_requests += 1
                            embedded = True
                            break
            # remove this path from the graph
            graph_copy = copy.deepcopy(graph_for_embeddings)
            # print('path',path)
            # graph_copy.remove_edges_from(path)
            graph_copy.remove_edge(path[0], path[1])
            graph_for_embeddings = graph_copy
        if not embedded:
            all_request_statuses[req] = (False, [])
            failed_in_this_iteration += 1
            
    remaining_capacity_in_network = 0
    remaining_capacity_in_network_ground = 0
    remaining_capacity_in_network_satellite = 0
    remaining_capacity_in_network_ground_satellite = 0
    for edge in list(graph.edges):
        remaining_capacity_in_network += graph.edges[edge]['capacity']
        source, target = edge
        if source.startswith('Node') and target.startswith('Node'):
            remaining_capacity_in_network_ground += graph.edges[edge]['capacity']
        elif source.startswith('Satellite') and target.startswith('Satellite'):
            remaining_capacity_in_network_satellite += graph.edges[edge]['capacity']
        else:
            remaining_capacity_in_network_ground_satellite += graph.edges[edge]['capacity']

    successful_req = 0
    failed_req = 0
    for req, status in all_request_statuses.items():
        source, destination, id = req
        # required = requests[req]['required_rate']
        if status[0]:
            successful_req += 1
        else:
            failed_req += 1
    
    print('successful in this iteration:',successful_in_this_iteration)
    print('failed in this iteration:', failed_in_this_iteration)
    print('iteration: ',iteration,'no of reuests:', len(all_request_statuses.items()))
    print('total successful:',successful_req)
    print('total failed:', failed_req)
    total_reqs = successful_req + failed_req
    reqs_saved_from_disruption = total_reqs - len(requests)
    print('total reqs:',total_reqs)
    print('reqs saved from disruption:',reqs_saved_from_disruption)
    percentage_reqs_saved_from_disruption = (reqs_saved_from_disruption/total_reqs)
    # print('nodes:',len(graph.nodes))
    print()

    return (percentage_reqs_saved_from_disruption, remaining_capacity_in_network, remaining_capacity_in_network_ground, 
            remaining_capacity_in_network_satellite, remaining_capacity_in_network_ground_satellite)

'''----------------------------------------------------------------SATELLITE MOTION--------------------------------------------------'''

def move_satellites(graph):
    global satellite_nodes, satellite_path, current_satellite_positions, satellite_positions, node_positions
    for satellite in satellite_nodes:
            current_pos = graph.nodes[satellite]['pos']
            #print('current pos of ',satellite,' is ',current_pos)
            next_pos = satellite_path[(satellite_path.index(current_pos) + 1) %len(satellite_path)] # next position in orbit
            #print('next pos of ',satellite,' is ',next_pos)
            current_satellite_positions[satellite] = next_pos
            graph.nodes[satellite]['pos'] = next_pos  # Update the position of the node
    satellite_positions = list(current_satellite_positions.values())
    node_positions = {**gateway_positions, **current_satellite_positions}
    update_links() #tear down old ground to air links, and make new ones, as needed (done every time satellites move)

def simulate_satellite_motion(graph, num_iterations, traffic_requests, no_of_reqs, occ_in_this_run, saved_disruptions, times_list):
    global total_capacity, all_request_statuses, total_gr_sat_capacity, total_gr_capacity, total_sat_capacity, times_reembedded_needlessly, requests_reembedded_needlessly,num_points_along_orbit, satellite_positions, node_positions, current_pos, capacity_vs_noOfReqs, changing_top_plotted, p_capacity_vs_noOfRequests, g_capacity_vs_noOfRequests, s_capacity_vs_noOfRequests, gs_capacity_vs_noOfRequests, ground_capacity, satellite_capacity, avg_prop_delay, total_delay_till_now, total_no_of_requests, initial_status, current_available_capacity, current_available_capacity_ground, current_available_capacity_satellite, current_available_capacity_ground_satellite
    # print('req_statuses outsdie for loop',req_statuses)
    # print('traffic reqs outsdie for loop',traffic_requests)
    # The satellites move along a fixed path; at each snapshot, they have moved to the next position on this path
    time_elapsed_in_one_step_along_orbit = LEO_period/num_points_along_orbit# period is LEO_period min, for, say, 20 steps along the orbit, covering 1/20th of the orbit takes LEO_period/20 min
    topologies = store_network_changes(graph)
    for i in range(num_iterations):
        print("=================================================================================")
        print('iteration no: ', i)
        print("=================================================================================")
        requests_to_reembed = {} # dict containing all requests not embedded last time, or those whose embeddings are no longer valid
        for request in traffic_requests:
            if (all_request_statuses[request][0]) == True and (not path_exists_in_graph(topologies[i], all_request_statuses[request][1])):
                # Release capacities along the old path
                old_path = all_request_statuses[request][1]
                required_capacity = traffic_requests[request]['required_rate']
                for e in zip(old_path, old_path[1:]):
                    if graph.has_edge(*e):
                        graph.edges[e]['capacity'] += required_capacity
                # Mark the request as needing re-embedding
                # src, dest, id = req
                requests_to_reembed[request] = traffic_requests[request]
            elif all_request_statuses[request][0] == False:
                requests_to_reembed[request] = traffic_requests[request]

        # move_satellites(graph)
        
        print('requests_to_reembed',len(requests_to_reembed))
        if len(requests_to_reembed) > 0: 
            percent_disruptions_saved, current_available_capacity, current_available_capacity_ground, current_available_capacity_satellite, current_available_capacity_ground_satellite = embed_virtual_network(graph, requests_to_reembed, no_of_reqs, i+1) # embed some requests again after updating links
            saved_disruptions.append(percent_disruptions_saved)
            occ_in_this_run[0].append(round( (total_capacity - current_available_capacity)/total_capacity, 4))
            occ_in_this_run[1].append(round( (total_gr_capacity - current_available_capacity_ground)/total_gr_capacity, 4))
            occ_in_this_run[2].append(round( (total_sat_capacity - current_available_capacity_satellite)/total_sat_capacity, 4))
            occ_in_this_run[3].append(round( (total_gr_sat_capacity - current_available_capacity_ground_satellite)/total_gr_sat_capacity, 4))
        else:
            saved_disruptions.append(1)
            occ_in_this_run[0].append(occ_in_this_run[0][-1])
            occ_in_this_run[1].append(occ_in_this_run[1][-1])
            occ_in_this_run[2].append(occ_in_this_run[2][-1])
            occ_in_this_run[3].append(occ_in_this_run[3][-1])
        times_list.append( round((i+1), 2) ) #*time_elapsed_in_one_step_along_orbit
        move_satellites(graph)

    print('no of reqs', no_of_reqs, 'total_capacity:',total_capacity, 'ground_capacity:',total_gr_capacity, 'satellite_capacity:',total_sat_capacity, 'gs_capacity:',total_gr_sat_capacity)
    print('rem_capacity:',current_available_capacity, 'rem_g_capacity:',current_available_capacity_ground, 'rem_s_capacity:',current_available_capacity_satellite, 'rem_gs_capacity:',current_available_capacity_ground_satellite)
       
    print()
'''----------------------------------------------------------------RUNNING THE SIMULATION--------------------------------------------------'''

def reset_simulation():
    global G, reset_to, all_request_statuses, initial_total_capacity, initial_total_capacity_ground, initial_total_capacity_satellite, initial_total_capacity_ground_satellite, times_reembedded_needlessly, requests_reembedded_needlessly, satellite_nodes, satellite_path, current_satellite_positions, satellite_positions, total_capacity, total_capacity, total_gr_capacity, total_sat_capacity, total_gr_sat_capacity

    G = copy.deepcopy(reset_to)

    total_capacity = initial_total_capacity
    total_gr_capacity = initial_total_capacity_ground
    total_sat_capacity = initial_total_capacity_satellite
    total_gr_sat_capacity = initial_total_capacity_ground_satellite

    all_request_statuses = {}

    print('total cap at end of reset',total_capacity)
    print('total gr cap at end of reset',total_gr_capacity)
    print('total sat cap at end of reset',total_sat_capacity)
    print('total gr sat cap at end of reset',total_gr_sat_capacity)

def run_simulation(graph, iterations, no_of_reqs):
    global total_capacity, total_gr_capacity, total_sat_capacity, current_available_capacity, current_available_capacity_ground, current_available_capacity_satellite, current_available_capacity_ground_satellite
    traffic_requests = generate_requests(G,no_of_reqs) # Randomly generate traffic requests
    percent_disruptions_saved, current_available_capacity, current_available_capacity_ground, current_available_capacity_satellite, current_available_capacity_ground_satellite = embed_virtual_network(graph, traffic_requests, no_of_reqs, 0) # Embed the requests onto the original substrate network
    # initializing spectrum occupancy lists
    disruptions_saved_in_this_run = [0]
    spectrum_occupancy_in_this_run, times = [round((total_capacity - current_available_capacity)/total_capacity, 4)], [0]
    ground_spectrum_occupancy_in_this_run = [round((total_gr_capacity - current_available_capacity_ground)/total_gr_capacity, 4)]
    sat_spectrum_occupancy_in_this_run = [round((total_sat_capacity - current_available_capacity_satellite)/total_sat_capacity, 4)]
    gs_spectrum_occupancy_in_this_run = [round((total_gr_sat_capacity - current_available_capacity_ground_satellite)/total_gr_sat_capacity, 4)]
    occupancies_in_this_run = (spectrum_occupancy_in_this_run, ground_spectrum_occupancy_in_this_run, sat_spectrum_occupancy_in_this_run, gs_spectrum_occupancy_in_this_run)
    simulate_satellite_motion(graph, iterations, traffic_requests, no_of_reqs, occupancies_in_this_run, disruptions_saved_in_this_run, times)
    print('disruptions_saved_in_this_run',disruptions_saved_in_this_run)
    return ( disruptions_saved_in_this_run, (spectrum_occupancy_in_this_run, times), (ground_spectrum_occupancy_in_this_run,times), (sat_spectrum_occupancy_in_this_run,times), (gs_spectrum_occupancy_in_this_run,times))

def reset_simulation():
    global G, reset_to, total_capacity, total_gr_capacity, total_sat_capacity, total_gr_sat_capacity, all_request_statuses
    
    # Reset the graph topology
    G = copy.deepcopy(reset_to)
    
    # Reset capacity to initial values
    total_capacity = initial_total_capacity
    total_gr_capacity = initial_total_capacity_ground
    total_sat_capacity = initial_total_capacity_satellite
    total_gr_sat_capacity = initial_total_capacity_ground_satellite
    
    # Reset request statuses
    all_request_statuses = {}

    # Debugging print statements
    print('Total capacity reset:', total_capacity)
    print('Ground capacity reset:', total_gr_capacity)
    print('Satellite capacity reset:', total_sat_capacity)
    print('Satellite-ground capacity reset:', total_gr_sat_capacity)

'''------------------------------------------------------------------MAIN-----------------------------------------------------------------'''
if __name__ == "__main__":
    # Initialize nodes (ground and satellite nodes)
    terrestrial_nodes = ['Node' + str(i) for i in range(start_value_node, end_value_node)]
    satellite_nodes = ['Satellite' + str(i) for i in range(start_value_satellite, end_value_satellite)]
    nodes = terrestrial_nodes + satellite_nodes
    num_of_nodes = len(nodes)

    # Set gateway positions (ground nodes)
    gateway_positions = {}
    set_gateway_positions(gateway_positions, len(terrestrial_nodes))

    # Define satellite path (elliptical orbit)
    satellite_path = calculate_ellipse_points(41.7, -90.7, 168.26, 84.13, num_points_along_orbit)
    current_satellite_positions = {}  # Dictionary to store current satellite positions
    for satellite in satellite_nodes:
        current_satellite_positions[satellite] = satellite_path[satellite_nodes.index(satellite)]

    # Combine positions of satellites and ground nodes (gateways)
    satellite_positions = list(current_satellite_positions.values())
    node_positions = {**gateway_positions, **current_satellite_positions}

    # Create the network graph
    G = nx.Graph()

    # Add nodes to the graph with their positions
    for i in range(num_of_nodes):
        G.add_node(nodes[i], pos=list(node_positions.values())[i])

    # Initial topology: fully connect all gateways (ground nodes)
    all_possible_gateway_edges = [(gateway1, gateway2) for idx, gateway1 in enumerate(terrestrial_nodes) for gateway2 in terrestrial_nodes[idx + 1:]]
    gateway_edges = random.sample(all_possible_gateway_edges, GROUND_EDGES)
    G.add_edges_from(gateway_edges)

    # Add consecutive edges between satellites (satellite-to-satellite links)
    satellite_edges = [(satellite1, satellite2) for idx, satellite1 in enumerate(satellite_nodes) for satellite2 in satellite_nodes[idx + 1: idx + 2]]
    G.add_edges_from(satellite_edges)

    # Initialize variables for tracking capacity and statuses
    all_request_statuses = {}
    total_capacity = 0
    total_sat_capacity = 0
    total_gr_capacity = 0
    total_gr_sat_capacity = 0

    current_available_capacity = 0
    current_available_capacity_ground = 0
    current_available_capacity_satellite = 0
    current_available_capacity_ground_satellite = 0

    # Assign initial capacities to ground-ground and satellite-ground links
    for edge in G.edges:
        source, target = edge
        if source.startswith('Node') and target.startswith('Node'):
            G[source][target]['capacity'] = GROUND_CAPACITY
            total_gr_capacity += GROUND_CAPACITY
            total_capacity += GROUND_CAPACITY
        else:
            G[source][target]['capacity'] = S2S_AND_G2S_CAPACITY
            total_sat_capacity += S2S_AND_G2S_CAPACITY
            total_capacity += S2S_AND_G2S_CAPACITY

    # Update links between satellites and ground nodes based on communication range
    update_links()

    # Store the initial topology (to reset later)
    reset_to = copy.deepcopy(G)

    # Store the initial total capacities
    initial_total_capacity = total_capacity
    initial_total_capacity_ground = total_gr_capacity
    initial_total_capacity_satellite = total_sat_capacity
    initial_total_capacity_ground_satellite = total_gr_sat_capacity

    # Variables for storing results and statistics
    total_no_of_requests = 0
    total_delay_till_now = 0
    avg_prop_delay = []

    # Occupancy dictionaries for storing intermediate results
    spectrum_occupancy = {}
    ground_spectrum_occupancy = {}
    sat_spectrum_occupancy = {}
    gs_spectrum_occupancy = {}
    disruptions_saved = {}

    # Intermediate occupancy dictionaries to aggregate values over multiple runs
    spectrum_occupancy_intermed = {}
    ground_spectrum_occupancy_intermed = {}
    sat_spectrum_occupancy_intermed = {}
    gs_spectrum_occupancy_intermed = {}
    disruptions_saved_intermed = {}

    # Initialize intermediate spectrum occupancy lists for each request load
    for r in range(len(requests)):
        spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
        ground_spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
        sat_spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
        gs_spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
        disruptions_saved_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)]

    # Main loop for running simulations across multiple runs
    for j in range(no_of_runs):
        print("Run:", j + 1)

        # Reset intermediate lists for each request load before every run
        for r in range(len(requests)):
            spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
            ground_spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
            sat_spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
            gs_spectrum_occupancy_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)], []
            disruptions_saved_intermed[requests[r]] = [0 for i in range(num_of_iter + 1)]

        # Run simulation for different numbers of requests
        for i in range(len(requests)):
            times_list = [0 for _ in range(num_of_iter + 1)]
            list_of_saved_disrup, (spectrum_occu_for_one_run, times_list), (ground_spectrum_occ_for_one_run, times_list), \
                (sat_spectrum_occ_for_one_run, times_list), (gs_spectrum_occ_for_one_run, times_list) = run_simulation(G, num_of_iter, requests[i])

            # Aggregate results for disruptions saved and spectrum occupancy
            disruptions_saved_intermed[requests[i]] = [list_of_saved_disrup[x] + disruptions_saved_intermed[requests[i]][x] for x in range(len(list_of_saved_disrup))]
            spectrum_occupancy_intermed[requests[i]] = ([spectrum_occu_for_one_run[x] + spectrum_occupancy_intermed[requests[i]][0][x] for x in range(len(spectrum_occu_for_one_run))], times_list)
            ground_spectrum_occupancy_intermed[requests[i]] = ([ground_spectrum_occupancy_intermed[requests[i]][0][x] + ground_spectrum_occ_for_one_run[x] for x in range(len(ground_spectrum_occ_for_one_run))], times_list)
            sat_spectrum_occupancy_intermed[requests[i]] = ([sat_spectrum_occupancy_intermed[requests[i]][0][x] + sat_spectrum_occ_for_one_run[x] for x in range(len(sat_spectrum_occ_for_one_run))], times_list)
            gs_spectrum_occupancy_intermed[requests[i]] = ([gs_spectrum_occupancy_intermed[requests[i]][0][x] + gs_spectrum_occ_for_one_run[x] for x in range(len(gs_spectrum_occ_for_one_run))], times_list)

            # Reset the simulation after each run
            reset_simulation()

    # Print intermediate spectrum occupancy results for debugging
    print(spectrum_occupancy_intermed)

    # Average the results over multiple runs
    spectrum_occupancy = {k: ([round(x / no_of_runs, 4) for x in v[0]], v[1]) for k, v in spectrum_occupancy_intermed.items()}
    ground_spectrum_occupancy = {k: ([round(x / no_of_runs, 4) for x in v[0]], v[1]) for k, v in ground_spectrum_occupancy_intermed.items()}
    sat_spectrum_occupancy = {k: ([round(x / no_of_runs, 4) for x in v[0]], v[1]) for k, v in sat_spectrum_occupancy_intermed.items()}
    gs_spectrum_occupancy = {k: ([round(x / no_of_runs, 4) for x in v[0]], v[1]) for k, v in gs_spectrum_occupancy_intermed.items()}
    disruptions_saved = {k: [round(x / no_of_runs, 4) for x in v] for k, v in disruptions_saved_intermed.items()}

    # Print the final results for disruptions saved
    print('Disruptions saved:', disruptions_saved)

    # Plot the final results
    plot_spectrum_occupancy()
    plot_ground_spectrum_occupancy()
    plot_sat_spectrum_occupancy()
    plot_gs_spectrum_occupancy()
    plot_saved_disruptions()

    # Show the plots
    plt.show()
