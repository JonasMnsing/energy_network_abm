import parameters
import numpy as np
from city import City
from multiprocessing import Pool

city = None

def link_agents(
    city: City,
    connection_length: float
) -> np.array:
    """
    Links all agents which are in the given connection_length.

    Parameters
    ----------
    city: City
        City to be linked.
    connection_length: float
        Radius in which agents are linked.
    
    Returns
    -------
    Array of shape(population, population) with ones in the lower left
    triangular matrix if the corresponding agents are linked.
    """
    population = len(city.agents)
    network_matrix = np.zeros((population, population), dtype=int)

    for i in range(1, population):
        for j in range(i):
            distance = city.agents[i].position - city.agents[j].position
            if np.hypot(distance[0], distance[1]) < connection_length:
                network_matrix[i, j] = 1

    return network_matrix

def exchange_energy(
    agent1,
    agent2,
    time: int
    ) -> float:
    """
    Helper function to exchange energy between two agents.

    Parameters
    ----------
    * agent1: Agent
    * agent2: Agent
    * time: int
        Time at which the exchange happens.

    Returns
    -------
    How much energy was exchanged (float)
    """
    exchange = 0
    if agent1.surplus[time] == 0:
        return exchange
    elif agent1.surplus[time] < 0:
        if agent2.surplus[time] <= 0:
            return exchange
        else:
            if agent2.surplus[time] >= abs(agent1.surplus[time]):
                exchange = abs(agent1.surplus[time])
                agent2.surplus[time] += agent1.surplus[time]
                agent1.surplus[time] = 0
                return exchange
            else:
                exchange = agent2.surplus[time]
                agent1.surplus[time] += agent2.surplus[time]
                agent2.surplus[time] = 0
                return exchange
    else:
        if agent2.surplus[time] >= 0:
            return exchange
        else:
            if agent1.surplus[time] >= abs(agent2.surplus[time]):
                exchange = abs(agent2.surplus[time])
                agent1.surplus[time] += agent2.surplus[time]
                agent2.surplus[time] = 0
                return exchange
            else:
                exchange = agent1.surplus[time]
                agent2.surplus[time] += agent1.surplus[time]
                agent1.surplus[time] = 0
                return exchange

def save_energy_to_reservoir(
    agent,
    time: int):
    """
    Helper function to save all remaining surplus in the reservoir.

    Parameters
    ----------
    * agent: Agent
    * time: int
        Time at which the energy is saved.

    Returns
    -------
    How much energy was saved (float)
    
    """
    if not agent.is_producer \
       or not agent.has_reservoir \
       or agent.surplus[time] <= 0 \
       or agent.current_reservoir >= agent.capacity_reservoir:
        return 0
    
    if agent.capacity_reservoir <= agent.current_reservoir+agent.surplus[time]:
        saved = agent.capacity_reservoir - agent.current_reservoir
        agent.current_reservoir = agent.capacity_reservoir
        agent.surplus[time] -= saved
        return saved
    else:
        saved = agent.surplus[time]
        agent.current_reservoir += agent.surplus[time]
        agent.surplus[time] = 0
        return saved

def use_energy_from_reservoir(
    agent,
    time: int):
    """
    Helper function to use the energy from the reservoir.

    Parameters
    ----------
    * agent: Agent
    * time: int
        Time at which the energy is used.

    Returns
    -------
    How much energy was used (float)
    
    """
    if not agent.is_producer \
       or not agent.has_reservoir \
       or agent.surplus[time] >= 0 \
       or agent.current_reservoir <= 0:
        return 0
    
    if -agent.surplus[time] >= agent.current_reservoir:
        used = agent.current_reservoir
        agent.current_reservoir = 0
        agent.surplus[time] += used
        return used
    else:
        used = -agent.surplus[time]
        agent.current_reservoir += agent.surplus[time]
        agent.surplus[time] = 0
        return used

def distr_energy1(
    city: City,
    network_matrix: np.array,
    time: int
    ):
    """
    Distributes the energy (randomly) at a given timestep (without reservoirs).

    Parameters
    ----------
    * city: City
    * network_matrix: np.array
    * time: int

    Returns
    -------
    * Total power demand during the timestep (float)
    * Total production during the timestep (float)
    * Total exchange during the timestep (float)
    * Total central supply during the timestep (float)
    * Total energy wasted during the timestep (float)
    * Links used during the timestep (array of shape(pop, pop) with the
       corresponding number of link usages in the lower left triangular matrix)
    
    """
    population = len(city.agents)

    total_con = 0
    total_pro = 0
    total_exchange = 0
    total_central_supply = 0
    total_energy_wasted = 0
    links_used = np.zeros((population, population))

    for i in range(population):
            total_con += city.agents[i].demand_profile[time]
            total_pro += city.agents[i].production_profile[time]
            for j in range(i):
                if network_matrix[i, j]:
                    exchange = exchange_energy(
                        city.agents[i],
                        city.agents[j],
                        time)
                    if exchange > 0:
                        links_used[i, j] += 1
                        total_exchange += exchange
                        
    for i in range(population):
        if city.agents[i].surplus[time] < 0:
            total_central_supply -= city.agents[i].surplus[time]
        else:
            total_energy_wasted += city.agents[i].surplus[time]

    return total_con, total_pro, total_exchange, \
           total_central_supply, total_energy_wasted, links_used

def distr_energy2(
    city: City,
    network_matrix: np.array,
    time: int
    ):
    """
    Realistic energy distribution with energy reservoirs.
    First every producer saves his surplus in his reservoir/uses his energy
    from the reservoir. If there is still energy left, it is distributed
    randomly.

    Parameters
    ----------
    * city: City
    * network_matrix: np.array
    * time: int

    Returns
    -------
    * Total power demand during the timestep (float)
    * Total production during the timestep (float)
    * Total exchange during the timestep (float)
    * Total central supply during the timestep (float)
    * Total energy wasted during the timestep (float)
    * Links used during the timestep (array of shape(pop, pop) with the
       corresponding number of link usages in the lower left triangular matrix)
    
    """
    population = len(city.agents)

    total_con = 0
    total_pro = 0
    total_exchange = 0
    total_central_supply = 0
    total_energy_wasted = 0
    links_used = np.zeros((population, population))

    for i in range(population):
        save_energy_to_reservoir(city.agents[i], time)
        use_energy_from_reservoir(city.agents[i], time)

    for i in range(population):
            total_con += city.agents[i].demand_profile[time]
            total_pro += city.agents[i].production_profile[time]
            for j in range(i):
                if network_matrix[i, j]:
                    exchange = exchange_energy(
                        city.agents[i],
                        city.agents[j],
                        time)
                    if exchange > 0:
                        links_used[i, j] += 1
                        total_exchange += exchange
                        
    for i in range(population):
        if city.agents[i].surplus[time] < 0:
            total_central_supply -= city.agents[i].surplus[time]
        else:
            total_energy_wasted += city.agents[i].surplus[time]

    return total_con, total_pro, total_exchange, \
           total_central_supply, total_energy_wasted, links_used

def distr_energy3(
    city: City,
    network_matrix: np.array,
    time: int
    ):
    """
    Communistic energy distribution with energy reservoirs.
    First every producer uses his saved energy from the reservoir (if surplus
    is negative). Then remaining surplus is distributed randomly.
    If there is still energy left, it is saved in the reservoirs.
    
    Parameters
    ----------
    * city: City
    * network_matrix: np.array
    * time: int

    Returns
    -------
    * Total power demand during the timestep (float)
    * Total production during the timestep (float)
    * Total exchange during the timestep (float)
    * Total central supply during the timestep (float)
    * Total energy wasted during the timestep (float)
    * Links used during the timestep (array of shape(pop, pop) with the
       corresponding number of link usages in the lower left triangular matrix)
    
    """
    population = len(city.agents)

    total_con = 0
    total_pro = 0
    total_exchange = 0
    total_central_supply = 0
    total_energy_wasted = 0
    links_used = np.zeros((population, population))

    for i in range(population):
        use_energy_from_reservoir(city.agents[i], time)

    for i in range(population):
            total_con += city.agents[i].demand_profile[time]
            total_pro += city.agents[i].production_profile[time]
            for j in range(i):
                if network_matrix[i, j]:
                    exchange = exchange_energy(
                        city.agents[i],
                        city.agents[j],
                        time)
                    if exchange > 0:
                        links_used[i, j] += 1
                        total_exchange += exchange
                        
    for i in range(population):
        save_energy_to_reservoir(city.agents[i], time)
                        
    for i in range(population):
        if city.agents[i].surplus[time] < 0:
            total_central_supply -= city.agents[i].surplus[time]
        else:
            total_energy_wasted += city.agents[i].surplus[time]

    return total_con, total_pro, total_exchange, \
           total_central_supply, total_energy_wasted, links_used

def simulate_single(
    city: City,
    network_matrix: np.array,
    distr_energy,
    index_interval: list = [8, 17],
    links_per_threshold: float = 0.05
    ):
    """
    Simulates a single run of a given city with a given network_matrix and a
    given distribute energy function.

    Parameters
    ----------
    * city: City
    * network_matrix: np.array
    * distr_energy: function
        Distribute energy function
    * index_interval: list
        Interval in which the indices are evaluated
    * links_er_threshold: float
        Threshold at which links are considered to be active.

    Returns
    -------
    * links_percentage (float)
    * energy_loss_percentage (float)
    * supply_percentage (float)
    
    """
    time_steps = len(city.agents[0].demand_profile)
    population = len(city.agents)
    total_links = np.sum(network_matrix != 0)

    t0, t1 = index_interval

    total_con = np.zeros(time_steps)
    total_pro = np.zeros(time_steps)
    total_exchange = np.zeros(time_steps)
    total_central_supply = np.zeros(time_steps)
    total_energy_wasted = np.zeros(time_steps)
    total_links_used = np.zeros((population, population))

    for time in range(time_steps):
        temp = distr_energy(city, network_matrix, time)
        total_con[time] = temp[0]
        total_pro[time] = temp[1]
        total_exchange[time] = temp[2]
        total_central_supply[time] = temp[3]
        total_energy_wasted[time] = temp[4]
        total_links_used += temp[5]


    links_percentage = np.sum(total_links_used > (t1-t0)*links_per_threshold)\
                              /total_links
    energy_loss_percentage = np.sum(total_energy_wasted[t0:t1])\
                                    /np.sum(total_pro[t0:t1])
    supply_percentage = np.sum(total_central_supply[t0:t1])\
                               /np.sum(total_con[t0:t1])

    return links_percentage, energy_loss_percentage, supply_percentage

def create_and_simulate(
    link_agents,
    distr_energy,
    base_demand_profile: np.array,
    base_production_profile: np.array,
    connection_length,
    capacity_reservoir: float = 0,
    current_reservoir: np.array = None,
    demand_std: float = 0,
    production_std: float = 0,
    gridsize: np.array = None,
    population: int = None,
    producer_percentage: float = None,
    reservoir_percentage: float = None,
    positions: np.array = None,
    producers_reservoir: np.array = None,
    index_interval: list = [8, 17],
    links_per_threshold: float = 0.05
    ):
    """
    Creates. links and simulates a city.
    There are three ways creating a city:

    1. gridsize, population, producer_percentage and reservoir_percentage
       are given. Agents will be randomly placed on the grid with the given
       producer and reservoir percentage.

    2. positions and producers_reservoir are given.
       City will be created with the agents at the given locations and the given
       attributes

    3. positions, producer_percentage and reservoir_percentage are given.
       Agents will be located at the given positions. The first 
       producer_percentage agents will be producers. (The same goes for the
       agents with reservoirs)

    Parameters
    ----------
    * link_agents: function
        Linking function for the agents.
    * distr_energy: function
        Distribute energy fuction.
    * base_demand_profile: np.array
        Base demand profile of each agent. Array of shape(#timesteps).
    * base_production_profile: np.array
        Base production profile of each producer agent. Array of 
        shape(#timesteps).
    * capacity_reservoir: float
        Capacity of the energy reservoir of each reservoir agent.
        Default: 0
    * current_reservoir: np.array
        Current energy level of the reservoir of each reservoir agent.
        Array of shape(#agents)
        Default: None
    * demand_std: float
        Standard deviation of the normal distribution added to the demand
        profile of each agent.
        Default: 0
    * production_std: float
        Standard deviation of the normal distribution added to the production
        profile of each producer agent.
        Default: 0
    * gridsize: np.array
        Size of the grid the agents are placed on.
        Array of shape(2).
        Default: None
    * population: int
        Number of agents to be placed on the grid.
        Default: None
    * producer_percentage: float
        Percentage of producers.
        Default: None
    * reservoir_percentage: float
        Percentage of producer with a reservoir.
        Default: None
    * positions: np.array
        Positions of the agents.
        Array of shape(#agents, 2)
        Default: None
    * producers_reservoir: np.array
        Which agents are producers and which additionally have a reservoir.
        producers_reservoir[:, 0] gives the producers
        producers_reservoir[:, 1] gives the producers with reservoir
        Array of shape(#agents, 2, dtype=bool)
        Default: None
    * index_interval: list
        Interval in which the indices are evaluated
    * links_er_threshold: float
        Threshold at which links are considered to be active.
    
    """
    city = City(
        base_demand_profile,
        base_production_profile,
        capacity_reservoir,
        current_reservoir,
        demand_std,
        production_std,
        gridsize,
        population,
        producer_percentage,
        reservoir_percentage,
        positions,
        producers_reservoir)

    network_matrix = link_agents(city, connection_length)

    return simulate_single(city, network_matrix, distr_energy, index_interval,
                           links_per_threshold)

def simulate(
    link_agents,
    distr_energy,
    base_demand_profile: np.array,
    base_production_profile: np.array,
    connection_lengths: np.array,
    capacity_reservoir: np.array,
    producer_percentage: np.array,
    reservoir_percentage: np.array,
    demand_std: float = 0,
    production_std: float = 0,
    gridsize: np.array = None,
    population: int = None,
    positions: np.array = None,
    producers_reservoir: np.array = None,
    current_reservoir: np.array = None,
    index_interval: list = [8, 17],
    links_per_threshold: float = 0.05
    ):
    """
    Simulates cities with different connection lengths, reservoir capacities,
    producer percentages and reservoir percentages using all threads of the
    CPU.

    Executes create_and_simulate multiple times with varying connection length,
    reservoir capacity, producer percentage and reservoir percentage.

    Important parameters
    --------------------
    * connection_lengths: np.array
        1D-Array of the connection lengths
    * capacity_reservoir: np.array
        1D-Array of the reservoir capacities
    * producer_percentage: np.array
        1D-Array of the producer percentages
    * reservoir_percentage: np.array
        1D-Array of the reservoir percentages
    * For the other parameters see create_and_simulate

    Returns
    -------
    * links_percentage: np.array
        Array of shape(len(connection_lengths),
                       len(capacity_reservoir),
                       len(producer_percentage),
                       len(reservoir_percentage))
        with the links percentage of a specific setting at the corresponding 
        position in the array.
    * energy_loss_percentage: np.array
        Array of shape(len(connection_lengths),
                       len(capacity_reservoir),
                       len(producer_percentage),
                       len(reservoir_percentage))
        with the energy loss percentage of a specific setting at the
        corresponding position in the array.
    * supply_percentage: np.array
        Array of shape(len(connection_lengths),
                       len(capacity_reservoir),
                       len(producer_percentage),
                       len(reservoir_percentage))
        with the supply percentage of a specific setting at the corresponding
        position in the array.
    
    """

    links_percentage = np.zeros((len(connection_lengths),
                                 len(capacity_reservoir),
                                 len(producer_percentage),
                                 len(reservoir_percentage)))
    energy_loss_percentage  = np.zeros((len(connection_lengths),
                                 len(capacity_reservoir),
                                 len(producer_percentage),
                                 len(reservoir_percentage)))
    supply_percentage  = np.zeros((len(connection_lengths),
                                 len(capacity_reservoir),
                                 len(producer_percentage),
                                 len(reservoir_percentage)))

    par_list = []

    for length in connection_lengths:
        for cap in capacity_reservoir:
            for prod in producer_percentage:
                for res in reservoir_percentage:
                    par_list.append([
                        link_agents,
                        distr_energy,
                        base_demand_profile,
                        base_production_profile,
                        length,
                        cap,
                        current_reservoir,
                        demand_std,
                        production_std,
                        gridsize,
                        population,
                        prod,
                        res,
                        positions,
                        producers_reservoir,
                        index_interval,
                        links_per_threshold
                    ])

    with Pool() as pool:
        results = pool.starmap(create_and_simulate, par_list)

    n = 0
    for i in range(len(connection_lengths)):
        for j in range(len(capacity_reservoir)):
            for k in range(len(producer_percentage)):
                for l in range(len(reservoir_percentage)):
                    links_percentage[i, j, k, l] = results[n][0]
                    energy_loss_percentage[i, j, k, l] = results[n][1]
                    supply_percentage[i, j, k, l] = results[n][2]

                    n += 1

    return links_percentage, energy_loss_percentage, supply_percentage

def batch_simulate(
    link_agents,
    distr_energy,
    base_demand_profile: np.array,
    base_production_profile: np.array,
    connection_lengths: np.array,
    capacity_reservoir: np.array,
    producer_percentage: np.array,
    reservoir_percentage: np.array,
    simulations: int = 10,
    demand_std: float = 0,
    production_std: float = 0,
    gridsize: np.array = None,
    population: int = None,
    positions: np.array = None,
    producers_reservoir: np.array = None,
    current_reservoir: np.array = None,
    index_interval: list = [8, 17],
    links_per_threshold: float = 0.05
    ):
    """
    Simulates cities with different connection lengths, reservoir capacities,
    producer percentages and reservoir percentages using all threads of the
    CPU. Each parameter combination is simulated 'simulations' amount of times.

    Executes create_and_simulate multiple times with varying connection length,
    reservoir capacity, producer percentage and reservoir percentage.

    Important parameters
    --------------------
    * connection_lengths: np.array
        1D-Array of the connection lengths
    * capacity_reservoir: np.array
        1D-Array of the reservoir capacities
    * producer_percentage: np.array
        1D-Array of the producer percentages
    * reservoir_percentage: np.array
        1D-Array of the reservoir percentages
    * simulations: int (default = 10)
        Number of simulations per parameter combination
    * For the other parameters see create_and_simulate

    Returns
    -------
    * links_percentage_mean: np.array
        Array of shape(len(connection_lengths),
                       len(capacity_reservoir),
                       len(producer_percentage),
                       len(reservoir_percentage))
        with the mean links percentage of a specific setting at the
        corresponding position in the array.
    * energy_loss_percentage_mean: np.array
        Array of shape(len(connection_lengths),
                       len(capacity_reservoir),
                       len(producer_percentage),
                       len(reservoir_percentage))
        with the mean energy loss percentage of a specific setting at the
        corresponding position in the array.
    * supply_percentage_mean: np.array
        Array of shape(len(connection_lengths),
                       len(capacity_reservoir),
                       len(producer_percentage),
                       len(reservoir_percentage))
        with the mean supply percentage of a specific setting at the
        corresponding position in the array.
    * 'arrays above'_std: np.array
        Array of shape(len(connection_lengths),
                       len(capacity_reservoir),
                       len(producer_percentage),
                       len(reservoir_percentage))
        with the standard deviations of the values above.
    
    """
    links_percentage = np.zeros((len(connection_lengths),
                                 len(capacity_reservoir),
                                 len(producer_percentage),
                                 len(reservoir_percentage),
                                 simulations))
    energy_loss_percentage  = np.zeros((len(connection_lengths),
                                        len(capacity_reservoir),
                                        len(producer_percentage),
                                        len(reservoir_percentage),
                                        simulations))
    supply_percentage  = np.zeros((len(connection_lengths),
                                   len(capacity_reservoir),
                                   len(producer_percentage),
                                   len(reservoir_percentage),
                                   simulations))

    par_list = []

    for length in connection_lengths:
        for cap in capacity_reservoir:
            for prod in producer_percentage:
                for res in reservoir_percentage:
                    for i in range(simulations):
                        par_list.append([
                            link_agents,
                            distr_energy,
                            base_demand_profile,
                            base_production_profile,
                            length,
                            cap,
                            current_reservoir,
                            demand_std,
                            production_std,
                            gridsize,
                            population,
                            prod,
                            res,
                            positions,
                            producers_reservoir,
                            index_interval,
                            links_per_threshold
                        ])

    with Pool() as pool:
        results = pool.starmap(create_and_simulate, par_list)

    n = 0
    for i in range(len(connection_lengths)):
        for j in range(len(capacity_reservoir)):
            for k in range(len(producer_percentage)):
                for l in range(len(reservoir_percentage)):
                    for m in range(simulations):
                        links_percentage[i, j, k, l, m] = results[n][0]
                        energy_loss_percentage[i, j, k, l, m] = results[n][1]
                        supply_percentage[i, j, k, l, m] = results[n][2]
    
                        n += 1
                        
    links_percentage_mean = np.mean(links_percentage, axis=4)
    energy_loss_percentage_mean = np.mean(energy_loss_percentage, axis=4)
    supply_percentage_mean = np.mean(supply_percentage, axis=4)
    
    links_percentage_std = np.std(links_percentage, axis=4)
    energy_loss_percentage_std = np.std(energy_loss_percentage, axis=4)
    supply_percentage_std = np.std(supply_percentage, axis=4)

    return links_percentage_mean, energy_loss_percentage_mean, \
           supply_percentage_mean, links_percentage_std, \
           energy_loss_percentage_std, supply_percentage_std
