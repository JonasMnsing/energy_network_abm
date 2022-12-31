import numpy as np
from agent import Agent
import parameters
import json

rng = np.random.default_rng()

"""
Read JSON:
with open("default_values.json", "r") as file:
            self.default_dict = json.load(file)
"""

class City:
    """
    A city of agents.
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

    Variables of a city
    -------------------
    * agents: list
        List of the agents in the city.
    * positions: np.array
        Positions of the agents.
    * producers_reservoir: np.array
        Information on which agents are producer/have a reservoir
    
    """

    def __init__(
        self,
        base_demand_profile: np.array,
        base_production_profile: np.array,
        capacity_reservoir: float = 0,
        current_reservoir: np.array = None,
        demand_std: float = 0,
        production_std: float = 0,
        gridsize: np.array = None,
        population: int = None,
        producer_percentage: float = None,
        reservoir_percentage: float = None,
        positions: np.array = None,
        producers_reservoir: np.array = None
    ) -> None:
        
        if gridsize is not None and \
           population is not None and \
           producer_percentage is not None and \
           reservoir_percentage is not None:
           
            nproducers = int(population*producer_percentage)
            nproducers_reservoir = int(nproducers*reservoir_percentage)
            nproducers = nproducers - nproducers_reservoir

            self.producers_reservoir = np.zeros((population, 2), dtype=bool)
            self.producers_reservoir[0:(nproducers+nproducers_reservoir), 0] = \
                True
            self.producers_reservoir[0:nproducers_reservoir, 1] = True 

            self.positions = np.empty((population, 2))
            self.positions[:, 0] = rng.random(population) * gridsize[0]
            self.positions[:, 1] = rng.random(population) * gridsize[1]
        elif positions is not None and \
             producers_reservoir is not None:
             self.positions = positions
             self.producers_reservoir = producers_reservoir
        elif positions is not None and \
             producer_percentage is not None and \
             reservoir_percentage is not None:

            self.positions = positions

            nproducers = int(population*producer_percentage)
            nproducers_reservoir = int(nproducers*reservoir_percentage)
            nproducers = nproducers - nproducers_reservoir

            self.producers_reservoir = np.zeros((self.positions.shape[0], 2), \
                                                dtype=bool)
            self.producers_reservoir[0:(nproducers+nproducers_reservoir), 0] = \
                True
            self.producers_reservoir[0:nproducers_reservoir, 1] = True 
        else:
            raise ValueError("Wrong parameters given!")

        if current_reservoir is None:
            current_reservoir = np.zeros((population))

        self.agents = []
        for i in range(self.positions.shape[0]):
            temp_demand_profile = base_demand_profile + rng.normal(
                    0, demand_std, len(base_demand_profile))
            temp_demand_profile[temp_demand_profile < 0] = 0
            temp_position = self.positions[i, :]
            temp_is_producer = self.producers_reservoir[i, 0]
            temp_has_reservoir = self.producers_reservoir[i, 1]
            if temp_is_producer:
                temp_production_profile = base_production_profile + rng.normal(
                    0, production_std, len(base_production_profile))
                temp_production_profile[temp_production_profile < 0] = 0
            else:
                temp_production_profile = None
            if temp_has_reservoir:
                temp_current_reservoir = current_reservoir[i]
            else:
                temp_current_reservoir = None

            self.agents.append(Agent(
                temp_position,
                temp_demand_profile,
                temp_production_profile,
                temp_is_producer,
                temp_has_reservoir,
                capacity_reservoir,
                temp_current_reservoir
            ))
    
    def add_agent(
        self,
        agent
    ) -> None:
        """
        Manually add an agent to the city.

        Parameters
        ----------
        * agent: Agent
            Agent to add to the city.
        """
        self.agents.append(agent)


    def save(
        self,
        fname: str
    ) -> None:
        """
        Save the positions, as well as the type of the agents in city to a
        JSON-file.

        Parameters
        ----------
        fname: str
            File to save to. Has to end with ".json"
        """
        export_dict = {"posx": self.positions[:, 0].tolist,
                       "posy": self.positions[:, 1].tolist,
                       "producers": self.producers_reservoir[:, 0].tolist,
                       "producers_reservoir": \
                        self.producers_reservoir[:, 1].tolist}
        json_object = json.dumps(export_dict, indent=4)
        with open(fname, "w") as file:
            file.write(json_object)