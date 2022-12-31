import numpy as np
import parameters


class Agent:
    """
    An agent representing a houshold, or a power producer.

    Parameters
    ----------
    * position: np.array
        Array of shape (2) with the x and y value of the position of the agent.
    * demand_profile: np.array
        Demand profile of the agent. Array of shape(#timesteps).
    * production_profile: np.array
        Production profile of the agent. Array of shape(#timesteps).
        Default: None
    * is_producer: bool
        Whether or not the agent is a producer.
        Default: False
    * has_reservoir: bool
        Whether or not the agent has a energy reservoir.
        Default: False
    * capacity_reservoir: float
        Capacity of the energy reservoir.
        Default: None
    * current_reservoir: float
        Current energy level of the energy reservoir.
        Default: None

    Variabels of an agent
    ---------------------
    * position
    * demand_profile
    * production_profile
    * is_producer
    * has_reservoir
    * capacity_reservoir
    * current_reservoir
    * surplus

    Returns
    -------
    None.
    
    """

    def __init__(
        self,
        position: np.array,
        demand_profile: np.array,
        production_profile: np.array = None,
        is_producer: bool = False,
        has_reservoir: bool = False,
        capacity_reservoir: float = None,
        current_reservoir: float = None
        ) -> None:
        if is_producer and production_profile is None:
            raise ValueError("If the agent is a producer, a production " +
            "profile must be given!")
        if has_reservoir and (capacity_reservoir is None or \
                              current_reservoir is None):
            raise ValueError("If the agent has a energy reservoir, a " +
            "reservoir capacity and the current energy level of the reservoir" +
            "must be given!")

        self.position = position
        self.demand_profile = demand_profile
        if production_profile is not None:
            self.production_profile = production_profile
        else:
            self.production_profile = np.zeros_like(demand_profile)
        self.is_producer = is_producer
        self.has_reservoir = has_reservoir
        self.capacity_reservoir = capacity_reservoir
        self.current_reservoir = current_reservoir

        self.surplus = self.production_profile - self.demand_profile