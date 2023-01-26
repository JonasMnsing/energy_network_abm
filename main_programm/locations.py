import numpy as np
import matplotlib.pyplot as plt

def grid(size: float, number: int, Xsteps: int, Ysteps: int):
    """
    Args
        size (float): size of the grid in meters
        number (int): number of agents to be placed

    Returns:
        np.ndarray: 2d Array length (n x 2), where every agent gets a 2d position

    """

    pos = np.empty([0, 3])

    
    i = np.floor(np.sqrt(size))
    j = np.floor(np.sqrt(size))
    
    while(i*j != number):
        if i*j < number: 
            i += 1
        else:
            j -= 1
        if i == number or j == 0:
            ValueError("number can not be a prime")
        
    A = 0
    B = 0
    for k in np.linspace(0., int(size), num=int(i)):
        A += 1
        B = 0
        for l in np.linspace(0., int(size), num=int(j)):
            B += 1
            pos = np.append(pos, [[k, l, A%Xsteps == 0 and B%Ysteps == 0] ], axis=0)
    return pos


def polar(size: float, pos: np.ndarray, radius: float, falloff: float, producerFalloff, center: np.ndarray):
    """
    Args
        pos (np.ndarray): center of the circles
        falloff (np.ndarray): falloff of the circles
        producerFalloff (np.ndarray): falloff of the circles for producers

    Returns:
        np.ndarray: 2d Array length (n x 2), where every agent gets a 2d position

    """
    r = np.empty([0, 3])
    for i in range(len(pos)):
        
        if pos[i, 2]:
            temp_radius = radius * np.power(np.sqrt(pos[i, 0]), producerFalloff+1)
            x = center[0] + np.cos(pos[i, 1]) * temp_radius
            y = center[1] + np.sin(pos[i, 1]) * temp_radius
        else:
            temp_radius = radius * np.power(np.sqrt(pos[i, 0]), falloff+1)
            x = center[0] + np.cos(pos[i, 1]) * temp_radius
            y = center[1] + np.sin(pos[i, 1]) * temp_radius
        r = np.append(r, [[x, y, pos[i, 2]]], axis=0)
    return r


def circles(size: float, count: np.ndarray, pos: np.ndarray, radius:np.ndarray, falloff:np.ndarray, producerFalloff: np.ndarray, producerAnteil: float):
    """
    Args
        size (float): size of the grid in meters
        count (np.ndarray): number of agents to be placed in each circle
        pos (np.ndarray): center of the circles
        radius (np.ndarray): radius of the circles
        falloff (np.ndarray): falloff of the circles
        producerFalloff (np.ndarray): falloff of the circles for producers

    Returns:
        np.ndarray: 2d Array length (n x 2), where every agent gets a 2d position

    """
    
    r = np.empty([0, 3])
    for i in range(len(count)):
            for j in range(count[i]):
                x = -1
                y = -1
                p = j <= producerAnteil*count[i]
                while(not(size >= x >= 0 and size >= y >= 0)):
                    angle = 2 * np.pi * np.random.rand()
                    if p:
                        temp_radius = radius[i] * np.power(np.sqrt(np.random.rand()), producerFalloff[i]+1)
                    else:
                        temp_radius = radius[i] * np.power(np.sqrt(np.random.rand()), falloff[i]+1)
                    x = pos[i][0] + temp_radius * np.cos(angle)
                    y = pos[i][1] + temp_radius * np.sin(angle)
                   
                
                r = np.append(r, [[x, y, p]], axis=0)
    return r

def prodPercentage(producers_array: np.ndarray) -> float:
    """
    Args
        array (np.ndarray): boolean array of all agents 

    Returns:
        float: percentage of producers

    """
    return np.count_nonzero(producers_array == True) / len(producers_array)
  

def random(size, number, producerAnteil):
    """
    Args
        size (float): size of the grid in meters
        number (int): number of agents to be placed

    Returns:
        np.ndarray: 2d Array length (n x 2), where every agent gets a 2d position

    """
    
    return np.transpose([np.random.rand(int(number)) * size, np.random.rand(int(number)) * size, np.arange(int(number)) <= producerAnteil*number])

#------------------#
#   Plotting and Testing       #

def draw_locations(array: np.ndarray, size: int):
    plt.figure(figsize=(5,5))
    producers = np.empty([0, 2])
    consumers = np.empty([0, 2])

    for i in array:
   
        if i[2]==1:
            
            producers = np.append(producers,[i[:-1]], axis=0)
            
        else:
            consumers = np.append(consumers, [i[:-1]], axis=0)
    plt.plot(consumers[:,0], consumers[:,1], ".", markersize=4)
    plt.plot(producers[:,0], producers[:,1], ".", markersize=7)
    plt.ylim(-0.025*size, 1.025*size)
    plt.xlim(-0.025*size, 1.025*size)
    plt.show()

def draw_dist_destribution(array: np.ndarray, size: int):
    distances = [np.linalg.norm(array[i][0:1] - array[j][0:1]) for i in range(len(array)) for j in range(len(array)) if i < j and array[i][2] == 1]
    plt.hist(distances, bins=100, density=True)
    plt.xlim(0, size*np.sqrt(2))
    
  

