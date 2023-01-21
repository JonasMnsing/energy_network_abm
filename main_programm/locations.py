import numpy as np
import matplotlib.pyplot as plt

def grid(size: float, number: int, producerSteps: int):
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
        for l in np.linspace(0., int(size), num=int(j)):
            B += 1
            pos = np.append(pos, [[k, l, A%producerSteps == 0 and B%producerSteps == 0] ], axis=0)
    return pos


def circles(size: float, count: np.ndarray, pos: np.ndarray, radius:np.ndarray, falloff:np.ndarray, producerFalloff: np.ndarray, producerAnteil: float):
    """
    Args
        size (float): size of the grid in meters
        count (np.ndarray): number of agents to be placed in each circle
        pos (np.ndarray): center of the circles
        radius (np.ndarray): radius of the circles

    Returns:
        np.ndarray: 2d Array length (n x 2), where every agent gets a 2d position

    """
    
    r = np.empty([0, 3])
    for i in range(len(count)):
            for j in range(count[i]):
                x = -1
                y = -1
                while(not(size >= x >= 0 and size >= y >= 0)):
                    p = np.random.rand() <= producerAnteil
                    angle = 2 * np.pi * np.random.rand()
                    if p:
                        temp_radius = radius[i] * np.power(np.sqrt(np.random.rand()), producerFalloff[i]+1)
                    else:
                        temp_radius = radius[i] * np.power(np.sqrt(np.random.rand()), falloff[i]+1)
                    x = pos[i][0] + temp_radius * np.cos(angle)
                    y = pos[i][1] + temp_radius * np.sin(angle)
                    if p==False:
                        print(p)
                
                r = np.append(r, [[x, y, p]], axis=0)
    return r

def random(size: float, number: int):
    """
    Args
        size (float): size of the grid in meters
        number (int): number of agents to be placed

    Returns:
        np.ndarray: 2d Array length (n x 2), where every agent gets a 2d position

    """
    return np.random.rand(number, 2) * size

#------------------#
#   Plotting and Testing       #

def draw_locations(array: np.ndarray, size: int):
    producers = np.empty([0, 2])
    consumers = np.empty([0, 2])

    for i in array:
   
        if i[2]==1:
            
            producers = np.append(producers,[i[:-1]], axis=0)
            
        else:
            consumers = np.append(consumers, [i[:-1]], axis=0)
    plt.plot(consumers[:,0], consumers[:,1], ".")
    plt.plot(producers[:,0], producers[:,1], ".")
    plt.ylim(0, size)
    plt.xlim(0, size)
  

def draw_dist_destribution(array: np.ndarray, size: int):
    distances = [np.linalg.norm(array[i][0:1] - array[j][0:1]) for i in range(len(array)) for j in range(len(array)) if i < j and array[i][2] == 1]
    plt.hist(distances, bins=100, density=True)
    plt.xlim(0, size*np.sqrt(2))
  
plt.figure(figsize=(5,5))

#draw_locations(grid(1300, 1000, 3), 1300)
#draw_dist_destribution(circles(1300, [750, 250], [[500, 650], [1000, 650]], [2000, 1000], [1, 3], True, 25), 1300)
#draw_dist_destribution(random(1300, 1300), 1300)
#draw_dist_destribution(grid(1300, 1000), 1300)

draw_locations(circles(1300, [750, 250], [[500, 650], [1000, 650]], [2000, 1000], [1, 3], [1, 1], .35), 1300)
draw_dist_destribution(circles(1300, [750, 250], [[500, 650], [1000, 650]], [2000, 1000], [1, 3], [1, 1], .35), 1300)
#draw_locations(random(1300, 1000), 1300)
#draw_locations(grid(1300, 512), 1300)
plt.show()