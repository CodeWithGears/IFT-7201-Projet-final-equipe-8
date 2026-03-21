def draw_diamond(map_desc, x, y, radius, value='H'):
    size = len(map_desc)
    for i in range(size):
        for j in range(size):
            if abs(i - x) + abs(j - y) <= radius and map_desc[i][j] == 'F':
                map_desc[i][j] = value

def map_generator(size, alpha, beta, hole_size):
    """
    Generate a random map for the FrozenLake environment.

    Parameters:
    - size (int): The size of the map (size x size).
    - alpha (float): The probability of placing a hole in each cell.
    - beta (float): The probability of the hole being large.
    - hole_size (int): The number of holes to place in the map.

    Returns:
    - list of str: A list of strings representing the map, where each string is a row.
        - 'S': Start, 'F': Frozen, 'H': Hole, 'G': Goal.
    """
    import random

    # Create an empty map filled with frozen tiles
    map_desc = [['F' for _ in range(size)] for _ in range(size)]

    # Place the start and goal positions
    map_desc[0][0] = 'S'
    map_desc[size-1][size-1] = 'G'

    # Randomly place holes in the map
    holes_placed = 0
    while holes_placed < alpha * size**2:
        if random.random() < beta and hole_size > 0:
            x = random.randint(hole_size, size-1-hole_size)
            y = random.randint(hole_size, size-1-hole_size)
            hole_size = random.randint(1, hole_size)
            draw_diamond(map_desc, x, y, hole_size)
        else: 
            x = random.randint(0, size-1)
            y = random.randint(0, size-1)
            map_desc[x][y] = 'H'
        holes_placed += 1

    # Convert the 2D list to a list of strings
    return [''.join(row) for row in map_desc]

MAP = map_generator(16, 0.1, 0.5, 3)

for row in MAP:
    print(row)