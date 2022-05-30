"""
You can create any other helper funtions.
Do not modify the given functions
"""
def searchq(open_set, path):
    for indexvalue in range(len(open_set)):
        if open_set[indexvalue][1] == path:
            return indexvalue

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path=[]
    visited = []
    path = [start_point]
    open_set = [[0+heuristic[start_point], path]]
    while len(open_set) > 0:
        currentcost, currentpath = open_set.pop(0)
        n = currentpath[-1]
        currentcost -= heuristic[n]
        if n in goals:
            return currentpath
        visited.append(n)
        children = [i for i in range(len(cost[0]))
                    if cost[n][i] not in [0, -1]]
        for i in children:
            new_curr_path = currentpath + [i]
            new_path_cost = currentcost + cost[n][i] + heuristic[i]
            if i not in visited and new_curr_path not in [i[1] for i in open_set]:
                open_set.append((new_path_cost, new_curr_path))
                open_set = sorted(open_set, key=lambda x: (x[0], x[1]))
            elif new_curr_path in [i[1] for i in open_set]:
                index = searchq(open_set, new_curr_path)
                open_set[index][0] = min(open_set[index][0], new_path_cost)
                open_set = sorted(open_set, key=lambda x: (x[0], x[1]))
    return path
         

def dfs(cost, start_point, path, visited, goals):
    visited[start_point] = 1
    path.append(start_point)
    if (start_point not in goals):
        value = cost[start_point]
        for n in range(len(value)):
            if ((visited[n] == 0) and (value[n] > 0)):
                nextnode = dfs(cost, n, path, visited, goals)
                if nextnode == -1:
                    return -1
                path.pop()
    else:
        return -1

def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    # TODO
    visited = [0]*(len(cost[0]))
    dfs(cost, start_point, path, visited, goals)
    return path