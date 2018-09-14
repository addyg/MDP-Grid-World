__author__ = "Piyush Umate"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
move_index = [
    "Walk Up",
    "Walk Down",
    "Walk Left",
    "Walk Right",
    "Run Up",
    "Run Down",
    "Run Left",
    "Run Right",
    "Exit"
]

from collections import deque
row_size = column_size = 0
utility_grid = []
detect_state = []
action_state_grid= []
policy= []
grid_rewards =[]
walk_probability = run_probability = 0.0
walk_reward = run_reward = gamma = 0.0
key_coordinates = {}
state_priorities = []
visited_states = {}
priority_queue = deque()

#epsilon = 0.0
encode_factor = 10000
import time

def encode(x, y):
    return x*encode_factor+y

def value_iteration():
    global row_size, column_size, utility_grid, action_state_grid, column_size, gamma, walk_reward, run_reward, \
        policy, grid_rewards, key_coordinates, state_priorities
    while True:
        delta = 0.0
        #x=time.time()
        for state_priority in state_priorities:
            row = key_coordinates[state_priority][0]
            column = key_coordinates[state_priority][1]
            if grid_rewards[row][column] is not None:
                max_value = -float("inf")
                max_index = 8
                for index, action in enumerate(action_state_grid[row][column]):
                    sum = 0.0

                    for next_action, probability in action.iteritems():
                        next_action_row = key_coordinates[next_action][0]
                        next_action_column = key_coordinates[next_action][1]
                        sum += probability * utility_grid[next_action_row][next_action_column]
                    sum = (sum * gamma) + (walk_reward if index < 4 else run_reward)
                    if sum > max_value:
                        max_value = sum
                        max_index = index
                if max_index == 8:
                    max_value = 0
                new_val = grid_rewards[row][column] + max_value
                delta = (max(delta, (abs(new_val-utility_grid[row][column]))))
                utility_grid[row][column] = new_val
                policy[row][column] = move_index[max_index]
        #print time.time()-x
        #print delta, beta
        if delta == 0.0:
            return policy


def update_grid_state(grid, row, column, value=None):
    grid[row-1][column-1] = value

def process_output(list, row_size,output=OUTPUT_FILE):
    write_list = []
    for row in range(row_size):
        write_list.append(','.join(map(str,list[row_size-row-1])) + '\n')
    file = open(OUTPUT_FILE, "w")
    file.writelines(write_list)
    file.close()

def update_dict_val(dictionary, key, value=0.0):
    dictionary[key] = dictionary.get(key, 0.0) + value

def update_action_state_grid():
    global row_size, column_size, detect_state, utility_grid, walk_probability, run_probability
    for row in range(row_size):
        for column in range(column_size):
            #that state is not wall or terminal
            if detect_state[row][column] is not None:
                #walk action up , check boundary conditions and not wall
                walk_up_dict = {}
                if row+1 < row_size and utility_grid[row+1][column] is not None:
                    update_dict_val(
                        walk_up_dict,
                        encode(row+1,column),
                        walk_probability
                    )
                else:
                    update_dict_val(
                        walk_up_dict,
                        encode(row,column),
                        walk_probability
                    )

                if column-1 >= 0 and utility_grid[row][column-1] is not None:
                    update_dict_val(
                        walk_up_dict,
                        encode(row, column-1),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_up_dict,
                        encode(row, column),
                        0.5*(1-walk_probability)
                    )


                if column+1 < column_size and utility_grid[row][column+1] is not None:
                    update_dict_val(
                        walk_up_dict,
                        encode(row, column+1),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_up_dict,
                        encode(row, column),
                        0.5*(1-walk_probability)
                    )
                action_state_grid[row][column].append(walk_up_dict)

                #walk action down
                walk_down_dict = {}

                if row-1 >= 0 and utility_grid[row-1][column] is not None:
                    update_dict_val(
                        walk_down_dict,
                        encode(row-1,column),
                        walk_probability
                    )
                else:
                    update_dict_val(
                        walk_down_dict,
                        encode(row, column),
                        walk_probability
                    )

                if column-1 >= 0 and utility_grid[row][column-1] is not None:
                    update_dict_val(
                        walk_down_dict,
                        encode(row, column-1),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_down_dict,
                        encode(row, column),
                        0.5*(1-walk_probability)
                    )


                if column+1 < column_size and utility_grid[row][column+1] is not None:
                    update_dict_val(
                        walk_down_dict,
                        encode(row, column+1),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_down_dict,
                        encode(row, column),
                        0.5*(1-walk_probability)
                    )
                action_state_grid[row][column].append(walk_down_dict)

                #walk action left
                walk_left_dict = {}
                if column-1 >= 0 and utility_grid[row][column-1] is not None:
                    update_dict_val(
                        walk_left_dict,
                        encode(row, column-1),
                        walk_probability
                    )
                else:
                    update_dict_val(
                        walk_left_dict,
                        encode(row, column),
                        walk_probability
                    )

                if row-1 >= 0 and utility_grid[row-1][column] is not None:
                    update_dict_val(
                        walk_left_dict,
                        encode(row-1,column),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_left_dict,
                        encode(row,column),
                        0.5*(1-walk_probability)
                    )


                if row+1 < row_size and utility_grid[row+1][column] is not None:
                    update_dict_val(
                        walk_left_dict,
                        encode(row+1, column),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_left_dict,
                        encode(row, column),
                        0.5*(1-walk_probability)
                    )
                action_state_grid[row][column].append(walk_left_dict)

                #walk action right
                walk_right_dict = {}
                if column+1 < column_size and utility_grid[row][column+1] is not None:
                    update_dict_val(
                        walk_right_dict,
                        encode(row, column+1),
                        walk_probability
                    )
                else:
                    update_dict_val(
                        walk_right_dict,
                        encode(row, column),
                        walk_probability
                    )

                if row-1 >= 0 and utility_grid[row-1][column] is not None:
                    update_dict_val(
                        walk_right_dict,
                        encode(row-1, column),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_right_dict,
                        encode(row, column),
                        0.5*(1-walk_probability)
                    )


                if row+1 < row_size and utility_grid[row+1][column] is not None:
                    update_dict_val(
                        walk_right_dict,
                        encode(row+1, column),
                        0.5*(1-walk_probability)
                    )
                else:
                    update_dict_val(
                        walk_right_dict,
                        encode(row, column),
                        0.5*(1-walk_probability)
                    )
                action_state_grid[row][column].append(walk_right_dict)

                #run action up
                run_up_dict = {}
                if row+2 < row_size and \
                        utility_grid[row+2][column] is not None and \
                        utility_grid[row+1][column] is not None:
                    update_dict_val(
                        run_up_dict,
                        encode(row+2, column),
                        run_probability
                    )
                else:
                    update_dict_val(
                        run_up_dict,
                        encode(row, column),
                        run_probability
                    )

                if column-2 >= 0 and \
                        utility_grid[row][column-2] is not None and \
                        utility_grid[row][column-1] is not None:
                    update_dict_val(
                        run_up_dict,
                        encode(row, column-2),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_up_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )


                if column+2 < column_size and \
                        utility_grid[row][column+2] is not None and \
                        utility_grid[row][column+1] is not None:
                    update_dict_val(
                        run_up_dict,
                        encode(row, column+2),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_up_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )
                action_state_grid[row][column].append(run_up_dict)

                #run action down
                run_down_dict = {}
                if row-2 >= 0 and \
                        utility_grid[row-2][column] is not None and \
                        utility_grid[row-1][column] is not None:
                    update_dict_val(
                        run_down_dict,
                        encode(row-2, column),
                        run_probability
                    )
                else:
                    update_dict_val(
                        run_down_dict,
                        encode(row, column),
                        run_probability
                    )

                if column-2 >= 0 and \
                        utility_grid[row][column-2] is not None and \
                        utility_grid[row][column-1] is not None:
                    update_dict_val(
                        run_down_dict,
                        encode(row, column-2),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_down_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )


                if column+2 < column_size and \
                        utility_grid[row][column+2] is not None and \
                        utility_grid[row][column+1] is not None:
                    update_dict_val(
                        run_down_dict,
                        encode(row, column+2),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_down_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )
                action_state_grid[row][column].append(run_down_dict)

                #run action left
                run_left_dict = {}
                if column-2 >= 0 and \
                        utility_grid[row][column-2] is not None and \
                        utility_grid[row][column-1] is not None:
                    update_dict_val(
                        run_left_dict,
                        encode(row, column-2),
                        run_probability
                    )
                else:
                    update_dict_val(
                        run_left_dict,
                        encode(row, column),
                        run_probability
                    )

                if row-2 >= 0 and \
                        utility_grid[row-2][column] is not None and \
                        utility_grid[row-1][column] is not None:
                    update_dict_val(
                        run_left_dict,
                        encode(row-2, column),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_left_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )


                if row+2 < row_size and \
                        utility_grid[row+2][column] is not None and \
                        utility_grid[row+1][column] is not None:
                    update_dict_val(
                        run_left_dict,
                        encode(row+2, column),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_left_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )
                action_state_grid[row][column].append(run_left_dict)

                #run action right
                run_right_dict = {}
                if column+2 < column_size and \
                        utility_grid[row][column+2] is not None and \
                        utility_grid[row][column+1] is not None:
                    update_dict_val(
                        run_right_dict,
                        encode(row, column+2),
                        run_probability
                    )
                else:
                    update_dict_val(
                        run_right_dict,
                        encode(row, column),
                        run_probability
                    )

                if row-2 >= 0 and \
                        utility_grid[row-2][column] is not None and \
                        utility_grid[row-1][column] is not None:
                    update_dict_val(
                        run_right_dict,
                        encode(row-2,column),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_right_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )


                if row+2 < row_size and \
                        utility_grid[row+2][column] is not None and \
                        utility_grid[row+1][column] is not None:
                    update_dict_val(
                        run_right_dict,
                        encode(row+2, column),
                        0.5*(1-run_probability)
                    )
                else:
                    update_dict_val(
                        run_right_dict,
                        encode(row, column),
                        0.5*(1-run_probability)
                    )
                action_state_grid[row][column].append(run_right_dict)




'''
    flag dict - track all states visited {'row_column': False/True}
    put terminals initially in queue
    while queue is not empty:
        dequeue an element
        check flag to see if already visited
        enqueue in order of walk up, walk down, walk left right only if not visited
        set them as visited
    loop through flag dict and if there are any false, add them to queue and run the function
    create a list of stringified of coordinates
'''
def process_queue():
    global  state_priorities, priority_queue, visited_states, row_size, column_size
    while priority_queue:
        #handle if already there
        stringy_state = priority_queue.popleft()
        if not visited_states[stringy_state]:
            visited_states[stringy_state] = True
            state_priorities.append(stringy_state)
            row, column = key_coordinates[stringy_state]
            #enqueue walk up
            new_state = encode(row+1, column)
            if row+1 < row_size:
                priority_queue.append(new_state)

            #enqueue walk down
            new_state = encode(row-1, column)
            if row-1 >= 0:
                priority_queue.append(new_state)

        #enqueue walk left
            new_state = encode(row, column-1)
            if column-1 >= 0:
                priority_queue.append(new_state)

        #enqueue walk right
            new_state = encode(row, column+1)
            if column+1 < column_size:
                priority_queue.append(new_state)

        #enquee run up
            new_state = encode(row+2, column)
            if row + 2 < row_size:
                priority_queue.append(new_state)

        #enque run down
            new_state = encode(row-2, column)
            if row - 2 >= 0:
                priority_queue.append(new_state)

         #enque run left
            new_state = encode(row, column-2)
            if column - 2 >= 0:
                 priority_queue.append(new_state)

        #enque run right
            new_state = encode(row, column+2)
            if column + 2 < column_size:
                priority_queue.append(new_state)


def create_state_priorities(terminals):
    global priority_queue, visited_states
    for key,value in sorted(terminals.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        priority_queue.append(key)

    #call queue
    process_queue()
    for state, visited in enumerate(visited_states):
        if not visited:
            priority_queue.append(state)
    #call queue
    #process_queue()

def process_input(input=INPUT_FILE):
    #print(time.ctime())
    terminals = {}
    global row_size, column_size, utility_grid, detect_state, \
        walk_reward, run_reward, walk_probability, run_probability, gamma, action_state_grid, policy, \
        grid_rewards, key_coordinates, visited_states, state_priorities

    with open(input, 'r') as file_pointer:
        lines = file_pointer.read().splitlines()
        grid_size = map(int,lines[0].split(','))
        row_size, column_size = grid_size[0], grid_size[1]
        detect_state = [[0]* column_size for r in range(0, row_size)]
        utility_grid = [[0.0] * column_size for r in range(0, row_size)]

        wall_cells_count = int(lines[1])
        for waller in lines[2: 2+wall_cells_count]:
            wall = waller.split(',')
            x, y = int(wall[0]), int(wall[1])
            update_grid_state(detect_state,x,y)
            update_grid_state(utility_grid,x,y)

        terminal_cells_count = int(lines[2+wall_cells_count])
        for j in lines[3+wall_cells_count: 3+wall_cells_count+terminal_cells_count]:
            terminal = j.split(',')
            x,y, reward= int(terminal[0]),int(terminal[1]),float(terminal[2])
            terminals[encode(x-1,y-1)] = reward
            update_grid_state(utility_grid,x,y,reward)
            update_grid_state(detect_state,x,y)

        count = 3+wall_cells_count+terminal_cells_count
        probabilities = map(float,lines[count].split(','))
        walk_probability = probabilities[0]
        run_probability = probabilities[1]
        count +=1

        rewards = map(float, lines[count].split(','))
        walk_reward = rewards[0]
        run_reward = rewards[1]
        count += 1
        gamma = float(lines[count])

    for r in range(row_size):
        row = []
        for c in range(column_size):
            stringy_coordinates = encode(r, c)
            key_coordinates[stringy_coordinates] = r,c
            visited_states[stringy_coordinates] = False
            row.append([])
        action_state_grid.append(row)

    update_action_state_grid()
    create_state_priorities(terminals)
    #print len(state_priorities)
    #print state_priorities
    for row in range(row_size):
        p_row = []
        r_row = []
        for column in range(column_size):
            p_row.append(utility_grid[row][column])
            r_row.append(utility_grid[row][column])
            utility_grid[row][column] = 0.0
        policy.append(p_row)
        grid_rewards.append(r_row)
    #print(time.ctime())
    policy = value_iteration()

    process_output(policy, row_size)

#x=time.time()
process_input()
#print time.time()-x
