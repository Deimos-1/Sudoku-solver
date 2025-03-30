import numpy as np
import sys
import time
np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize) ## allow to print on a wider range


# extreme:      avg. 0.91 [s]
# hard:         avg. 0.31 [s]
# easy:         avg. 0.03 [s]

#############################################################
#                        PARAMETERS                         #
#############################################################

only_one = False     ## If you want to solve only one sudoku
index = 25           ## Index of the sudoku in that case

line = True         ## If you want to use the line method
branch = True       ## If you want to use the branch method

file = "./Sudokus/easy.txt"   ## the file of interest ## current best : 43/52 

global_debug = False   ## Will show the remaining possibilities after each iteration
line_debug = False      
branch_debug = False

used_line = False
used_branch = False

timer = True


#############################################################
#                      VARIABLE INIT                        #
#############################################################

stack = []
iterations = 0
exit = False




#############################################################
#                        FUNCTIONS                          #
#############################################################


## Will take a line (sudoku) and stransform it into a list of the 9 rows, each is a string
def to_matrix(sudokus: np.ndarray) -> dict:
    """
    Returns 9x9 matrices dictionnary representing the sudoku of each line
    
    :input: 1d np.array of all sudokus strings
    """
    res = {}
    for i in range(len(sudokus)): 
        lines = [sudokus[i][9*j:9*(j+1)] for j in range(9)] ## cut each line in sections of length 9 (each line)
        m = [[lines[k][letter] for letter in range(9)] for k in range(9)] ## split each string in a 1D array and create the matrix
        res[i] = np.array(m)
    return res


## Every empty cell will be given all numbers (all possibilities), we will then remove unfeasible ones
def build_possibilities(matrix: np.ndarray) -> np.ndarray:
    possibilities = np.empty((9,9),dtype='<U9')
    for line in range(9):
        for col in range(9):
            if matrix[line][col] == '0': # we set the starting values given by the sudoku.
                possibilities[line][col] = str(123456789)
            else: 
                possibilities[line][col] = ''
    return np.array(possibilities)


## If all row/col sums equal 45, the sudoku is considered valid
def valid(matrix: np.ndarray) -> bool:
    ok = True
    matrix = matrix.astype(int)

    for row in range(9):
        if sum(list(matrix[row,:])) != 45:
            ok = False
    if ok:
        for col in range(9):
            if sum(list(matrix[:,col])) != 45:
                ok = False
    return ok


## Before eliminating possibilities, we need a function that returns the values of the neighbors (in the 3x3 cell) of a cell
def neighbors(row_index: int, col_index: int, matrix: np.ndarray) -> np.ndarray:
    # careful: only take values of length 1 (dont account for all possibilities)
    # we locate in which quadrant our cell is 
    # +---+---+---+
    # |0,0|0,1|0,2|
    # +---+---+---+
    # |1,0|1,1|1,2|
    # +---+---+---+
    # |2,0|2,1|2,2|
    # +---+---+---+

    quadrant = [row_index//3,col_index//3] ## ex: [8,4] --> [2,1]
    if quadrant == [0,0]:
        return [matrix[i,j] for i in range(0,3) for j in range(0,3) if len(matrix[i,j]) >= 1]
    elif quadrant == [0,1]:
        return [matrix[i,j] for i in range(0,3) for j in range(3,6) if len(matrix[i,j]) >= 1]
    elif quadrant == [0,2]:
        return [matrix[i,j] for i in range(0,3) for j in range(6,9) if len(matrix[i,j]) >= 1]
    elif quadrant == [1,0]:
        return [matrix[i,j] for i in range(3,6) for j in range(0,3) if len(matrix[i,j]) >= 1]
    elif quadrant == [1,1]:
        return [matrix[i,j] for i in range(3,6) for j in range(3,6) if len(matrix[i,j]) >= 1]
    elif quadrant == [1,2]:
        return [matrix[i,j] for i in range(3,6) for j in range(6,9) if len(matrix[i,j]) >= 1]
    elif quadrant == [2,0]:
        return [matrix[i,j] for i in range(6,9) for j in range(0,3) if len(matrix[i,j]) >= 1]
    elif quadrant == [2,1]:
        return [matrix[i,j] for i in range(6,9) for j in range(3,6) if len(matrix[i,j]) >= 1]
    elif quadrant == [2,2]:
        return [matrix[i,j] for i in range(6,9) for j in range(6,9) if len(matrix[i,j]) >= 1]
    else: 
        raise ValueError(f"Given index out of the matrix: {row_index,col_index} in quadrant: {quadrant}")


## We remove the possibilities by following the 3 basic rules of sudoku...
def eliminate_possibilities(sudoku: np.ndarray, possibilities: np.ndarray) -> tuple[np.array, bool] :
    change = False
    for i in range(9):
        for j in range(9):

            if sudoku[i,j] == '0': ## if a case is not solved we try to solve it
                for poss in possibilities[i,j]:
                    if poss in neighbors(i,j,sudoku):
                        new_str = str(possibilities[i,j]).replace(poss,'')
                        possibilities[i,j] = new_str
                        change = True
                    elif poss in sudoku[i,:]:
                        new_str = str(possibilities[i,j]).replace(poss,'')
                        possibilities[i,j] = new_str
                        change = True
                    elif poss in sudoku[:,j]:
                        new_str = str(possibilities[i,j]).replace(poss,'')
                        possibilities[i,j] = new_str
                        change = True

    for i in range(9):
        for j in range(9):
            if possibilities[i,j] != '':
                sub = list(neighbors(i,j,possibilities))
                sub.remove(possibilities[i,j])

                for k in possibilities[i,j]: 
                    found = False
                    for elem in sub:
                        if str(k) in elem:
                            found = True
                    if not found:
                        possibilities[i,j] = str(k)

    return possibilities, change



## if a possibiliy in a sub-grid forms a line, it can be removed from other cells in that line -> the line method
def line_elimination(sudoku: np.ndarray, possibilities: np.ndarray) -> tuple[np.array, bool] :
    change = False

    if line_debug: 
        print(f'-> Starting a line elimination of the following possibilities:\n{possibilities}\n-> And sudoku:\n{sudoku}\n')

    for qrow in range(3):
        for qcol in range(3):
            
            if line_debug:
                print(f'->   | Searching sub-matrix [{qrow,qcol}]...')

            sub_poss = possibilities[qrow*3:(qrow+1)*3, qcol*3:(qcol+1)*3]
            sub_sudoku = sudoku[qrow*3:(qrow+1)*3, qcol*3:(qcol+1)*3]

            for number in range(1,10):
                ## if the number is not already there
                if str(number) not in sub_sudoku: 
                    ## col/row mask
                    for mask in range(3):
                        ## check for sublines
                        can_eliminate = False
                        masked_sub_poss = np.delete(sub_poss, mask, axis=0)
                        if np.any(np.char.find(sub_poss[mask,:], str(number)) != -1): ## if it's in the masked line (if no string found it gives -1, by evaluating we get an array)
                            if np.all(np.char.find(masked_sub_poss, str(number)) == -1): ## and not in all other cells of the sub_poss
                                can_eliminate = True

                        if can_eliminate: 
                            if line_debug: 
                                print(f'->       | Pattern found (line): number = {number}, mask = {mask}')

                            for i in range(9): 
                                if i < qcol*3 or i >= (qcol+1)*3:
                                    string = str(possibilities[mask + qrow*3, i])
                                    if str(number) in string:
                                        new_str = string.replace(str(number),'')
                                        possibilities[mask + qrow*3, i] = new_str
                                        change = True

                                        if line_debug: 
                                            print(f'->           | 1 possibility eliminated (line): {number} @ [{mask + qrow*3, i}] : mask = {mask}, qrow = {qrow}!')

                        ## need to do the same for subcolumns
                        can_eliminate = False
                        masked_sub_poss = np.delete(sub_poss, mask, axis=1)
                        if np.any(np.char.find(sub_poss[:,mask], str(number)) != -1): ## if it's in the masked line (if no string found it gives -1, by evaluating we get an array)
                            if np.all(np.char.find(masked_sub_poss, str(number)) == -1): ## and not in all other cells of the sub_poss
                                can_eliminate = True

                        if can_eliminate: 
                            if line_debug: 
                                print(f'->       | Pattern found (col): number = {number}, mask = {mask}')

                            for i in range(9): 
                                if i < qrow*3 or i >= (qrow+1)*3: 
                                    string = str(possibilities[i, mask + qcol*3])
                                    if str(number) in string:
                                        new_str = string.replace(str(number),'')
                                        possibilities[i, mask + qcol*3] = new_str
                                        change = True

                                        if line_debug: 
                                            print(f'->           | 1 possibility eliminated (col): {number} @ [{mask + qrow*3, i}] : mask = {mask}, qrow = {qrow}!')
    return possibilities, change
                



def solve(sudoku: np.ndarray, possibilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    :sudoku: The sudoku matrix
    :possibilities: The possibilities matrix

    :return: the solved matrix
    """
    global line, branch, stack, iterations, exit, used_line, used_branch
    change = True

    if global_debug:
        print('Solving...')

    while change: 

        possibilities, change = eliminate_possibilities(sudoku, possibilities)
        if change == False and line:
            used_line = True
            if line_debug: 
                print('-> No obvious elimination, proceding to a line search...') 
            possibilities, change = line_elimination(sudoku, possibilities)
            
        if global_debug:
            print(f"-> Possibilities: \n{possibilities}\n")


        for i in range(9):
            for j in range(9):
                if len(possibilities[i,j]) == 1:
                    sudoku[i,j] = f'{possibilities[i,j]}'
                    possibilities[i,j] = ''
                    change = True



        if change == False and branch:
            used_branch = True
            if branch_debug:
                print(f'The stack: {[(item[2], item[3], [item[4]]) for item in stack]}')


            iterations += 1
            # if iterations >= 100:
            #     raise Exception('Went too deep !!!')

            if branch_debug:
                print(f'Found: \n{sudoku}\n\n{possibilities}\n')


            
            empty_slot_has_no_poss = False
            
            for i in range(9):
                for j in range(9):
                    if sudoku[i,j] == '0' and possibilities[i,j] == '':
                            empty_slot_has_no_poss = True
                            if branch_debug:
                                print(f'slot @ [{i},{j}] has possibility {possibilities[i,j]} and no number: {sudoku[i,j]}')



            if np.any(possibilities != '') and empty_slot_has_no_poss == False and not exit: ## we can still dig down...
                
                if branch_debug:
                    print(f'Branching at depth {len(stack)}')

                ## parse the possibilities to find a possible branch
                breaking = False
                for i in range(9):
                    for j in range(9):
                        if possibilities[i,j] != '':
                            ## save the last sudoku and possibilities defore change, and the necessary data
                            if branch_debug:
                                print(f'Adding {possibilities[i,j][0]} at position [{i},{j}] to the stack, saving this sudoku: \n{sudoku}')
                            stack.append((np.copy(sudoku), np.copy(possibilities), i, j, possibilities[i,j][0]))

                            if branch_debug:
                                print(f'The stack sudokus/possibilities are: {[(item[0],item[1]) for item in stack]}')
                                print(f'Choosing {possibilities[i,j][0]} at position [{i},{j}]\n{possibilities}\n')

                            sudoku[i,j] = possibilities[i,j][0]
                            possibilities[i,j] = ''

                            if branch_debug:
                                print(f'--->\n{possibilities}\n\n{sudoku}\n')

                            breaking = True

                        if breaking: 
                            break
                    if breaking:
                        break
                
                ## we try to solve with a new possibility
                sudoku, possibilities, _ = solve(sudoku, possibilities)

                if exit:
                    return sudoku, possibilities, iterations
            
            elif valid(sudoku): ## successfully solved it

                if branch_debug:
                    print('=> SUCCESS !!!')

                exit = True ## can no longer solve so we need to break out of the loop
                return sudoku, possibilities, iterations
            
            
            else: 
                
                if branch_debug:
                    print(f'[DEBUG] Not a valid board: \n{sudoku}\nempty_slot_has_no_poss: {empty_slot_has_no_poss}')
                try:
                    previous_sudoku, previous_poss, x, y, z = stack.pop(-1) ## we remove the last try 
                    if branch_debug:
                        print(f'Removed from the stack the number {z} at position [{x},{y}], \n{previous_sudoku}')

                    ## recover data from previous problem
                    sudoku = np.copy(previous_sudoku)
                    
                    if branch_debug:
                        print(f'recovering the previous sudoku: \n{sudoku}\n')

                    possibilities = previous_poss
                    l = x
                    c = y
                    val = z
                except: raise Exception(f'[ERROR]: The stack was empty: {[(item[2], item[3], [item[4]]) for item in stack]} \nsudoku: \n{sudoku} \npossibilities: \n{possibilities}\n')

                

                ## change the string of interest (remove the problematic possibility)
                string = str(possibilities[l,c])
                new_str = string.replace(val,'')
                possibilities[l,c] = new_str

                ## No need the revert change in the sudoku as it was saved before adding the value in it

                if branch_debug:
                    print(f'string: {string}, new string: {new_str} @ [{l},{c}], possibilities: \n{possibilities}\n\nsudoku:\n{sudoku}\n')

                ## solve without it
                sudoku, possibilities, _ = solve(sudoku, possibilities) ## solve no2

                if exit:
                    return sudoku, possibilities, iterations







#############################################################
#                        THE PROGRAM                        #
#############################################################


if line_debug:
    print("#===# LINE DEBUGGING MESSAGES ENABLED #===#\n")
if branch_debug:
    print("#===# BRANCH DEBUGGING MESSAGES ENABLED #===#\n")
if global_debug:
    print("#===# GLOBAL DEBUGGING MESSAGES ENABLED #===#\n")


## starting with the easy ones
with open(file=file, mode="r", encoding="utf8") as file:
    lines = file.read().splitlines() ## list of sudokus (each one is an array)
    grids = to_matrix(lines)
        
    if only_one: 
        if timer:
            start_time = time.process_time()

        sudoku = np.array(grids[index],dtype="<U9")
        print(f"-> Sudoku: \n{sudoku}\n")
        possibilities = build_possibilities(sudoku)

        sudoku, _,  iterations = solve(sudoku, possibilities)

        if timer: 
            end_time = time.process_time()

        print(f"-> Solved in {iterations} iterations, used line: {used_line}, used branch: {used_branch}: {valid(sudoku)}\n{sudoku}\n")
        if timer:
            print(f'Time: {(end_time-start_time):.2f} [s]')

    else:
        solved_count = 0
        total = len(grids.keys())
        
        ## Turn off debugging to prevent console spam
        global_debug = False
        line_debug = False
        branch_debug = False

        if timer: 
            start_time = time.process_time()
        for k in grids.keys():
            
            sudoku = np.array(grids[k],dtype="<U9")
            possibilities = build_possibilities(sudoku)
            
            try:
                sudoku, _, _ = solve(sudoku, possibilities)
            except: raise Exception(f'There was a problem solving problem {k}...')


            if valid(sudoku): 
                solved_count += 1

            iterations = 0
            exit = False
            stack.clear()
            del sudoku, possibilities
        if timer:
            end_time = time.process_time()

        print(f'solved {solved_count} sudokus out of {total}')
        if timer: 
            print(f'Time: {(end_time-start_time):.3f} [s] (avg: {((end_time-start_time)/total):.2f})')