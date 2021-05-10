import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2 #Hmm van de, tai sao lai la 1,0 trong khi o duoi tinh theo row truoc col sau
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']] #Len, XUong, Trai, Phai
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter #push thi la in hoa
        else:
            action.pop(3) # drop the upper letter #Khong push thi la chu in thuong
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) #Tra ve vi tri cua nhung cai hop tu gameState
    beginPlayer = PosOfPlayer(gameState) #Tra ve vi tri ban dau cua Agent

    startState = (beginPlayer, beginBox) #startState la mot tuple chua nhung vi tri ban dau quan trong
    frontier = collections.deque([[startState]]) #Frontier la mot deque va chua startState luc khoi dau
    exploredSet = set() #set is for making sure there is no duplicate item to downsize storage
    actions = [[0]] 
    temp = [] #temp is a collections of move that make StartState into GoalState
    while frontier: #Khi ma van con node trong frontier
        node = frontier.pop() #Thi lay node do ra tu frontier
        node_action = actions.pop() #Lấy action cuối cùng trong list actions ra
        if isEndState(node[-1][-1]): #Neu node cuoi cung la End State thi 
            temp += node_action[1:] #temp sẽ lấy hết phần tử từ vị trí thứ nhất tới hết
            break #Out ra khỏi vòng lập vì đã tìm ra được đường tới End State
        if node[-1] not in exploredSet: #Nếu node cuối cùng chưa có trong exploredSet
            exploredSet.add(node[-1]) #Thì thêm nó vào exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): #Với tất cả những legal action có được từ vị trí hiện tại của hộp và người chơi
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Thực hiện việc gán vị trí mới
                if isFailed(newPosBox): #Nếu như vị trí mới của hộp có khả năng fail thì 
                    continue #Tiếp tục vòng lặp với vị trí có thể đi khác của player
                frontier.append(node + [(newPosPlayer, newPosBox)]) #Thêm vào hàng đợi vị trí mới của player và box
                actions.append(node_action + [action[-1]]) #Thêm vào actions hành động mới nhất
    return temp 

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) #Vị trí ban đầu của hộp    
    beginPlayer = PosOfPlayer(gameState) #Vị trí ban đầu của người chơi

    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]]) # store states
    actions = collections.deque([[0]]) # store actions
    exploredSet = set() #Set of explored state to avoid going back to the old state
    temp = []
    ### Implement breadthFirstSearch here
    while frontier: #Khi ma van con node trong frontier
        node = frontier.popleft() #Thi lay node do ra tu frontier
        node_action = actions.popleft() #node_action chứa những action tương ứng với state trong node
        if isEndState(node[-1][-1]): #Neu node cuoi cung la End State thi 
            temp += node_action[1:] #Lay tat ca duong di tu âu StartState tới GoalState
            break
        if node[-1] not in exploredSet: #Nếu như bước đi này vẫn chưa có trong những bước đã đi
            exploredSet.add(node[-1]) #Thì thêm vào những bước đã đi
            for action in legalActions(node[-1][0], node[-1][1]): #Với mỗi action có thể đi với vị trí của người chơi và hộp
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Update vị trí người chơi và vị trí hộp mới
                if isFailed(newPosBox):#Nếu như vị trí mới của hộp có khả năng thất bại thì thử vị trí khác
                    continue #Thì bỏ qua
                frontier.append(node + [(newPosPlayer, newPosBox)]) #Thêm bước đi mới nhất vào đường đi
                actions.append(node_action + [action[-1]]) #Thêm cách thay đổi vị trí mới nhất của người chơi vào actions
    return temp #Trả về đường đi đến đích khi tìm thấy đường đi legal ngắn nhất đầu tiên
    
def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()]) #Trả về những bước đi chữ thường trên đường đi

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) #Vị trí của hộp lúc xuất phát
    beginPlayer = PosOfPlayer(gameState) #Vị trí của người chơi lúc xuất phát

    startState = (beginPlayer, beginBox) #StartState là một tuple chứa vị trí ban đầu của người chơi và hộp
    frontier = PriorityQueue() #frontier ở dạng PriorityQueue theo cấu trúc heap với các node được thêm vào theo priority
    frontier.push([startState], 0) #Sử dụng hàm push của PriorityQueue để thêm vào bước đi đầu tiên với priority = 0 như là gốc của heap
    exploredSet = set() #Khai báo exploredSet có kiểu dữ liệu set để tránh lập lại
    actions = PriorityQueue() #Khai báo actions có cấu trúc giống với frontier
    actions.push([0], 0) #Sử dụng hàm push để push vào action tương ứng với vị trí ban đầu là đứng yên như là gốc của heap
    temp = [] #Đây sẽ là nơi chứa đường đi tới đích cuối cùng
    ### Implement uniform cost search here
    while not frontier.isEmpty(): #Sử dụng hàm check isEmpty của cấu trúc PriorityQueue
        node = frontier.pop() #Sử dụng hàm pop của cấu trúc để lấy node với giá trị Priority nhỏ nhất ra khỏi heap tương ứng với bước đi có cost nhỏ nhất
        node_action = actions.pop() #Sử dụng hàm pop của cấu trúc để lấy node_action có cost thấp nhất tương ứng với bước đi ở node vừa lấy ở phía trên ra
        if isEndState(node[-1][-1]): #Sử dụng hàm kiểm tra xem tương ứng với bước đi cuối cùng thì hộp đã đạt GoalState chưa
            temp += node_action[1:] #Lấy tất cả các bước đi từ sau StartState
            break #Thoát ra khỏi vòng lập và chuẩn bị trả về
        if node[-1] not in exploredSet: #Nếu như bước đi mới nhất chưa ở trong những bước đi đã đi
            exploredSet.add(node[-1]) #Thì thêm vào những bước đã đi
            for action in legalActions(node[-1][0],node[-1][-1]): #Với mỗi action có thể đi tương ứng được trả về từ hàm legalActions với tham số là vị trí của người chơi và hộp ở bước đi mới nhất
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Update vị trí mới nhất của người chơi và hộp
                if isFailed(node[-1][1]): #Nếu như vị trí của hộp có khả năng fail thì bỏ qua bước đi đó
                    continue
                priority = cost(node_action[1:] + [action[-1]])#Tính priority với nó bằng với cost của tất cả những bước trước đó và bước trong action hiện tại +1
                frontier.push(node + [(newPosPlayer, newPosBox)], priority) #frontier sẽ thêm vào hàng đợi bước đi mới nhất của người chơi và hộp và đặt vị trí dựa theo priority
                actions.push(node_action + [action[-1]], priority) #actions sẽ push node_action tương ứng với bước đi mới nhất của người chơi với priority giống trong heap PriorityQueue
    return temp #trả về bước đi tới goal với cost thấp nhất

def heuristic(posPlayer,posBox):
    distance = 0 #Set distance = 0 as default
    targets = set(posGoals) & set(posBox) #Trả về những vị trí của hộp đã nằm trên goal

    sortposBox = list(set(posBox).difference(targets)) #Trả về một list những hộp không nằm trên goal
    sortposGoals = list(set(posGoals).difference(targets)) #Trả về một list những goal vẫn chưa có hộp nằm ở đấy

    for i in range(len(sortposBox)): #Với mỗi hộp chưa được nằm trên goal
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1])) #Khoảng cách sẽ bằng tổng khoảng cách đường vuông góc của vị trí hộp chưa đuọc đặt trên goal với vị trí goal còn lại
    return distance #Trả về khoảng cách đã tính được cho hàm heuristic

def greedySearch(gameState): # Thuật toán tìm kiếm Tham lam được dựa trên việc lựa chọn bước đi ngắn nhất theo hàm ước lượng heuristic
    beginBox = PosOfBoxes(gameState) #beginBox chứa các cặp vị trí tương ứng của các hộp có trong màn chơi
    beginPlayer = PosOfPlayer(gameState) #beginPlayer chứa cặp vị trí tương ứng của người chơi trong màn

    startState = (beginPlayer, beginBox) #startState là tuple chứa vị trí khởi điểm của người chơi và các hộp
    frontier = PriorityQueue() #Khai báo hàng đợi kiểu PriorityQueue
    frontier.push([startState], heuristic(beginPlayer, beginBox)) #Thêm vào hàng đợi vị trí khởi đầu với Priority là giá trị tính toán của hàm ước lượng khoảng cách heuristic
    exploredSet = set() #Khai báo exploredSet để tránh lập lại những nước đi cũ
    actions = PriorityQueue() #Khai báo actions kiểu PriorityQueue
    actions.push([0], heuristic(beginPlayer, beginBox)) #Thêm vào action tương ứng với vị trí người chơi hiện tại là 0 vào actions với Priority tương ứng với vị trí như trên hàng đợi
    temp = [] #temp sẽ lưu trữ những bước đi đúng để trả về cuối cùng

    while frontier: #Khi mà hàng đợi vẫn còn chứa node
        node = frontier.pop() #Sử dụng hàm pop để lấy node với giá trị Priority thấp nhất ra
        node_action = actions.pop() #Lấy action của node tương ứng ra
        if isEndState(node[-1][-1]): #Nếu như đây là EndState thì sẽ 
            temp += node_action[1:]  #Lấy tất cả những hành động của node kể từ sau vị trí đầu tiên ra           
            break #Và ngưng vòng lập
        if node[-1] not in exploredSet: #Nếu như node được thêm vào cuối cùng chưa được explored thì
            exploredSet.add(node[-1]) #Thêm node đó vào exploredSet 
            for action in legalActions(node[-1][0], node[-1][1]): #Với mỗi hành động có thể làm được và legal tương ứng với vị trí người chơi và vị trí hộp hiện tại được lưu trong node[-1]
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Thì update mới vị trí người chơi và hộp
                if isFailed(newPosBox): #Nếu như vị trí hộp mới có khả năng fail thì
                    continue #Bỏ qua action mới nhất và thử action khác
                Heuristic = heuristic(newPosPlayer, newPosBox) #Tính giá trị ước lượng Heuristic với vị trí hộp và người chơi mới
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic) #Thêm vào hàng đợi bước đi để tới vị trí đó được lưu trong node với Priority là giá trị Heuristic vừa tính được
                actions.push(node_action + [action[-1]], Heuristic) #Thêm vào actions những hành động tương ứng với bước đi tới vị trí hiện tại vào node_action với Priority là giá trị Heuristic vừa tính được
    return temp #Trả về đường đi để đưa node về vị trí của nó



def AstarSearch(gameState): #Thuật toán tìm kiếm A* sử dụng có hàm ước lượng vị trí tương đối là Heuristic và cả giá trị thực tế là Cost để định hướng được vị trí goal và không đi những con đường lạc
    beginBox = PosOfBoxes(gameState) #beginBox chứa các cặp vị trí tương ứng của các hộp có trong màn chơi
    beginPlayer = PosOfPlayer(gameState) #beginPlayer chứa cặp vị trí tương ứng của người chơi trong màn

    startState = (beginPlayer, beginBox) #startState là tuple chứa vị trí khởi điểm của người chơi và các hộp
    frontier = PriorityQueue() #Khai báo hàng đợi kiểu PriorityQueue
    frontier.push([startState], heuristic(beginPlayer, beginBox)) #Thêm vào hàng đợi vị trí khởi đầu với Priority là giá trị tính toán của hàm ước lượng khoảng cách heuristic
    exploredSet = set() #Khai báo exploredSet để tránh lập lại những nước đi cũ
    actions = PriorityQueue() #Khai báo actions kiểu PriorityQueue
    actions.push([0], heuristic(beginPlayer, beginBox)) #Thêm vào action tương ứng với vị trí người chơi hiện tại là 0 vào actions với Priority tương ứng với vị trí như trên hàng đợi
    temp = [] #temp sẽ lưu trữ những bước đi đúng để trả về cuối cùng

    while frontier: #Khi mà hàng đợi vẫn còn chứa node
        node = frontier.pop() #Sử dụng hàm pop để lấy node với giá trị Priority thấp nhất ra
        node_action = actions.pop() #Lấy action của node tương ứng ra
        if isEndState(node[-1][-1]): #Nếu như đây là EndState thì sẽ 
            temp += node_action[1:]  #Lấy tất cả những hành động của node kể từ sau vị trí đầu tiên ra           
            break #Và ngưng vòng lập
        if node[-1] not in exploredSet: #Nếu như node được thêm vào cuối cùng chưa được explored thì
            exploredSet.add(node[-1]) #Thêm node đó vào exploredSet 
            Cost = cost(node_action[1:]) #Giá trị cost được thêm vào sẽ được tính trong hàm cost với những hành động tương ứng kể từ sau bước đi đầu tiên
            for action in legalActions(node[-1][0],node[-1][-1]): #Với mỗi action có thể đi tương ứng được trả về từ hàm legalActions với tham số là vị trí của người chơi và hộp ở bước đi mới nhất
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Update vị trí mới nhất của người chơi và hộp
                if isFailed(node[-1][1]): #Nếu như vị trí của hộp có khả năng fail thì bỏ qua bước đi đó
                    continue
                
                Heuristic = heuristic(newPosPlayer, newPosBox) #Tính giá trị ước lượng Heuristic với vị trí hộp và người chơi mới
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost) #Thêm vào hàng đợi vị trí mới của hộp và người chơi với Priority bằng giá trị ước lượng Heuristic cộng với giá trị thực tế đạt được trước khi đi bước đi đó Cost
                actions.push(node_action + [action[-1]], Heuristic + Cost) #Thêm vào hành động tương ứng với bước đi đạt được của vị trí mới với Priority bằng với Priority đã tính phía trên
    return temp #Trả về đường đi để đưa node về vị trí của nó
"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'greedy':
        result = greedySearch(gameState)
    elif method == 'astar':
        result = AstarSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result
