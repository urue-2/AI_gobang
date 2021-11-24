import copy
search_area_distance = 2
human = 1
ai = 2
blank = 99

def in_board(pos):
    if pos[0] <0:
        return False
    elif pos[0] > 14:
        return False
    elif pos[1] < 0:
        return False
    elif pos[1] > 14:
        return False
    else:
        return  True


# def evaluate_state():


def find_possible(cur_state,cur_piece):
    global search_area_distance
    re = []
    i,j = cur_piece[0],cur_piece[1]
    xu_bound = max(0,i-search_area_distance)
    xd_bound = min(14,i+search_area_distance)
    yl_bound = max(0,j-search_area_distance)
    yr_bound = min(14,j+search_area_distance)
    for ti in range(xu_bound,xd_bound+1):
        for tj in range(yl_bound,yr_bound+1):
            if cur_state[ti][tj] == 0:
                re.append([ti, tj])

    return re

# 连五 活四 冲四 活三 眠三 活二 眠二
def find_all_lines(board,role):
    board = board.copy()
    re = {5:0,44:0,4:0,33:0,3:0,22:0,2:0}
    start_point_dir = [[1,0],[0,1]]
    dir = [[0,1],[1,0],[1,1],[1,-1]]


    # 横竖两个方向
    for i in range (2):
        start_point = [0,0]
        for j in range(15):


            cur_point = copy.deepcopy(start_point)

            flag1 = 0
            flag1_role = 0 # 99 edge
            count = 0
            for k in range(15):
                if cur_point[0]==15: break
                if board[cur_point[0]][cur_point[1]] == role and flag1 == 0:
                    flag1 = k-1
                    if flag1 == -1:
                        flag1_role = 99
                    else:
                        flag1_role = board[cur_point[0]-dir[i][0]][cur_point[1]-dir[i][1]]
                    count += 1
                elif board[cur_point[0]][cur_point[1]] == role and flag1 != 0:
                    count += 1
                elif board[cur_point[0]][cur_point[1]] != role and flag1 != 0:
                    if flag1_role == 99:
                        num = k
                    else:
                        num = k-flag1
                    if num > 1:
                        if num==5:
                            re[5] = re.get(5,0)+1
                        elif board[cur_point[0]][cur_point[1]] == 0 and flag1_role==0 :
                            #huo num
                            re[num+num*10] = re.get(num+num*10,0)+1
                        else:
                            re[num] = re.get(num,0)+1


                    flag1 = 0
                    flag1_role = 0
                    count = 0

                cur_point[0] = cur_point[0] + dir[i][0]
                cur_point[1] = cur_point[1] + dir[i][1]

            start_point[0] = start_point[0] + start_point_dir[i][0]
            start_point[1] = start_point[1] + start_point_dir[i][1]



    # 左右斜边的两个方向
    start_point = [0,0]
    for i in range(2,4): # 2 右下 1,1            3 1,-1 左下 dir
        start_point = [0, 0]
        for j in range(2): #j  0 下 1,0  1 右0,1 start_point_direction

            for k in range(15):
                cur_point = copy.deepcopy(start_point)
                flag1 = 0
                flag1_role = 0
                count = 0
                for l in range(0,15-k+1):
                    if cur_point[0] == 15 or cur_point[1] == 15 : break
                    if board[cur_point[0]][cur_point[1]] == role and flag1 == 0:

                        flag1 = l - 1
                        if flag1 < 0:
                            flag1_role = 99
                        else:
                            flag1_role = board[cur_point[0] - dir[i][0]][cur_point[1] - dir[i][1]]
                        count += 1
                    elif board[cur_point[0]][cur_point[1]] == role and flag1 != 0:
                        count += 1
                    elif board[cur_point[0]][cur_point[1]] != role and flag1 != 0:
                        if flag1 == -1:
                            num = l
                        else:
                            num = l - flag1
                        if num > 1:
                            if num == 5:
                                re[5] = re.get(5, 0) + 1
                            elif board[cur_point[0]][cur_point[1]] == 0 and flag1_role == 0:
                                # huo num
                                re[num + num * 10] = re.get(num + num * 10, 0) + 1
                            else:
                                re[num] = re.get(num, 0) + 1

                        flag1 = 0
                        flag1_role = 0
                        count = 0

                    cur_point[0] = cur_point[0] + dir[i][0]
                    cur_point[1] = cur_point[1] + dir[i][1]

                start_point[0] = start_point[0]+start_point_dir[j][0]
                start_point[1] = start_point[1] + start_point_dir[j][1]

    return re

def evaluate_board(tboard):
    #[3,4,3.5,3,2.5,2,2] Human:[5,4.5,4,3.5,3,2.8,2.5]
    #Ai:[5,4.5,3.8,3.5,3.2,2.9,2.5] ]Human:[5, 4.8 ,4 ,3 .5 ,3 ,2.8  ,2.5]

    paraai = {5:5,44:4.5,4:3.8,33:3.5,3:3.2,22:2.9,2:2.5}
    parahuman = {5: 5, 44: 4.8, 4: 4, 33: 3.5, 3: 3, 22: 2.8, 2: 2.5}
    po = find_all_lines(board=tboard , role= ai)
    po_sorce = 0
    for key in paraai.keys():
        po_sorce += paraai[key]*po[key]
    ne = find_all_lines(board=tboard , role= human)
    ne_sorce = 0
    for key in parahuman.keys():
        ne_sorce += parahuman[key] * ne[key]

    return po_sorce-ne_sorce

# cur_state np   cur_piece list
def α_β_prun(cur_state,cur_piece):
    ##  α_β_pruning for two level
    global  search_area_distance
    tboard = cur_state.copy()


    tboard1 = tboard.copy()
    close_table = [] #[[[x,y],s],...]
    # find possible move for ai:
    open_table = find_possible(cur_state=cur_state,cur_piece=cur_piece)
    print("open_table",open_table)
    cur_node = open_table[0]
    trole = ai
    tboard1[cur_node[0]][cur_node[1]] = trole
    close_table.append(cur_node)
    cur_node_pos = cur_node
    del open_table[0]

    # calculate the first ai move's min
    min_e = float('inf')
    opp_possible = find_possible(cur_state=cur_state,cur_piece=cur_node)
    for opp_one in opp_possible:
        trole = human
        tboard1[opp_one[0]][opp_one[1]] = trole
        tscore = evaluate_board(tboard1)
        if  tscore < min_e:
            min_e = tscore
    close_table[0].append([min_e])
    cur_node_score = min_e

    print("close_table",close_table)


    # min_e is now the value of the first branch
    for (index,cur_node) in enumerate(open_table):
        tboard2 = tboard.copy()
        close_table.append(cur_node)
        del open_table[index]
        trole = ai
        tboard2[cur_node[0]][cur_node[1]] = trole

        opp_possible = find_possible(cur_state=tboard,cur_piece=cur_node)
        ts = 0
        for opp_one in opp_possible:
            trole = human
            tboard2[opp_one[0]][opp_one[1]] = trole
            ts = evaluate_board(tboard2)
            if   ts <= cur_node_score:
                break

        close_table[-1].append([ts])

    print("close_table", close_table)
    for point in close_table:
        if point[-1][0]> cur_node_score:
            cur_node_score = point[-1][0]
            cur_node_pos = point[:-1]

    print(cur_node_pos,cur_node_score)
    print("after pru",cur_state)

    return cur_node_pos

# def MCTS(cur_state):
#     return

def cal_next_move(cur_state,cur_piece,method):
    tboard = cur_state.copy()
    if method == "α-β":
        next_move = α_β_prun(cur_state=tboard,cur_piece=cur_piece)
    print("ai next move will be",next_move)


    return next_move



