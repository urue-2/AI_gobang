from lab2 import next_move
from GUI import *
from lab2.next_move import *


if __name__ == '__main__':
    # 初始化棋局
    a = Chess()

    while a.winner == blank :
        a.humanmove()
        print(a.human_latest_move)
        print(a.matrix)
        if a.winner == human:
            break
        # figure solution


        print(next_move.find_possible(a.matrix, list(a.human_latest_move)))
        p = cal_next_move(cur_state=a.matrix, cur_piece=a.human_latest_move, method="α-β")
        a.aimove(p)
        print("                      "+str(a.winner))

    #message()
    print(str(a.winner) +"                 won")


    a.root.mainloop()









