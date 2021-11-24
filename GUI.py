from tkinter import *
from tkinter.messagebox import *
import numpy as np
import  time
import math

human = 1
ai = 2
blank = 99
# 人1 白   ai 2 黑

class Chess(object):

    def __init__(self):

        ## param
        self.row , self.column = 15,15
        self.matrix = np.zeros((15,15))
        self.currole = human
        self.inhuman = False
        self.winner = blank
        self.human_latest_move = np.array([0,0])


        ## gui
        self.root = Tk()
        self.root.title("Gobang")
        self.canvas = Canvas(self.root , bg = "SandyBrown", width = 480, height = 480)
        self.canvas.grid(row = 0, column = 0, rowspan = 10)
        self.r = 10

        for i in range(15):
            self.canvas.create_line(30, (30 * i + 30), 450, (30 * i + 30))
            self.canvas.create_line((30 * i + 30), 30, (30 * i + 30), 450)

        point_x = [3, 3, 11, 11, 7]
        point_y = [3, 11, 3, 11, 7]
        for i in range(5):
            self.canvas.create_oval(30 * point_x[i] + 27, 30 * point_y[i] + 27,
                               30 * point_x[i] + 33, 30 * point_y[i] + 33, fill="black")


        #gui para
        self.click_x = None
        self.click_y = None
        self.canvas.bind("<Button-1>", self.clickpointget)



    def processclick(self,x,y):
        i = 0
        j = 0
        while x > (30 + 30 * i):  # 确定鼠标点击位置所在的格子
            i += 1
        if math.fabs(x-(30 + 30 * i)) > math.fabs(x-(30 + 30 * (i-1))):
            i = i-1
        while y > (30 + 30 * j):
            j += 1
        if math.fabs(y-(30 + 30 * j)) > math.fabs(y-(30 + 30 * (j-1))):
            j = j-1
        return j,i


    def putpiece(self,x,y,role):
        if role == human:
            piece_color = "white"
        else:
            piece_color = "black"

        tx = 30 * (y+1)
        ty = 30 * (x+1)

        self.canvas.create_oval(tx - self.r, ty - self.r,
                           tx + self.r, ty + self.r,
                           fill=piece_color)

    def clickpointget(self,event):
        print("in clickget")
        if self.inhuman == True:
            print("in clickpointget inhuman ")
            self.root.update()
            self.click_x = event.x
            self.click_y = event.y
            print(event.x, event.y)
        print("leave clickpointget")

    def humanmove(self): # 1 白
        print("in human")
        self.currole = human
        self.inhuman = True

        while self.click_x == None and self.click_y == None:
            self.root.update()
        tx , ty = self.click_x,self.click_y
        print(tx,ty)
        # 预处理x y
        x , y = self.processclick(tx,ty)
        print("after pro",x,y)

        #下棋
        if self.matrix[x,y] == 0:
            #成功下棋
            self.human_latest_move = np.array([x,y])
            self.putpiece(x, y, human)
            self.matrix[x,y] = 1
            if self.judge_fiveinline((x,y),human):
                self.winner = human

            self.currole = blank
            self.click_x = None
            self.click_y = None

        else:
            print("人 换个位置")
        #判断输赢
        print("leave human")
        #before leave humanmove
        self.inhuman = False


    def aimove(self,pos): # -1 黑
        print("in ai")
        self.currole = ai
        x,y = pos[0] , pos[1]
        print("before ai go",self.matrix)
        if self.matrix[x,y] ==0:

            self.putpiece(x, y,ai)
            self.matrix[x,y]=2

            if self.judge_fiveinline((x,y),ai):
                self.winner = ai

            self.currole = blank
        else:
            print("ai 要换个位置")
        print("leave ai")

        #判断输赢

    def judge_fiveinline(self,pos,role):
        x = pos[0]
        y = pos[1]
        direction = [[(0,-1),(0,1)],[(-1,0),(1,0)],[(-1,-1),(1,1)],[(1,-1),(-1,1)]]
        for i in range(4):
            tx1 , ty1 = x, y
            tx2 , ty2 = x, y
            count = 1
            for j in range(4):
                tx1 += direction[i][0][0]
                ty1 += direction[i][0][1]
                tx2 += direction[i][1][0]
                ty2 += direction[i][1][1]
                if (not (tx1<0)) and (not (tx1>14)) and (not (ty1<0)) and (not (ty1>14)) and self.matrix[tx1,ty1] == role :
                    count += 1
                if (not (tx2<0)) and (not (tx2>14)) and (not (ty2<0)) and (not (ty2>14)) and self.matrix[tx2,ty2] == role :
                    count += 1
            print("count                   "+str(count))
            if not (count < 5):
                return True
        return False














