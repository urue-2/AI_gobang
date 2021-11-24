#<font color=white>Readme</font>

##<font color=white>GoBang Game</font>

###<font color=white> Explanation</font>

    1.This is a gobang game with GUI
    2.Human always first with white piece,AI with black piece
    3.If either side won the game,the game stops,and you can not put another piece anymore


### <font color=white>To  Play</font>
    With α_β_pruning method with GUI : run main.py

### <font color=white>To Read The code</font>

    GUI.py : GUI Graphical User Interface for this game

    1)-3) labs

    1) Using supervised learning to identify the position

        poistion_cnnmodel.py : CNN model (trained model stored in model.pth)
        generate_train_cnn_data : randomly generate positions  
        poisition_abstract.py : using trained model to identify piece postion
        model.pth : model for position identification

    2) Game search algorithm to generate next move for AI

        next_move.py : using α_β pruning method to search for solution

    3) ANN to generate solution :
        
        ANN_next_move.py : using ANN to generate next move
        generate_train_gameCNN_data : prepare data for CNN



        
