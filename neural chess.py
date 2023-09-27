import chess
import chess.engine
import numpy as np
import random
import tensorflow as tf
import chess.svg
from keras.layers import Conv2D, Flatten, Dense
from keras import models, layers, optimizers
import keras.callbacks


def random_board(max_depth=200): #random position
    board = chess.Board()
    depth = random.randrange(0, max_depth)

    for _ in range(depth):
        move = random.choice(list(board.legal_moves))
        board.push(move)
        if board.is_game_over():
            break

    return board

def stockfish(board, depth): #stockfish evaluation
    with chess.engine.SimpleEngine.popen_uci('/opt/homebrew/Cellar/stockfish/16/bin/stockfish') as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        return result['score'].white().score(mate_score=10000)
        return score
    

board = random_board()
#print(board)
#print(stockfish(board, 20))

#covert random into 3D matrix
square_to_index = {
    'a':0 , 'b':1 , 'c':2 , 'd':3 , 'e':4 , 'f':5 , 'g':6 , 'h':7
}

def board_to_matrix(square):
    pgn = chess.square_name(square)
    return 8 - int(pgn[1]), square_to_index[pgn[0]]

def split_dims(board):
    board3d = np.zeros((14, 8, 8), dtype=np.uint8)

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1

        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        row, col = board_to_matrix(move.to_square)
        board3d[12][row][col] = 1

    board.turn = chess.BLACK
    for move in board.legal_moves:
        row, col = board_to_matrix(move.to_square)
        board3d[13][row][col] = 1
    
    board.turn = aux
    #print(board3d)
    return board3d

split_dims(board)

#training datasets with tensorflow
def build_model(conv_size, conv_depth):
  board3d = layers.Input(shape=(14, 8, 8))
  x = board3d
  for _ in range(conv_depth):
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, 'relu')(x)
  x = layers.Dense(1, 'sigmoid')(x)

  return models.Model(inputs=board3d, outputs=x)

model = build_model(32, 4)

def get_dataset():
    container = np.load('dataset.npz')
    x = container['arr_0']
    y = container['arr_1']
    x_train = x[:int(len(x) * 0.9)]
    y_train = y[:int(len(y) * 0.9)]
    print(x.shape)
    print(y.shape)

    return x_train, y_train

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
x_train, y_train = get_dataset()
checkpoint_filepath = '/tmp/checkpoint'
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

model.save('model.h5')

model = models.load_model('model.h5')
def eval_board(board):
    x = split_dims(board)
    x = np.expand_dims(x, axis=0)
    return model.predict(x)[0][0]

def play_game():
    max_moves = None
    max_eval = -np.inf

    for move in board.legal_moves:
        board.push(move)
        e = eval_board(board)
        if e > max_eval:
            max_eval = e
            max_moves = [move]
        elif e == max_eval:
            max_moves.append(move)
        board.pop()

        return random.choice(max_moves)
    
board = chess.Board()
board.svg = chess.svg.board(board=board)
with open('board.svg', 'w') as f:
    f.write(board.svg)
board.push(play_game())