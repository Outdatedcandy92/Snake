
# Snake Game AI

This is a snake game made in python with pygame, which is played by a AI.  
You can train you custom models or play with a pre trained one



## Installation & Usage

1) Clone & CD into repository

```bash
  > git clone https://github.com/Outdatedcandy92/Snake.git
  > cd Snake
```
2)  Activate virtual enviornment

```bash
  > python -m venv venv
  > .\venv\bin\activate
```
1) Install Requirments & Run

```bash
  > pip install -r requirments.txt
  > python agent.py
```

    
## Variables

Play around with these variables to your liking

#### Agent.Py
`LR` Learning rate of the model

`Max_Memory` Maximum number of values stored in the model

`Batch Size` Number of values the model picks

`EPSILON` Explore vs Exploit. Value should be less than 1

`DR` The discount rate

`HIDDEN_LAYER` Number of neurons in the hidden layer
![Neurons](https://github.com/Outdatedcandy92/Snake/assets/138517406/bffe4de6-3c22-4fd5-9168-f809523a577c)

#### Game.py

`BLOCK_SIZE` controls how big the blocks are.

`SPEED` speed of the snake

`def __init__(self, w=640, h=480):` Set the width and height of screen

`reward` Sets the reward point for the action 

`TEXT_COLOR`
`APPLE_COLOR`
`SNAKE_COLOR`
`BG_COLOR` Set color value in RGB
## Training Data

![plot](https://github.com/Outdatedcandy92/Snake/assets/138517406/537ac5c6-d89b-4ad1-a464-79920ac44fef)



