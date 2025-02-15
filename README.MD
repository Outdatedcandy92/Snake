
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
  > python run.py
```

    
# Variables

Play around with these variables to your liking

## Agent.Py

#### `LR` Learning rate of the model 

- Default Value (0.001) 
- Images here
 


#### `Max_Memory` Maximum number of values stored in the model

- Default Value 100_000

#### `Batch Size` Number of values the model picks

- Default Value 1000

#### `EPSILON` Explore vs Exploit

- Default Value = 0 (No Exploration)
- Introduces Randomness into the model
- Value should be less than `1` and greater than `0`

#### `DR` The discount rate

- Default Value = 0.9
- Determines how important rewards are for the model
- Value should be between `1` and `0`

#### `HIDDEN_LAYER` Number of neurons in the hidden layer

- Default Value = 256
![Neurons](https://github.com/Outdatedcandy92/Snake/assets/138517406/bffe4de6-3c22-4fd5-9168-f809523a577c)

#### `MODE` Training vs Using

- Default Value = 1 (Train)
- 2 = playing a model

## Game.py

#### `BLOCK_SIZE` 
- Default value = 20
- Controls size of the blocks
- Greater value means bigger snake 

`SPEED` 
- Default value = 80
- Controls the speed of the game

`WIDTH, HEIGHT` Set the width and height of screen
- Default value = 400 x 400
- Sets the game window size

`reward` Sets the reward point for the action 

`TEXT_COLOR`
`APPLE_COLOR`
`SNAKE_COLOR`
`BG_COLOR` Set custom color values


## Training Data

![plot](https://github.com/Outdatedcandy92/Snake/assets/138517406/3644ec1b-351b-4245-b272-a004b5852f68)
Total Time: 965.2057483196259  
Settings:  
 MAX_MEMORY: 300000  
 BATCH_SIZE: 3000  
 LR: 0.001  
 EPSILON: 0  
 DR: 0.9  
 HIDDEN_LAYER: 300  




## Demo

![ezgif-4-40f9153ba1-ezgif com-video-to-gif-converter](https://github.com/Outdatedcandy92/Snake/assets/138517406/67de9adb-50a3-4a77-8c3b-96f2d23234da)


## Issues

#### There are still issues with loadnig up models which will be fixed in the near future

