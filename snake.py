import pygame
import time
import random
import sys
# Initialize the game
pygame.init()

# Set up the game window
window_width = 300
window_height = 300
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Snake Game")

# Set up the colors
blue = pygame.Color(0, 0, 255)
green = pygame.Color(0, 255, 0)
red = pygame.Color(255, 0, 0)

# Set up the snake and food
snake_position = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50]]
food_position = [random.randrange(1, (window_width // 10)) * 10, random.randrange(1, (window_height // 10)) * 10]
food_spawn = True

# Set up the game clock
clock = pygame.time.Clock()

# Set up the game variables
direction = 'RIGHT'
change_to = direction
score = 0

# Set up the game over function
def game_over():
    font = pygame.font.SysFont('Arial', 30)
    game_over_text = font.render('Game Over!', True, red)
    game_over_rect = game_over_text.get_rect()
    game_over_rect.midtop = (window_width / 2, window_height / 4)
    window.blit(game_over_text, game_over_rect)
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()
def get_direction_towards_apple(snake_head, apple_position):
    dx = apple_position[0] - snake_head[0]
    dy = apple_position[1] - snake_head[1]
    directions = []
    if dx > 0: directions.append('RIGHT')
    elif dx < 0: directions.append('LEFT')
    if dy > 0: directions.append('DOWN')
    elif dy < 0: directions.append('UP')
    return directions

def is_direction_safe(direction, snake, walls):
    # Implement logic to check if moving in the given direction would hit the snake itself or walls
    # Return True if safe, False otherwise
    pass

# Modify the game loop to use the new logic
# Assuming the helper functions are defined here

while True:
    # Event handling loop (simplified for brevity)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Automatic direction decision logic
    snake_head = snake_body[0]
    preferred_directions = get_direction_towards_apple(snake_head, food_position)
    for direction in preferred_directions:
        if is_direction_safe(direction, snake_body, walls=window_width, window_height):
            change_to = direction
            break

    # Update the snake's direction based on `change_to`
    if change_to == 'RIGHT' and not direction == 'LEFT':
        direction = 'RIGHT'
    elif change_to == 'LEFT' and not direction == 'RIGHT':
        direction = 'LEFT'
    elif change_to == 'UP' and not direction == 'DOWN':
        direction = 'UP'
    elif change_to == 'DOWN' and not direction == 'UP':
        direction = 'DOWN'

    # Update snake position, check for collisions, update the screen, etc.

    pygame.display.flip()
    clock.tick(30)  # Adjust as necessary for your game's desired speed