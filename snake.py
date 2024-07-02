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

# Set up the game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                change_to = 'RIGHT'
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                change_to = 'LEFT'
            if event.key == pygame.K_UP or event.key == ord('w'):
                change_to = 'UP'
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                change_to = 'DOWN'

    # Validate the direction
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'

    # Update the snake position
    if direction == 'RIGHT':
        snake_position[0] += 10
    if direction == 'LEFT':
        snake_position[0] -= 10
    if direction == 'UP':
        snake_position[1] -= 10
    if direction == 'DOWN':
        snake_position[1] += 10

    # Snake body mechanism
    snake_body.insert(0, list(snake_position))
    if snake_position[0] == food_position[0] and snake_position[1] == food_position[1]:
        score += 1
        food_spawn = False
    else:
        snake_body.pop()

    # Spawn new food
    if not food_spawn:
        food_position = [random.randrange(1, (window_width // 10)) * 10, random.randrange(1, (window_height // 10)) * 10]
    food_spawn = True

    # Draw the snake and food
    window.fill(blue)
    for pos in snake_body:
        pygame.draw.rect(window, green, pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(window, red, pygame.Rect(food_position[0], food_position[1], 10, 10))

    # Game over conditions
    if snake_position[0] < 0 or snake_position[0] > window_width - 10:
        game_over()
    if snake_position[1] < 0 or snake_position[1] > window_height - 10:
        game_over()
    for block in snake_body[1:]:
        if snake_position[0] == block[0] and snake_position[1] == block[1]:
            game_over()

    # Update the game display
    pygame.display.update()

    # Set the game speed
    clock.tick(15)