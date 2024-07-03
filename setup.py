import time
import os

print("Welcome to Snake AI!")
time.sleep(0.5)
print("Please select an option:")
time.sleep(0.5)
print("1. Train")
print("2. Play Custom Model")
print("3. Exit")
user_choice = input("Enter your choice: ")

if user_choice == '1':
    print("Training...")
    os.system('python agent.py')
elif user_choice == '2':
    print("Warning! This will replace the current model you have")
    time.sleep(0.5)
    file_input = input("Enter the file path (should be a .pth file): ")
    file_name = "model.pth"
    model_folder = "model"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    os.rename(file_input, os.path.join(model_folder, file_name))

elif user_choice == '3':
    print("Exiting...")
    time.sleep(0.5)
    exit()

print(f"You selected: {user_choice}")