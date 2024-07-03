import time
import os
import subprocess

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
    subprocess.run(['python', 'agent.py'])
elif user_choice == '2':
    print("Select Your Custom Model Name")
    time.sleep(0.2)
    file_list = os.listdir("model")
    for file in file_list:
        print(file)
    file_input = input("Enter the file name: ")
    model_folder = "model"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    print(os.path.join(model_folder, file_input))

elif user_choice == '3':
    print("Exiting...")
    time.sleep(0.5)
    exit()
