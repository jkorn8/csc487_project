import os

for folder in os.listdir('./app_data'):
    if os.path.isdir(f'./app_data/{folder}'):
        print(folder)
