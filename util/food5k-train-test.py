import os
import shutil

def moveFiles(path):
  if not os.path.isdir(os.path.join(path,"not_food")):
    os.mkdir(os.path.join(path,"not_food"))
    
  if not os.path.isdir(os.path.join(path,"food")):
    os.mkdir(os.path.join(path,"food"))
    
  for file in os.listdir(path):
    if file.startswith('0'):
      shutil.move(os.path.join(path,file), os.path.join(path,"not_food"))
    elif file.startswith('1'):
      shutil.move(os.path.join(path,file), os.path.join(path,"food"))

moveFiles('data/training')
moveFiles('data/evaluation')
moveFiles('data/validation')

os.rename('data/training','data/train')
os.rename('data/evaluation','data/test')
os.rename('data/validation','data/valid')