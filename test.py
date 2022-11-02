model_path = "rsp_model.pt"

import os
import sys

from PIL import Image
import torch
from torchvision import transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
class_names = ['paper', 'rock', 'scissors'] 

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_advantageous_answer(img_value: str):
  # 0(주먹) 1(가위) 2(보)
  if img_value == 'paper':
    return 1
  elif img_value == 'rock':
    return 2
  elif img_value == 'scissors':
    return 0

# 파일에 결과를 적는 함수 
def write_file(result_dict: dict):
  result_dict_items = sorted(result_dict.items(), key=lambda v: v[0])

  with open('output.txt', 'w') as f:
    for key, value in result_dict_items:
      f.write(str(key)+'\t'+str(value)+'\n')


def main():
  if device == 'cpu':
    raise Exception('GPU로 실행해야 합니다.')

  img_file_path = sys.argv[1]
  
  model = torch.load('rsp_model.pt')
  model.eval()

  result = {}

  for img_filename in os.listdir(img_file_path):
    img_path = os.path.join(img_file_path, img_filename)
    if os.path.isfile(img_path):
      image = Image.open(img_path).convert('RGB')
      image = transforms_test(image).unsqueeze(0).to(device)

      with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

        answer = get_advantageous_answer(class_names[preds[0]])

        result[img_filename.split('.')[0]] = answer


  write_file(result)

main()