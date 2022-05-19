"""

Just for test draft code, not used for proposed method.

"""



from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms

img = Image.open('../data/kodak24/kodim01.png')
convert_tensor = transforms.ToTensor()
x = convert_tensor(img)
# print(x)
# print(x.size()[1])
crop1 = x[:, 0:256, 0:256]
crop2 = x[:, 256:512, 0:256]
crop3 = x[:, 0:256, 256:512]
crop4 = x[:, 256:512, 256:512]
crop5 = x[:, 0:256, 512:768]
crop6 = x[:, 256:512, 512:768]

blocks = []
blocks.extend([crop1, crop2, crop3, crop4, crop5, crop6])
#print(blocks)
i = 0
for b in blocks:
    
    print("b is \n", b)
    print("num is ", i)
    name = str(i) + '.jpg'
    item = transforms.functional.to_pil_image(b)
    item.save(name)
    i += 1
    
# 합치기

row1 = torch.cat([crop1, crop3, crop5], dim=2)
row2 = torch.cat([crop2, crop4, crop6], dim=2)
col = torch.cat([row1, row2], dim=1)
col = transforms.functional.to_pil_image(col)
col.save('full.jpg')


# crop1 = transforms.functional.to_pil_image(crop1)
# crop2 = transforms.functional.to_pil_image(crop2)
# crop3 = transforms.functional.to_pil_image(crop3)
# crop4 = transforms.functional.to_pil_image(crop4)
# crop5 = transforms.functional.to_pil_image(crop5)
# crop6 = transforms.functional.to_pil_image(crop6)


# crop1.save('1.jpg')
# crop2.save('2.jpg')
# crop3.save('3.jpg')
# crop4.save('4.jpg')
# crop5.save('5.jpg')
# crop6.save('6.jpg')

