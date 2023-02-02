from pybo.model import *

import os
from PIL import Image

X,y,X_train,X_test,y_train,y_test,labels_l,labels_r,idx,data = load_data()

model=Model().to(device)
criterion=MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
batch_size=32
epochs=250


for _ in tqdm(range(epochs)):
     for i in range(0,len(X_train),batch_size):
         X_batch = X_train[i:i+batch_size]
         y_batch = y_train[i:i+batch_size]
         model.to(device)
         preds = model(X_batch)
         loss = criterion(preds,y_batch)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         torch.cuda.empty_cache()
     torch.cuda.empty_cache()
     model.train()
wandb.finish()


img_list=[]    
folder_path='/content/drive/MyDrive/Find my lover/women/'
for i in range(50):
  img_path=str(i+1)+".jpg"
  img = cv2.imread(folder_path+img_path)
  img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



real_data, img_list=load_men_data()
predict=model(real_data)


fig, axs=plt.subplots(10, 5, figsize=(30, 100))
for i in range(10):
  for j in range(5):
    axs[i, j].imshow(img_list[i*5+j])
    axs[i, j].set_title(labels[torch.argmax(predict, axis=1)[i*5+j]], size=20)


label=torch.argmax(predict, axis=1).tolist()


for i in range(len(label)):
  folder="/content/drive/MyDrive/Find my lover/predicted_women/"+labels[label[i]]
  if(not os.path.isdir(folder)):
    os.mkdir(folder)
  cv2.imwrite(folder+"/"+str(i)+".jpg", img_list[i])
  