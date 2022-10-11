# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:10:47 2022

@author: Lee
"""

"""
model.fit(train_set.X, train_set.y, epochs=200, batch_size=32, scheduler='cosine',
          input_time_length=in_size, remember_best_column='valid_misclass',
          validation_data=(valid_set.X, valid_set.y),)
"""

# 모델을 학습을 마친 후, 아래의 코드를 실행
x = torch.tensor(test_set.X, dtype=torch.float32).unsqueeze(-1).cuda()
l = [x]+[None]*(len(model.network)+1)
for name, layer in enumerate(model.network): # 0, conv
    layer.cuda()
    x = layer(x)
    
    l[name+1] = x.cpu().detach().numpy()
    
l[-1] = np.mean(l[11],2) # 가장 마지막 output
y_pred = model.predict_outs(test_set.X) #실제 모델에서 자동으로 계산해주는 것과 동일한 결과를 보이는지 검증을 위해 넣어두었습니다.