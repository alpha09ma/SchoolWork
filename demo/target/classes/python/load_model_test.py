import time
import pandas as pd
import numpy as np
import torch
from recbole.data.interaction import Interaction
import recbile.dataset
# config = Config(
#         model=NCL,
#         dataset=args.dataset,
#         config_file_list=args.config_file_list
#     )
# model= NCL(config, train_data.dataset)
model2=torch.load("D:\\各种任务作业大集合\\大数据技术课程实践\\图神经网络模型\\NCL\\NCL-master\\saved\\NCL-final.pth")
#print(np.array(2)[np.newaxis,:])
data=pd.DataFrame({"user_id":19,"item_id":88},index=[0])
result=model.predict(Interaction(data))
print(result)
# model.load_state_dict(checkpoint["state_dict"])
# model.load_other_parameter(checkpoint.get("other_parameter"))