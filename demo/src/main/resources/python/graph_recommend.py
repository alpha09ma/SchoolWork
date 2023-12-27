import time
import pandas as pd
import numpy as np
import torch
from recbole.data.interaction import Interaction
import sys
import joblib
import heapq
from sklearn.preprocessing import LabelEncoder
import pymongo as mongo
if __name__ == '__main__':
    user_label=joblib.load("D:\\各种任务作业大集合\\大数据技术课程实践\\图神经网络模型\\user.pkl")
    item_label=joblib.load("D:\\各种任务作业大集合\\大数据技术课程实践\\图神经网络模型\\item.pkl")
    # user_id=user_label.transform([user_id])
    # user_id=[user_id[0] for i in range(0,item_id)]
    # item_id = [i for i in range(0, item_id)]
    model = torch.load("D:\\各种任务作业大集合\\大数据技术课程实践\\图神经网络模型\\NCL\\NCL-master\\saved\\NCL-final.pth")
    user_all_embeddings, item_all_embeddings, embeddings_list=model.forward()
    item_all_embeddings=item_all_embeddings / torch.norm(item_all_embeddings, dim=-1, keepdim=True)
    item_sim=torch.abs(torch.mm(item_all_embeddings,item_all_embeddings.T)).to(torch.device("cpu")).detach().numpy()
    mongoclient = mongo.MongoClient('127.0.0.1')
    db = mongoclient['recommend']
    dbcol = db["movie_sim"]
    item_id=item_label.inverse_transform([i for i in range(0,len(item_sim)-1)])
    j=0
    for row in range(0,len(item_sim)-1):
        j+=1
        recs = []
        for i in range(0, len(item_sim) - 1):
            recs.append({"mid": int(item_id[i]), "score": float(item_sim[row][i])})
        dbcol.insert_one({"mid":int(item_id[row]),"recs":recs})
        del recs
        # if j==1:
        #     break


