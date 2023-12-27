# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import NeuralCF
import pandas as pd
import numpy as np
from tqdm.auto import tqdm,trange
import random
import sklearn.preprocessing
rating = pd.read_csv("./ml-100k/u.data", header=None,sep='\t',
                       names=["userid","itemid","rating", "timestamp"])
item=pd.read_csv("./ml-100k/u.item.1",sep='|',header=None,names=["movieid","movietitle","release date","video release date","IMDBUrl","unknown","Action","Adventure","Animation","Children’s","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"])
user=pd.read_csv("./ml-100k/u.user",sep='|',header=None,names=["userid","age","gender","occupation","zip_code"])
item=item.drop("movietitle",axis=1)
item=item.drop("video release date",axis=1)
item:pd.DataFrame=item.drop("IMDBUrl",axis=1)
pd.set_option('display.max_columns', 1000)
occupation_name=["administrator","artist","doctor","educator","engineer","entertainment","executive","healthcare","homemaker","lawyer","librarian","marketing","none","other","programmer","retired","salesman","scientist","student","technician","writer"]
user["occupation"]=user["occupation"].replace(occupation_name,[i for i in range(len(occupation_name))])
user["gender"]=user["gender"].replace({"M":0,"F":1})
item["release date"]=pd.to_datetime(item["release date"]).astype("int64")*(10**-9)
item["release date"]=item["release date"].astype("int64")
# negative = []
# for userId in user["userid"]:
#     negatives = list()
#     momvielist = np.where(rating['userid'].values == userId)[0]
#     while len(negatives) < len(momvielist):
#         movieId = random.randint(1,len(item))
#         movieidlist=rating["itemid"][momvielist].values
#         if movieId not in movieidlist:
#             negative.append([userId,movieId])
#             negatives.append(movieId)
#     print(userId)
# negative = pd.DataFrame(negative,columns=["userid","itemid"])
# negative['labels'] = 0
# rating=rating[["userid","itemid"]]
# rating['labels']=1
# rating=pd.concat([rating,negative])
# rating.to_csv("./after_progressing.csv")


neuralcf=NeuralCF.NeuralCF(rating['userid'].max(),rating['itemid'].max(),50)
rating=pd.read_csv("./after_progressing.csv",index_col=0)
#neuralcf.train_EA(rating,100,100,0.8,0.5,20,20,10000,user,item)
neuralcf.train_loss(rating,20,0.1,user,item)

