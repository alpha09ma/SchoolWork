import sklearn.svm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import gc
import EA
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
import sklearn.metrics

import matplotlib.pyplot as plt
class NeuralCFModule(nn.Module):
    def __init__(self,userIds,itemIds,hidden_nums):
        super(NeuralCFModule, self).__init__()
        self.useridEmb=nn.Embedding(userIds,hidden_nums)#module.MLP(userIds,hidden_nums)
        self.itemidEmb=nn.Embedding(itemIds,hidden_nums)#module.MLP(itemIds,hidden_nums)
        self.useroccupationEmb=nn.Embedding(21,10)
        self.userGenderEmb=nn.Embedding(2,2)

        self.sequential = nn.Sequential(
            nn.Linear(hidden_nums*2+18+2+10, hidden_nums),
            nn.ReLU(),
            nn.Linear(hidden_nums,25),
            nn.ReLU(),
            nn.Linear(25,1),
            nn.Sigmoid()
        )
    def forward(self,x,gender,occupation,type):
        #left=self.userid_vec.transform(x[:, 1].numpy().reshape(-1, 1)).toarray()
        #right=self.movieid_vec.transform(x[:, 0].numpy().reshape(-1,1)).toarray()
        userid = torch.LongTensor(x[:,0].numpy()-1)#left
        itemid = torch.LongTensor(x[:,1].numpy()-1)#right
        userid :torch.Tensor= self.useridEmb(userid)
        itemid = self.itemidEmb(itemid)
        usergender=self.userGenderEmb(gender)
        useroccupation=self.useroccupationEmb(occupation)
        x=torch.cat([userid,usergender,useroccupation,itemid,type],dim=1)
        y=self.sequential(x)
        return y
    def set_layer(self, layers,useridEmb,itemidEmb,usergenderemb,occupationemb):
        self.sequential = layers
        self.useridEmb = useridEmb
        self.itemidEmb = itemidEmb
        self.userGenderEmb = usergenderemb
        self.useroccupationEmb = occupationemb
class MovieDataset(Dataset):
    def __init__(self, x,gender,occupation,type,y):
        super(MovieDataset, self).__init__()
        self.x = x
        self.y = y
        self.gender=gender
        #self.relesedate=releasedate
        self.occupation=occupation
        self.type=type
    def __getitem__(self, idx):

        return self.x[idx], self.y[idx],self.gender[self.x[idx][0]-1],self.occupation[self.x[idx][0]-1],self.type[self.x[idx][1]-1]

    def __len__(self):
        return len(self.x)
class NeuralCF:
    def __init__(self,userIds,itemIds,hidden_nums):
        self.neuralcf=NeuralCFModule(userIds,itemIds,hidden_nums)
        self.userids=userIds
        self.itemids=itemIds
        self.hidden_nums=hidden_nums
        #self.svc = sklearn.svm.SVC()

    def train_loss(self,data,epoch,learning_rate,user,item):
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:2], data.iloc[:, 2], test_size=0.15,random_state=2022,stratify=data.iloc[:, 2])
        train_dataset = MovieDataset(torch.tensor(np.array(x_train.values)),torch.tensor(np.array(user["gender"])),torch.tensor(np.array((user["occupation"]))),torch.tensor(np.array(item.iloc[:,3:])),torch.tensor(np.array(y_train.values)))#,torch.tensor(np.array(item["release date"]))
        train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
        test_dataset = MovieDataset(torch.tensor(np.array(x_test.values)), torch.tensor(np.array(user["gender"])),
                                     torch.tensor(np.array((user["occupation"]))),
                                     torch.tensor(np.array(item.iloc[:, 3:])), torch.tensor(
                np.array(y_test.values)))  # ,torch.tensor(np.array(item["release date"]))
        test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True)
        for i in range(0,epoch):
            running_loss = []
            total_loss = 0
            total_accuray = []
            bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
            self.neuralcf.train()

            for step, (x,y,gender,occupation,type) in bar:
                output = self.neuralcf(x,gender,occupation,type)
                #self.svc.fit(output.detach().numpy(),y.detach().numpy())
                accurancy = self.getAccuracy(output,torch.FloatTensor(y.numpy()))
                total_accuray.append(accurancy)
                loss:torch.Tensor = nn.BCELoss()(output.view(-1),torch.FloatTensor(y.numpy()))#torch.norm(torch.sub(y,output), p=2)
                total_loss += loss.data
                running_loss.append(loss.clone().detach().numpy())

                loss.backward()
                if (i < 5):
                    optimizer = torch.optim.Adam(self.neuralcf.parameters(), lr=learning_rate * (0.9 ** i))
                else:
                    optimizer = torch.optim.Adam(self.neuralcf.parameters(), lr=learning_rate * (0.9 ** 5))
                optimizer.step()
                optimizer.zero_grad()

                bar.set_description(f'Epoch: [{i}/{epoch}]')
                bar.set_postfix(Epoch=epoch, Train_loss=total_loss/(step+1),Train_accuracy=sum(total_accuray)/(step+1))

                gc.collect()
            fig=plt.figure(figsize=(30, 30))
            plt.title("train_loss_loss")
            plt.plot(np.array(range(len(running_loss))), np.array(running_loss))
            plt.savefig("./picture/train_loss_loss{}.jpg".format(i))
            plt.close(fig)
            self.valid_one_epoch(i,test_loader)
            fig=plt.figure(figsize=(30, 30))
            plt.title("train_loss_acc")
            plt.plot(np.array(range(len(total_accuray))), np.array(total_accuray))
            plt.savefig("./picture/train_loss_acc{}.jpg".format(i))
            plt.close(fig)
    def train_EA(self,data,epoch,pop_size,p_mutation,r_mutation,elite_num,poolsize,batch_size,user,item):
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:2], data.iloc[:, 2], test_size=0.15,
                                                            random_state=2022, stratify=data.iloc[:, 2])
        train_dataset = MovieDataset(torch.tensor(np.array(x_train.values)), torch.tensor(np.array(user["gender"])),
                                     torch.tensor(np.array((user["occupation"]))),
                                     torch.tensor(np.array(item.iloc[:, 3:])), torch.tensor(
                np.array(y_train.values)))  # ,torch.tensor(np.array(item["release date"]))
        train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
        test_dataset = MovieDataset(torch.tensor(np.array(x_test.values)), torch.tensor(np.array(user["gender"])),
                                    torch.tensor(np.array((user["occupation"]))),
                                    torch.tensor(np.array(item.iloc[:, 3:])), torch.tensor(
                np.array(y_test.values)))  # ,torch.tensor(np.array(item["release date"]))
        test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True)
        ea=EA.ElitismGA(
            _pop_size=pop_size,
            _p_mutation=p_mutation,
            _r_mutation=r_mutation,
            _epochs=epoch,
            _elite_num=elite_num,
            _mating_pool_size=poolsize,
            _batch_size=batch_size
        )
        ea.initialization(NeuralCFModule, self.userids, self.itemids, self.hidden_nums)
        for epoch in range(epoch):
            ea.train(train_loader)
            ea.test(test_loader)
    def test(self):
        pass
    def getAccuracy(self,outputs, labels):
        output = list()
        #y=self.svc.predict(outputs.detach().numpy())
        for i in outputs:
            if i < 0.5:
                output.append(0)
            else:
                output.append(1)

        return (torch.tensor(output).view(-1).data== labels.data).sum().item()/outputs.shape[0]#sklearn.metrics.accuracy_score(y,labels.data)
    def valid_one_epoch(self,epoch,dataLoader):
        self.neuralcf.eval()
        bar = tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader))
        total_accuracy = 0
        item = 0
        for step, (x,y,gender,occupation,type) in bar:
            item += 1
            outputs = self.neuralcf(x,gender,occupation,type)
            total_accuracy += self.getAccuracy(outputs, y)
            bar.set_description(f'Epoch: [{epoch}/{20}]')
            bar.set_postfix(Epoch=epoch, Accuracy=total_accuracy / item)
        gc.collect()
        return total_accuracy


# x = np.array ([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]])
# labels=np.array ([0, 1, 0, 1, 0, 0, 0, 1, 1, 1])
# neural=NeuralCF(10,10,5)
# neural.train_loss(x,10,torch.unsqueeze(torch.FloatTensor(labels),0))