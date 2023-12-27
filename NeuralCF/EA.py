# -*- coding: utf-8 -*-

import numpy as np
import copy
import torch.nn as nn
import NeuralCF
import torch
class ElitismGA:
    def __init__(self, _pop_size, _r_mutation, _p_mutation,
                 _epochs, _elite_num, _mating_pool_size, _batch_size=32):
        # 输入参数
        self.pop_size = _pop_size #种群大小
        self.r_mutation = _r_mutation #子代突变
        self.p_mutation = _p_mutation  # 亲代突变
        self.epochs = _epochs#训练次数
        self.elite_num = _elite_num  # 择优数量
        self.mating_pool_size = _mating_pool_size  # for elitism
        self.batch_size = _batch_size
        # other params
        self.chroms = []
        self.evaluation_history = []
        self.stddev = 0.5
        self.criterion = nn.BCELoss()#损失函数
        self.model = None
        self.stop=False

    def initialization(self,module,userids,itemids,hidden_nums):
        for i in range(self.pop_size):
            net = module(userids,itemids,hidden_nums)
            self.chroms.append(net)
        print('network initialization({}) finished.'.format(self.pop_size))

    def train(self,trainloader):
        print('Elitism GA is training...')
        with torch.no_grad():
                for step, (batch_x, batch_y,gender,occupation,type) in enumerate(trainloader):
                    evaluation_result = self.evaluation(batch_x, batch_y,gender,occupation,type, False)
                    self.selection(evaluation_result)

    def test(self,testloader):
        print('------ Test Start -----')
        correct = 0
        total = 0
        with torch.no_grad():
            for (x,y,gender,occupation,type) in testloader:
                # images, labels = test_x.cuda(), test_y.cuda()
                images, labels = x, y
                output = self.model(images,gender,occupation,type)
                output = torch.where(output > 0.5, 1.0, 0.0)
                total = y.size(0)
                correct = (output.view(-1).data == y.data).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy of the model is: %.4f %%' % accuracy)
        return accuracy

    def selection(self, evaluation_result):
        sorted_evaluation = sorted(evaluation_result, key=lambda x: x['train_error'],reverse=True)
        elites = [e['pop'] for e in sorted_evaluation[-self.elite_num:]]
        print('Elites: {}'.format(elites))
        children = [self.chroms[i] for i in elites]
        mating_pool = [e for e in sorted_evaluation[-self.elite_num:]]
        pairs = []
        while len(children) < self.pop_size:
            pair = [self.roulette_wheel_selection(mating_pool)for _ in range(2)]
            while(pair[0]==pair[1]):
                pair = [self.roulette_wheel_selection(mating_pool) for _ in range(2)]
            pairs.append(pair)
            c1,c2=self.crossover(pair)
            children.append(c1)
            children.append(c2)
        print('Cross over finished.')

        self.replacement(children)
        for i in range(self.elite_num, self.pop_size):  # do not mutate elites
            if np.random.rand() < self.p_mutation:
                mutated_child = self.mutation(i)
                del self.chroms[i]
                self.chroms.insert(i, mutated_child)

    def crossover(self, _selected_pop):
        if _selected_pop[0] == _selected_pop[1]:
            return copy.deepcopy(self.chroms[_selected_pop[0]])

        chrom1 = copy.deepcopy(self.chroms[_selected_pop[0]])
        chrom2 = copy.deepcopy(self.chroms[_selected_pop[1]])

        chrom1_layers = list(chrom1.modules())
        chrom2_layers = list(chrom2.modules())
        child = torch.nn.Sequential()
        child2= torch.nn.Sequential()
        fc = 4
        for i in range(len(chrom1_layers)):
            layer1 = chrom1_layers[i]
            layer2 = chrom2_layers[i]
            if isinstance(layer1, nn.Linear):
                b1 = layer1.weight.data.view(-1).numpy()
                b2 = layer2.weight.data.view(-1).numpy()
                p, q = np.random.randint(0, b1.shape[0], size=2)
                if p > q: p, q = q, p
                b1[p:q], b2[p:q] = b2[p:q].copy(), b1[p:q].copy();
                b1 = np.reshape(b1, layer1.weight.data.shape)
                b2 = np.reshape(b2, layer2.weight.data.shape)
                layer1.weight.data = torch.tensor(b1)
                layer2.weight.data = torch.tensor(b2)
                child.add_module(str(i - 2), layer1)
                child2.add_module(str(i-2),layer2)
            elif isinstance(layer1, nn.Embedding):
                b1 = layer1.weight.data.view(-1).numpy()
                b2 = layer2.weight.data.view(-1).numpy()
                p, q = np.random.randint(0, b1.shape[0], size=2)
                if p > q: p, q = q, p
                b1[p:q], b2[p:q] = b2[p:q].copy(), b1[p:q].copy();
                b1 = np.reshape(b1, layer1.weight.data.shape)
                b2 = np.reshape(b2, layer2.weight.data.shape)
                layer1.weight.data = torch.tensor(b1)
                layer2.weight.data = torch.tensor(b2)
                if (fc % 4 == 0):
                    useridemb = layer1
                    useridemb2 = layer2
                elif (fc % 4 == 1):
                    itemidemb = layer1
                    itemidemb2 = layer2
                elif (fc % 4 == 3):
                    gender = layer1
                    gender2 = layer2
                elif (fc % 4 == 2):
                    occupation = layer1
                    occupation2 = layer2
                fc += 1
            elif isinstance(layer1, (torch.nn.Sequential,NeuralCF.NeuralCFModule)):
                pass
            else:
                child.add_module(str(i - 2), layer1)
                child2.add_module(str(i-2),layer2)
        chrom1.set_layer(child,useridemb,itemidemb,gender,occupation)
        chrom2.set_layer(child2,useridemb2,itemidemb2,gender2,occupation2)
        return chrom1,chrom2

    def mutation(self, _selected_pop):
        child = torch.nn.Sequential()
        chrom = copy.deepcopy(self.chroms[_selected_pop])
        chrom_layers = list(chrom.modules())

        fc = 4

        # 变异比例，选择几层进行变异
        for i, layer in enumerate(chrom_layers):
            if isinstance(layer, nn.Linear):
                if np.random.rand() < self.r_mutation:
                    weights = layer.weight.detach().numpy()
                    w = weights.astype(np.float32) + np.random.normal(0, self.stddev, weights.shape).astype(np.float32)

                    layer.weight = torch.nn.Parameter(torch.from_numpy(w))
                child.add_module(str(i - 2), layer)
            elif isinstance(layer, nn.Embedding):
                if (fc % 4 == 0):
                    if np.random.rand() < self.r_mutation:
                        weights = layer.weight.detach().numpy()
                        w = weights.astype(np.float32) + np.random.normal(0, self.stddev, weights.shape).astype(
                            np.float32)

                        layer.weight = torch.nn.Parameter(torch.from_numpy(w))
                    useridemb = layer
                elif(fc%4 ==1):
                    if np.random.rand() < self.r_mutation:
                        weights = layer.weight.detach().numpy()
                        w = weights.astype(np.float32) + np.random.normal(0, self.stddev, weights.shape).astype(
                            np.float32)

                        layer.weight = torch.nn.Parameter(torch.from_numpy(w))
                    itemidemb = layer
                elif(fc%4==3):
                    if np.random.rand() < self.r_mutation:
                        weights = layer.weight.detach().numpy()
                        w = weights.astype(np.float32) + np.random.normal(0, self.stddev, weights.shape).astype(
                            np.float32)

                        layer.weight = torch.nn.Parameter(torch.from_numpy(w))
                    gender=layer
                elif(fc%4==2):
                    if np.random.rand() < self.r_mutation:
                        weights = layer.weight.detach().numpy()
                        w = weights.astype(np.float32) + np.random.normal(0, self.stddev, weights.shape).astype(
                            np.float32)

                        layer.weight = torch.nn.Parameter(torch.from_numpy(w))
                    occupation=layer
                fc+=1
            elif isinstance(layer, (torch.nn.Sequential,NeuralCF.NeuralCFModule)):
                pass
            else:
                child.add_module(str(i - 2), layer)

        chrom.set_layer(child, useridemb,itemidemb,gender,occupation)
        return chrom

    def replacement(self, _child):
        self.chroms[:] = _child

    def evaluation(self, batch_x, batch_y,gender,occupation,type,_is_batch=True):
        cur_evaluation = []
        for i in range(self.pop_size):
            model = self.chroms[i]
            output = model(batch_x,gender,occupation,type)

            train_loss = self.criterion(output.view(-1), torch.FloatTensor(batch_y.numpy())).item()

            output = torch.where(output > 0.5, 1.0, 0.0)
            total = batch_y.size(0)
            correct = (output.view(-1).data== batch_y.data).sum().item()
            train_acc = 100 * correct / total

            train_error=torch.norm(output.view(-1)-batch_y,p=2)

            cur_evaluation.append({
                'pop': i,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 4),
                'train_error':round(train_error.item(),4)
            })
        best_fit = sorted(cur_evaluation, key=lambda x: x['train_acc'])[-1]
        self.evaluation_history.append({
            'iter': len(self.evaluation_history) + 1,
            'best_fit': best_fit,
            'avg_fitness': np.mean([e['train_acc'] for e in cur_evaluation]).round(4),
            'avg_error': np.mean([e['train_error'] for e in cur_evaluation]).round(4),
            'evaluation': cur_evaluation,
        })
        print('\nIter: {}'.format(self.evaluation_history[-1]['iter']))
        print('Best_fit: {}, avg_fitness: {:.4f},avg_error:{:.4f}'.format(self.evaluation_history[-1]['best_fit'],
                                                         self.evaluation_history[-1]['avg_fitness'],
                                                         self.evaluation_history[-1]['avg_error']))
        self.model = self.chroms[best_fit['pop']]
        return cur_evaluation

    def roulette_wheel_selection(self, evaluation_result):
        sorted_evaluation = sorted(evaluation_result, key=lambda x: x['train_error'],reverse=True)
        cum_acc = np.array([e['train_acc'] for e in sorted_evaluation]).cumsum()
        extra_evaluation = [{'pop': e['pop'], 'train_acc': e['train_acc'], 'cum_acc': acc}
                            for e, acc in zip(sorted_evaluation, cum_acc)]
        extra_evaluation=sorted(extra_evaluation, key=lambda x:x['cum_acc'])
        rand = np.random.rand() * cum_acc[-1]
        for e in extra_evaluation:
            if rand < e['cum_acc']:
                return e['pop']
        return extra_evaluation[-1]['pop']


# if __name__ == '__main__':
#     g = ElitismGA(
#         _pop_size=100,
#         _p_mutation=0.1,
#         _r_mutation=0.1,
#         _epochs=20,
#         _elite_num=20,
#         _mating_pool_size=40,
#         _batch_size=32
#     )
#
#     g.train()
#
#     g.test()