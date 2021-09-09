from functools import partial

import graphviz
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import queue
import utils

class InnerNode():

    def __init__(self, depth, args):
        self.args = args
        self.fc = nn.Linear(self.args.input_dim, 1)
        beta = torch.randn(1)
        # beta = beta.expand((self.args.batch_size, 1))
        if self.args.cuda:
            beta = beta.cuda()
        self.beta = nn.Parameter(beta)
        self.leaf = False
        self.prob = None
        self.leaf_accumulator = []
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        self.build_child(depth)
        self.penalties = []
        self.index = None

    def reset(self):
        self.leaf_accumulator = []
        self.penalties = []
        self.left.reset()
        self.right.reset()

    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = InnerNode(depth + 1, self.args)
            self.right = InnerNode(depth + 1, self.args)
        else:
            self.left = LeafNode(self.args)
            self.right = LeafNode(self.args)

    def forward(self, x):
        return (torch.sigmoid(self.beta * self.fc(x)))

    def select_next(self, x):
        prob = self.forward(x)
        if prob < 0.5:
            return (self.left, prob)
        else:
            return (self.right, prob)

    def cal_prob(self, x, path_prob):
        self.prob = self.forward(x)  # probability of selecting right node
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1 - self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return (self.leaf_accumulator)

    def get_penalty(self):
        eps = np.finfo(np.float32).eps
        min_tensor = torch.tensor(eps)
        max_tensor = torch.sub(1., eps)
        penalty = (
            torch.min(
                torch.max(
                    torch.sum(self.prob * self.path_prob) / torch.max(torch.sum(self.path_prob), min_tensor)
                    , min_tensor), max_tensor), self.lmbda)

        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return (self.penalties)


class LeafNode():
    def __init__(self, args):
        self.args = args
        self.param = torch.randn(self.args.output_dim)
        if self.args.cuda:
            self.param = self.param.cuda()
        self.param = nn.Parameter(self.param)
        self.leaf = True
        self.softmax = nn.Softmax()
        self.index = None

    def forward(self):
        return (self.softmax(self.param.view(1, -1)))

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        Q = self.forward()
        # Q = Q.expand((self.args.batch_size, self.args.output_dim))
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))
        return ([[path_prob, Q]])


class SoftDecisionTree(nn.Module):

    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.root = InnerNode(1, self.args)
        self.collect_parameters()  ##collect parameters and modules under root node
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.test_acc = []
        self.define_extras(self.args.batch_size)
        self.best_accuracy = 0.0

    def define_extras(self, batch_size):
        ##define target_onehot and path_prob_init batch size, because these need to be defined according to batch size, which can be differ
        self.target_onehot = torch.FloatTensor(batch_size, self.args.output_dim)
        self.target_onehot = Variable(self.target_onehot)
        self.path_prob_init = Variable(torch.ones(batch_size, 1))
        if self.args.cuda:
            self.target_onehot = self.target_onehot.cuda()
            self.path_prob_init = self.path_prob_init.cuda()

    '''
    def forward(self, x):
        node = self.root
        path_prob = Variable(torch.ones(self.args.batch_size, 1))
        while not node.leaf:
            node, prob = node.select_next(x)
            path_prob *= prob
        return node()
    '''

    def cal_loss(self, x, y):
        batch_size = y.size()[0]
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [torch.zeros(self.args.output_dim) for _ in range(batch_size)]
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(y.view(batch_size, 1, self.args.output_dim),
                           torch.log(Q).view(batch_size, self.args.output_dim, 1)).view(-1, 1)
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()

        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 * (torch.log(penalty) + torch.log(1 - penalty))
        output = torch.stack(max_Q)

        self.root.reset()  ##reset all stacked calculation
        return (-loss + C,
                output)  ## -log(loss) will always output non, because loss is always below zero. I suspect this is the mistake of the paper?

    def collect_parameters(self):
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                param = node.param
                self.param_list.append(param)
            else:
                fc = node.fc
                beta = node.beta
                nodes.append(node.right)
                nodes.append(node.left)
                self.param_list.append(beta)
                self.module_list.append(fc)

    def train_(self, train_loader, epoch, crop):
        self.train()
        self.define_extras(self.args.batch_size)
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            data = torch.cat(list(map(partial(utils.img_to_tensor, crop=crop), data)))

            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            # data = data.view(self.args.batch_size,-1)
            target = Variable(target)
            target_ = target.view(-1, 1)
            batch_size = target_.size()[0]
            data = data.view(batch_size, -1)
            ##convert int target to one-hot vector
            data = Variable(data)

            if not batch_size == self.args.batch_size:  # because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()
            self.target_onehot.scatter_(1, target_, 1.)
            self.optimizer.zero_grad()

            loss, output = self.cal_loss(data, self.target_onehot)
            # loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct / len(data)

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item(),
                    correct, len(data),
                    accuracy))

    def test_(self, test_loader, crop):
        self.eval()
        self.define_extras(self.args.batch_size)
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = torch.cat(list(map(partial(utils.img_to_tensor, crop=crop), data)))

            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target = Variable(target)
            target_ = target.view(-1, 1)
            batch_size = target_.size()[0]
            data = data.view(batch_size, -1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size:  # because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()
            self.target_onehot.scatter_(1, target_, 1.)
            _, output = self.cal_loss(data, self.target_onehot)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy))

    def buildTree(self):
        Tree = graphviz.Digraph(format='png', graph_attr={"randir": "LR"},
                                node_attr={'shape': "box"})
        all_nodes = self.gather_nodes()
        for node in all_nodes:
            if not node.leaf:
                utils.weights_viz(node.fc.weight, node.index)
                Tree.node(str(node.index), image="weights_img/weights{}.png".format(node.index),
                          label="", style="rounded,filled")
            else:
                Tree.node(str(node.index),
                          label=str(node.forward().detach().numpy()[0]),
                          fillcolor="green", style="rounded,filled")

        for node in all_nodes:
            if not node.leaf:
                Tree.edge(str(node.index), str(node.right.index))
                Tree.edge(str(node.index), str(node.left.index))
        return Tree

    def gather_nodes(self):
        nodes_queue = queue.Queue()
        nodes_queue.put(self.root)
        all_nodes = []
        node_index = 0
        while not nodes_queue.empty():
            curr_node = nodes_queue.get()
            curr_node.index = node_index
            all_nodes.append(curr_node)
            if not curr_node.leaf:
                nodes_queue.put(curr_node.left)
                nodes_queue.put(curr_node.right)
            node_index += 1
        return all_nodes
