import time

import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from src.evaluate import get_all_metrics
from src.load_base import get_records, load_kg


class KARN(nn.Module):

    def __init__(self, args, n_entity, n_relation):
        super(KARN, self).__init__()
        self.dim = args.dim
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_path = args.n_path
        self.n_neighbor = args.n_neighbor
        self.path_len = args.path_len
        self.n_records = args.n_record

        entity_emb_mat = t.randn(n_entity, self.dim)
        nn.init.xavier_uniform_(entity_emb_mat)
        self.entity_emb_mat = nn.Parameter(entity_emb_mat)
        relation_emb_mat = t.randn(n_relation, self.dim)
        nn.init.xavier_uniform_(relation_emb_mat)
        self.relation_emb_mat = nn.Parameter(relation_emb_mat)

        self.conv1 = nn.Conv1d(args.n_neighbor, args.n_neighbor, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(args.n_neighbor, args.n_neighbor, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(args.n_neighbor, args.n_neighbor, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(args.n_neighbor, args.n_neighbor, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(args.n_neighbor, args.n_neighbor, kernel_size=5, padding=2)
        self.local_pool = nn.MaxPool2d(kernel_size=[args.n_neighbor, self.dim])

        self.user_lstm = nn.LSTM(self.dim, self.dim, batch_first=True)
        self.path_lstm = nn.LSTM(self.dim, self.dim, batch_first=True)

        self.weight_1 = nn.Linear(self.dim, self.dim)
        self.weight_2 = nn.Linear(self.dim, self.dim)

        self.weight_path_1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.weight_path_2 = nn.Linear(2 * self.dim, 2 * self.dim)

        self.weight_attention = nn.Linear(self.dim, 1)

        self.weight_path_attention = nn.Linear(4*self.dim, 1)

        self.weight_user_1 = nn.Linear(6 * self.dim, self.dim)
        self.weight_user_2 = nn.Linear(2 * self.dim, 1)

        self.g_1 = nn.Linear(5, self.dim)
        self.g_2 = nn.Linear(3 * self.dim, self.dim)

    def forward(self, history_items, items, entities_list, relations_list, kg_dict):

        user_embeddings = self.get_user_embeddings(history_items, kg_dict, entities_list, relations_list)
        item_embeddings = self.get_item_embeddings(items, kg_dict)
        # print(user_embeddings.shape, item_embeddings.shape)
        predict = t.sigmoid(self.weight_user_2(t.cat([user_embeddings, item_embeddings], dim=1)))

        return predict.reshape(-1)

    def get_item_embeddings(self, items, kg_dict):

        neighbors = []
        titles = []
        for i in range(len(items)):

            item_neighbors = kg_dict[items[i]]

            if len(item_neighbors) >= self.n_neighbor:
                indices = np.random.choice(len(item_neighbors), self.n_neighbor, replace=False)
            else:
                indices = np.random.choice(len(item_neighbors), self.n_neighbor, replace=True)

            neighbors.append([item_neighbors[k][1] for k in indices])
            titles.append([items[i]])

        item_representation = self.item_entity_representation(items, neighbors, titles)
        return item_representation

    def get_user_embeddings(self, history_items, kg_dict, entities_list, relations_list):

        items = []
        neighbors = []
        titles = []
        for i in range(len(history_items)):
            items.extend(history_items[i])
            for j in range(len(history_items[i])):
                item_neighbors = kg_dict[history_items[i][j]]

                if len(item_neighbors) >= self.n_neighbor:
                    indices = np.random.choice(len(item_neighbors), self.n_neighbor, replace=False)
                else:
                    indices = np.random.choice(len(item_neighbors), self.n_neighbor, replace=True)

                neighbors.append([item_neighbors[k][1] for k in indices])
                titles.append([history_items[i][j]])

        item_representation = self.item_entity_representation(items, neighbors, titles).reshape(-1, self.n_records, self.dim)  # (batch_size * n_neighbor, dim)
        s = self.user_history_interest_extraction(item_representation)
        s_ = self.user_potential_intent_extraction(entities_list, relations_list)
        # print(s.shape, s_.shape, item_representation.shape, len(history_items), len(items))
        return t.sigmoid(self.weight_user_1(t.cat([s, s_], dim=1)))

    def user_history_interest_extraction(self, history_item_embeddings):

        output = self.user_lstm(history_item_embeddings)[0]  # (batch_size, len, dim)

        return self.sra_layer(output)

    def sra_layer(self, output):

        A = self.weight_2(t.sigmoid(self.weight_1(output)))  # (batch_size, len, dim)

        a = t.matmul(output.reshape(A.shape[0], self.dim, -1), A).mean(dim=1)  # (batch_size, dim)

        s = t.cat([a, output[:, -1, :]], dim=1)
        # print(s.shape, '....')
        return s

    def path_sra_layer(self, output):
        A = self.weight_path_2(t.sigmoid(self.weight_path_1(output)))  # (batch_size, len, 2*dim)

        a = t.matmul(output.reshape(A.shape[0], 2*self.dim, -1), A).mean(dim=1)  # (batch_size, dim)

        s = t.cat([a, output[:, -1, :]], dim=1)
        # print(s.shape, '....')
        return s

    def user_potential_intent_extraction(self, entities_list, relations_list):

        entity_embedding_list = []
        relation_embedding_list = []
        zeros = t.zeros(1, self.path_len+1, self.dim)
        if t.cuda.is_available():
            zeros = zeros.to(self.entity_emb_mat[0].device)
        for i in range(len(entities_list)):
            for j in range(self.n_path):
                if entities_list[i] == []:
                    entity_embedding_list.append(zeros)
                    relation_embedding_list.append(zeros)
                else:
                    entity_embedding_list.append(self.entity_emb_mat[entities_list[i][j]].reshape(1, self.path_len+1, self.dim))
                    relation_embedding_list.append(self.relation_emb_mat[relations_list[i][j]].reshape(1, self.path_len+1, self.dim))

        entity_embeddings = t.cat(entity_embedding_list, dim=0)  # (batch_size * n_path, path_len, dim)
        relation_embeddings = t.cat(relation_embedding_list, dim=0)
        output = t.cat([entity_embeddings, relation_embeddings], dim=-1)
        # print(output.shape, ',,,,,,,,')
        path_embeddings = self.path_sra_layer(output).reshape(-1, self.n_path, 4*self.dim)
        # print(path_embeddings.shape, len(entities_list), output.shape, entity_embeddings.shape, '...')
        attention_weight = t.sigmoid(self.weight_path_attention(path_embeddings))
        user_potential_intent_embeddings = (attention_weight * path_embeddings).sum(dim=1)

        return user_potential_intent_embeddings

    def item_entity_representation(self, items, neighbors, titles):

        textual_representation = self.get_textual_representation(neighbors)  # （batch_size, dim）
        contextual_representation = self.get_contextual_reperesentation(titles)  # (batch_size, dim)
        item_embeddings = self.entity_emb_mat[items]   # (batch_size, dim)
        result = self.g_2(t.cat([item_embeddings, textual_representation, contextual_representation], dim=1))
        # print(textual_representation.shape, contextual_representation.shape, len(items))
        return t.sigmoid(result)  # (batch_size, dim)

    def get_contextual_reperesentation(self, tiles):
        title_embedding_list = [self.entity_emb_mat[tiles[i]].reshape(1, -1, self.dim) for i in range(len(tiles))]
        title_embeddings = t.cat(title_embedding_list, dim=0)

        return title_embeddings.mean(dim=1)

    def get_textual_representation(self, neighbors):

        neighbor_embedding_list = [self.entity_emb_mat[neighbors[i]].reshape(1, -1, self.dim) for i in range(len(neighbors))]

        neighbor_embeddings = t.cat(neighbor_embedding_list, dim=0)

        x1 = t.relu(self.conv1(neighbor_embeddings))  # (batch_size, n_neighbor, dim)
        p1 = self.local_pool(x1).reshape(-1, 1)  # (batch_size, 1)

        x2 = t.relu(self.conv1(neighbor_embeddings))  # (batch_size, n_neighbor, dim)
        p2 = self.local_pool(x2).reshape(-1, 1)  # (batch_size, 1)

        x3 = t.relu(self.conv1(neighbor_embeddings))  # (batch_size, n_neighbor, dim)
        p3 = self.local_pool(x3).reshape(-1, 1)  # (batch_size, 1)

        x4 = t.relu(self.conv1(neighbor_embeddings))  # (batch_size, n_neighbor, dim)
        p4 = self.local_pool(x4).reshape(-1, 1)  # (batch_size, 1)

        x5 = t.relu(self.conv1(neighbor_embeddings))  # (batch_size, n_neighbor, dim)
        p5 = self.local_pool(x5).reshape(-1, 1)  # (batch_size, 1)
        result = self.g_1(t.cat([p1, p2, p3, p4, p5], dim=1))
        return t.sigmoid(result)  # (batch_size, dim)


def get_scores(model, rec, paths_dict, args, user_records, relation_dict, n_relation, kg_dict):
    rec_item_dict = {}
    model.eval()
    for user in rec:

        pairs = [(user, item) for item in rec[user]]
        input_data = get_input_data(args, pairs, paths_dict, user_records, relation_dict, n_relation, kg_dict)
        predict = model(input_data[0], input_data[1], input_data[2], input_data[3], input_data[4])
        predict_np = predict.detach().cpu().numpy()
        item_scores = {rec[user][i]: predict_np[i] for i in range(len(pairs))}

        sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

        rec_item_dict[user] = [i[0] for i in sorted_item_scores]
    model.train()
    return rec_item_dict


def get_input_data(args, pairs, paths_dict, user_records, relation_dict, n_relation, kg_dict):
    users = [pair[0] for pair in pairs]
    history_items = get_history_items(users, user_records, args.n_record)
    entities_list, relations_list = get_paths(pairs, paths_dict, args.n_path, relation_dict, n_relation)
    items = [pair[1] for pair in pairs]
    return history_items, items, entities_list, relations_list, kg_dict


def get_history_items(users, user_records, n_record):

    history_items = []

    for user in users:
        records = user_records[user]

        if len(records) >= n_record:
            indices = np.random.choice(len(records), n_record, replace=False)
        else:
            indices = np.random.choice(len(records), n_record, replace=True)

        history_items.append([records[i] for i in indices])

    return history_items


def get_paths(pairs, paths_dict, n_path, relation_dict, n_relation):
    entities_list = []
    relations_list = []
    for pair in pairs:
        user = pair[0]
        item = pair[1]

        if len(paths_dict[(user, item)]) == 0:
            entities_list.append([])
            relations_list.append([])
        else:
            if len(paths_dict[(user, item)]) >= n_path:
                indices = np.random.choice(len(paths_dict[(user, item)]), n_path, replace=False)
            else:
                indices = np.random.choice(len(paths_dict[(user, item)]), n_path, replace=True)

            paths = [paths_dict[(user, item)][i] for i in indices]
            entities_list.append(paths)
            relations_list.append(get_relations(relation_dict, paths, n_relation))

    return entities_list, relations_list


def get_relations(relation_dict, paths, n_relation):
    relations = []
    for path in paths:
        relation_list = []
        for i in range(len(path)):
            if i == len(path) - 1:
                relation_list.append(n_relation)
            else:
                relation_list.append(relation_dict[(path[i], path[i+1])])

        relations.append(relation_list)

    return relations


def eval_ctr(model, pairs, paths_dict, args, user_records, relation_dict, n_relation, kg_dict):

    model.eval()
    pred_label = []
    input_data = get_input_data(args, pairs, paths_dict, user_records, relation_dict, n_relation, kg_dict)
    for i in range(0, len(pairs), args.batch_size):
        predicts = model(input_data[0][i: i + args.batch_size],
                         input_data[1][i: i + args.batch_size],
                         input_data[2][i: i + args.batch_size],
                         input_data[3][i: i + args.batch_size],
                         input_data[4])
        pred_label.extend(predicts.cpu().detach().numpy().tolist())
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return auc, acc


def train(args, is_topk=False):
    np.random.seed(123)

    data_dir = './data/' + args.dataset + '/'
    train_set = np.load(data_dir + str(args.ratio) + '_train_set.npy').tolist()
    eval_set = np.load(data_dir + str(args.ratio) + '_eval_set.npy').tolist()
    test_set = np.load(data_dir + str(args.ratio) + '_test_set.npy').tolist()
    test_records = get_records(test_set)
    entity_list = np.load(data_dir + '_entity_list.npy').tolist()
    relation_dict = np.load(data_dir + str(args.ratio) + '_relation_dict.npy', allow_pickle=True).item()
    n_entity = len(entity_list)
    paths_dict = np.load(data_dir + str(args.ratio) + '_3_path_dict.npy', allow_pickle=True).item()
    kg_dict, _, n_relation = load_kg(data_dir)
    user_records = get_records(train_set)

    model = KARN(args, len(entity_list), n_relation+3)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []
    for epoch in range(args.epochs):
        loss_sum = 0
        start = time.clock()
        np.random.shuffle(train_set)
        input_data = get_input_data(args, train_set, paths_dict, user_records, relation_dict, n_relation, kg_dict)
        labels = t.tensor([i[2] for i in train_set]).float()
        if t.cuda.is_available():
            labels = labels.to(args.device)
        start_index = 0
        size = len(train_set)
        model.train()
        while start_index < size:
            predicts = model(input_data[0][start_index: start_index+args.batch_size],
                             input_data[1][start_index: start_index+args.batch_size],
                             input_data[2][start_index: start_index+args.batch_size],
                             input_data[3][start_index: start_index+args.batch_size],
                             input_data[4])
            loss = criterion(predicts, labels[start_index: start_index + args.batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        train_auc, train_acc = eval_ctr(model, train_set, paths_dict, args, user_records, relation_dict, n_relation, kg_dict)
        eval_auc, eval_acc = eval_ctr(model, eval_set, paths_dict, args, user_records, relation_dict, n_relation, kg_dict)
        test_auc, test_acc = eval_ctr(model, test_set, paths_dict, args, user_records, relation_dict, n_relation, kg_dict)

        print('epoch: %d \t train_auc: %.4f \t train_acc: %.4f \t '
              'eval_auc: %.4f \t eval_acc: %.4f \t test_auc: %.4f \t test_acc: %.4f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            pass

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.4f \t train_acc: %.4f \t eval_auc: %.4f \t eval_acc: %.4f \t '
          'test_auc: %.4f \t test_acc: %.4f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]