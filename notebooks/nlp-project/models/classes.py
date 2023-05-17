import codecs
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import gensim.models
GoogleEmbs = gensim.models.KeyedVectors.load_word2vec_format(
                                '/work/nlp-project/models/GoogleNews-50k.bin', binary=True)

@dataclass
class Batch:
    inputs: Tensor
    targets: Tensor

class DataIterator:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        
    def __call__(self, inputs: Tensor, targets: Tensor) -> Batch:
        intervals = np.arange(0, len(inputs), self.batch_size)
        for start in intervals:
            end = start + self.batch_size
            batch_inputs = inputs[start: end]
            batch_targets = targets[start: end]
            
            yield Batch(batch_inputs, batch_targets)


class F1_evaluator():
    """
    Calculates f1 score on epoch level by aggregating metrics for each sentence.

    Pass true and predicted labels to pass_batch() each batch, then use f1_score() for epoch level entity F1 (resets metrics).

    Note: Assumes consecutive entity labels are multi-word entities.
    Source for metrics: https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    """
    
    def __init__(self, pad_label=2):
        self.pad_label = pad_label
        self.metrics = np.array([0,0,0,0,0,0,0])  # initializing epoch-level [COR, PAR, INC, MIS, SPU, ACT, POS] array to 0
        
    def _pre_process(self, targets, pred_tags):
        """turns torch arrays into lists of ints"""
        targets = targets.detach().numpy()
        # print(targets, pred_tags)
        pad_mask = targets != self.pad_label
        targets = list(map(int,targets[pad_mask]))
        pred_tags = torch.argmax(pred_tags, dim=0, keepdim=False)
        pred_tags = pred_tags.detach().numpy()
        pred_tags = pred_tags[pad_mask]
        # print(targets, pred_tags)
        return targets, pred_tags

    def _entities(self, labels):
        """finds entity indeces and groups them in list of lists"""
        entities = []
        current_ent = []
        for i, lab in enumerate(labels):
            if lab != 1 and len(current_ent) > 0:  # end of current entity, so append and reset
                entities.append(current_ent)
                current_ent = []
            elif lab == 1:  # entity label found, so append to curent entity
                current_ent.append(i)
                if i == len(labels)-1:  # if last label, append current entity
                    entities.append(current_ent)
        # print(entities)            
        return entities

    def _get_sentence_metrics(self, true_labels, pred_labels):
        true_labels, pred_labels = self._pre_process(true_labels, pred_labels)  # Turn torch tensors into lists of ints

        true_ents = self._entities(true_labels)  # Get entity indeces
        pred_ents = self._entities(pred_labels)
        
        POS = len(true_ents)  # number of true entities 
        ACT = len(pred_ents)  # number of predicted entities
        COR = len([e for e in true_ents if e in pred_ents])  # number of correctly predicted entities
        if POS == 0:  # no true entities
            SPU = ACT
            INC = SPU
            PAR = MIS = POS = 0
            return np.array([COR, PAR, INC, MIS, SPU, ACT, POS])  
        
        # Comupting number of partial matches (PAR) and missed entities (MIS)
        PAR = MIS = 0
        pred_entity_indeces = [x for e in pred_ents for x in e]
        non_predicted_entities = [e for e in true_ents if e not in pred_ents]
        for ent in non_predicted_entities:
            for index in ent:
                if index in pred_entity_indeces:
                    # part of the entity was predicted
                    PAR += 1
                    break
            else:
                # none of the entity was predicted
                MIS += 1
        SPU = ACT - COR - PAR  # SPU = predicted entities which are (wholly) nonexistant in true entities
        INC = MIS + SPU  # INC = Incorrectly predicted entities (missing or spurious)
        return np.array([COR, PAR, INC, MIS, SPU, ACT, POS])  #  return metrics of sentence

    def pass_batch(self, true_labels_batch, pred_labels_batch):
        # aggregating metrics from batch and updating metrics
        for true_labels, pred_labels in zip(true_labels_batch, pred_labels_batch):
            self.metrics += self._get_sentence_metrics(true_labels, pred_labels)

    def f1_score(self, relaxed=False, verbose=False):
        """
        relaxed: whether relaxed-match evaluation is used instead of strict (default)
        verbose: whether to return (precision, recall, f1)
        """
        COR, PAR, INC, MIS, SPU, ACT, POS = list(self.metrics)

        if POS == 0: recall = 0
        else: recall = COR/POS
        
        if ACT == 0: precision = 0
        elif relaxed: precision = (COR+.5*PAR)/ACT  # Relaxed F1 precision
        else: precision = COR/ACT  # Strict F1 precision
        
        if precision==0 and recall==0: f1 = 0
        else: f1 = 2*precision*recall / (precision+recall)

        self.metrics = np.array([0,0,0,0,0,0,0])  # epoch level metrics are reset

        if verbose: 
            return precision, recall, f1
        return f1

class WeightedCrossEntropy(nn.Module):
    def __init__(self, epochs, pad_label):
        super(WeightedCrossEntropy, self).__init__()
        self.epochs = epochs
        self.pad_label = pad_label

    def forward(self, labeled_output, pseudo_output, labeled_target, pseudo_target, epoch):
        alpha = epoch / (self.epochs)  # add 2* to weigh pseudo equally at last epoch
        L = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.pad_label)
        labeled_loss = L(labeled_output, labeled_target) / len(labeled_target)
        pseudo_loss = L(pseudo_output, pseudo_target) / len(pseudo_target)
        return (1-alpha) * labeled_loss + alpha * pseudo_loss, alpha  # added 1-alpha term to normalize

class PolyDataIterator:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
    def __call__(self, inputs: Sequence[Tensor], targets: Sequence[Tensor]) -> List[List[Tensor]]:
        self._validate(inputs, targets)
        set_sizes = [len(tar) for tar in targets]
        strata_sizes = [size / sum(set_sizes) for size in set_sizes]
        intervals = np.arange(0, sum(set_sizes), self.batch_size)
        for start in intervals:
            end = start + self.batch_size
            outputs = []
            for inp, tar, strata in zip(inputs, targets, strata_sizes):
                start_idx = round(start * strata)
                end_idx = round(end * strata)
                inp_ = inp[start_idx: end_idx]
                tar_ = tar[start_idx: end_idx]
                outputs.append([inp_, tar_])
            yield outputs

    def _validate(self, inputs, targets):
        for inp, tar in zip(inputs, targets):
            if len(inp) != len(tar):
                raise AttributeError(f"Mismatching number of samples for a dataset "
                                     f"and the corresponding targets ({len(inp)} and {len(tar)})")

class DevTrainBiLSTM(torch.nn.Module):
    def __init__(self,
                hidden_size=20,
                max_len=100,
                n_labels=3,
                batch_size=32,
                pad_token="<PAD>",
                pad_label=2,
                embedding_dim=300
                ):
        super().__init__()
        
        self.embedding_dim = embedding_dim  # length of embedding vectors
        self.hidden_size = hidden_size  # number of LSTM cells
        self.max_len=max_len  # maximum input sentence length, will be padded to this size
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.pad_token = pad_token
        self.pad_label = pad_label

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=2 * self.hidden_size, out_features=n_labels)

        self.data_iterator = PolyDataIterator(batch_size=self.batch_size)
        self.dev_iterator = DataIterator(batch_size=self.batch_size)
        self.optimizer = None  # initializes optimizer to None

        # Logs for performance in each training epoch
        self.train_f1_log = []
        self.dev_f1_log = []

    def _pad_inputs(self, collection: List[List[int]], padding_token):
        to_series = [pd.Series(el) for el in collection]
        enc_matrix = (pd.concat(to_series, axis=1)
                        .reindex(range(self.max_len))
                        .fillna(padding_token)
                        .T)
        collection = enc_matrix.values.tolist()
        return collection

    def _pad_data(self, documents, labels):
        padded_documents = self._pad_inputs(documents, self.pad_token)
        padded_labels = self._pad_inputs(labels, self.pad_label)
        padded_labels = [list(map(int,sentence)) for sentence in padded_labels]
        return padded_documents, padded_labels

    def forward(self, inputs):
        '''
        Implements a forward pass through the BiLSTM.
        inputs are a batch (list) of sentences.
        '''
        word_embeds = self._get_google_embeds(inputs)
        lstm_result, _ = self.lstm(word_embeds)
        tags = self.linear(lstm_result)
        log_probs = torch.nn.functional.softmax(tags, dim=2)
        return log_probs
    
    def _get_google_embeds(self, inputs):
        embeddings = torch.Tensor()
        for sentence in inputs:
            sentence_embeds = torch.Tensor()
            for word in sentence:
                if GoogleEmbs.__contains__(word):
                    embed = GoogleEmbs.get_vector(word)
                    embed.setflags(write = True)
                    embed = torch.from_numpy(embed)
                else:
                    embed = torch.zeros(300)  # the word is not in the model dictionary, so use zero vector
                sentence_embeds = torch.cat((sentence_embeds, embed), dim=0)
            embeddings = torch.cat((embeddings, sentence_embeds), dim=0)
        return embeddings.view(len(inputs), -1, self.embedding_dim)
    
    def fit(self, 
            documents, 
            labels, 
            dev=None,
            pseudo=None, 
            epochs=3, 
            print_metrics=False, 
            learning_rate=0.05,
            optimizer=None):
        
        padded_documents, padded_labels = self._pad_data(documents, labels)  # Padding training data

        pseudo_docs, pseudo_labels = pseudo
        pseudo_docs, pseudo_labels = self._pad_data(pseudo_docs, pseudo_labels)  # Padding pseudo data
        loss_func = WeightedCrossEntropy(epochs, pad_label=self.pad_label)
        Evaluator = F1_evaluator(self.pad_label)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            self.train()
            for labeled, paraphrased in self.data_iterator([padded_documents, pseudo_docs], [padded_labels, pseudo_labels]):
                pred_tags = self.forward(inputs=labeled[0])
                pred_pseudo_tags = self.forward(inputs=paraphrased[0])
                
                # probability distribution for each tag across all words
                pred_tags = pred_tags.view(-1, self.n_labels)
                pred_pseudo_tags = pred_pseudo_tags.view(-1, self.n_labels)
                
                # true label for each word
                targets = torch.tensor(labeled[1]).flatten()
                pseudo_targets = torch.tensor(paraphrased[1]).flatten()

                # passing all batch labels to evaluator
                Evaluator.pass_batch(torch.cat((targets, pseudo_targets), 0), 
                                     torch.cat((pred_tags, pred_pseudo_tags), 0))

                batch_loss, alpha = loss_func(pred_tags, pred_pseudo_tags, targets, pseudo_targets, epoch)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            train_metrics = Evaluator.metrics
            train_f1 = Evaluator.f1_score()
            self.train_f1_log.append(train_f1)

            if dev is None:  # print performance and go to next epoch if no dev data is supplied
                if print_metrics:
                    COR, PAR, INC, MIS, SPU, ACT, POS = list(train_metrics)
                    print(f"Epoch {epoch}, train: {train_f1:.3f} \n train metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
                else:
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, loss: {epoch_loss:.3f}")
            else:
                # Dev evaluation
                x_dev, y_dev = dev
                dev_f1, dev_metrics = self.evaluate(x_dev, y_dev, print_metrics=False, return_metrics=True)
                self.dev_f1_log.append(dev_f1)
                if print_metrics:
                    COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f} \n dev metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
                else:
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f}")

    def evaluate(self, x_dev, y_dev, print_metrics=True, return_metrics=False):
        """
        Evaluates model performance on supplied data.
        print_metrics set to print out metrics by default.
        return_metrics (optionally) returns F1 and metrics.
        """
        padded_dev_docs, padded_dev_labs = self._pad_data(x_dev, y_dev)  # Padding data
        self.eval()
        Evaluator = F1_evaluator(self.pad_label)
        for i, batch in enumerate(self.dev_iterator(padded_dev_docs, padded_dev_labs)):
            with torch.no_grad():
                pred_dev = self.forward(batch.inputs)
            pred_dev = pred_dev.view(-1, self.n_labels) 
            targets = torch.tensor(batch.targets).flatten()
            Evaluator.pass_batch(targets, pred_dev)
        dev_metrics = Evaluator.metrics
        dev_f1 = Evaluator.f1_score()
        if print_metrics:
            COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
            print(f"F1 score: {dev_f1:.3f}\nMetrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
        if return_metrics:
            return dev_f1, dev_metrics


class BaselineBiLSTM(torch.nn.Module):
    def __init__(self,
                hidden_size=20,
                max_len=100,
                n_labels=3,
                batch_size=32,
                pad_token="<PAD>",
                pad_label=2,
                embedding_dim=300
                ):
        super().__init__()
        
        self.embedding_dim = embedding_dim  # length of embedding vectors
        self.hidden_size = hidden_size  # number of LSTM cells
        self.max_len=max_len  # maximum input sentence length, will be padded to this size
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.pad_token = pad_token
        self.pad_label = pad_label

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=2 * self.hidden_size, out_features=n_labels)

        self.data_iterator = DataIterator(batch_size=self.batch_size)
        self.optimizer = None  # initializes optimizer to None

        # Logs for performance in each training epoch
        self.train_f1_log = []
        self.dev_f1_log = []

    def _pad_inputs(self, collection: List[List[int]], padding_token):
        to_series = [pd.Series(el) for el in collection]
        enc_matrix = (pd.concat(to_series, axis=1)
                        .reindex(range(self.max_len))
                        .fillna(padding_token)
                        .T)
        collection = enc_matrix.values.tolist()
        return collection

    def _pad_data(self, documents, labels):
        padded_documents = self._pad_inputs(documents, self.pad_token)
        padded_labels = self._pad_inputs(labels, self.pad_label)
        padded_labels = [list(map(int,sentence)) for sentence in padded_labels]

        return padded_documents, padded_labels

    def forward(self, inputs):
        '''
        Implements a forward pass through the BiLSTM.
        inputs are a batch (list) of sentences.
        '''
        word_embeds = self._get_google_embeds(inputs)
        lstm_result, _ = self.lstm(word_embeds)
        tags = self.linear(lstm_result)
        log_probs = torch.nn.functional.softmax(tags, dim=2)
        return log_probs
    
    def _get_google_embeds(self, inputs):
        embeddings = torch.Tensor()
        for sentence in inputs:
            sentence_embeds = torch.Tensor()
            for word in sentence:
                if GoogleEmbs.__contains__(word):
                    embed = GoogleEmbs.get_vector(word)
                    embed.setflags(write = True)
                    embed = torch.from_numpy(embed)
                else:
                    embed = torch.zeros(300)  # the word is not in the model dictionary, so use zero vector
                sentence_embeds = torch.cat((sentence_embeds, embed), dim=0)
            embeddings = torch.cat((embeddings, sentence_embeds), dim=0)
        return embeddings.view(len(inputs), -1, self.embedding_dim)
    
    def fit(self, 
            documents, 
            labels, 
            dev=None,  
            epochs=3, 
            print_metrics=False, 
            learning_rate=0.05,
            optimizer=None):
        
        padded_documents, padded_labels = self._pad_data(documents, labels)  # Padding training data

        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        loss_func = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=self.pad_label)  # ignores loss for padding label
        Evaluator = F1_evaluator(self.pad_label)
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            self.train()
            for i, batch in enumerate(self.data_iterator(padded_documents, padded_labels)):
                pred_tags = self.forward(inputs=batch.inputs)
                pred_tags = pred_tags.view(-1, self.n_labels) # probability distribution for each tag across all words in batch
                targets = torch.tensor(batch.targets).flatten()  # true label for each word
                Evaluator.pass_batch(targets, pred_tags)  # passing batch labels to evaluator
                batch_loss = loss_func(pred_tags, targets)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            train_metrics = Evaluator.metrics
            train_f1 = Evaluator.f1_score()
            self.train_f1_log.append(train_f1)

            if dev is None:  # print performance and go to next epoch if no dev data is supplied
                if print_metrics:
                    COR, PAR, INC, MIS, SPU, ACT, POS = list(train_metrics)
                    print(f"Epoch {epoch}, train: {train_f1:.3f} \n train metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
                else:
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, loss: {epoch_loss:.3f}")
            else:
                # Dev evaluation
                x_dev, y_dev = dev
                dev_f1, dev_metrics = self.evaluate(x_dev, y_dev, print_metrics=False, return_metrics=True)
                self.dev_f1_log.append(dev_f1)

                if print_metrics:
                    COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f} \n dev metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
                else:
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f}")

    def evaluate(self, x_dev, y_dev, print_metrics=True, return_metrics=False):
        """
        Evaluates model performance on supplied data.
        print_metrics set to print out metrics by default.
        return_metrics (optionally) returns F1 and metrics.
        """
        padded_dev_docs, padded_dev_labs = self._pad_data(x_dev, y_dev)  # Padding data
        self.eval()
        Evaluator = F1_evaluator(self.pad_label)
        for i, batch in enumerate(self.data_iterator(padded_dev_docs, padded_dev_labs)):
            with torch.no_grad():
                pred_dev = self.forward(batch.inputs)
            pred_dev = pred_dev.view(-1, self.n_labels) 
            targets = torch.tensor(batch.targets).flatten()
            Evaluator.pass_batch(targets, pred_dev)
        dev_metrics = Evaluator.metrics
        dev_f1 = Evaluator.f1_score()
        if print_metrics:
            COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
            print(f"F1 score: {dev_f1:.3f}\nMetrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
        if return_metrics:
            return dev_f1, dev_metrics




class SecondLSTM(torch.nn.Module):
    def __init__(self,
                embedding_type = 'bert',
                LSTM_HIDDEN=20,
                max_len=100,
                n_labels=3,
                batch_size=32,
                padding=("<PAD>",-100)
                ):
        super().__init__()
        
        self.embedding_type = embedding_type  # 'bert' or 'google' for where to get embeddings from
        if embedding_type == 'google':
            self.EMBEDDING_DIM = 300  # length of embedding vectors
        if embedding_type == 'bert':
            self.EMBEDDING_DIM = 768  # length of embedding vectors
        self.LSTM_HIDDEN = LSTM_HIDDEN  # number of LSTM cells
        self.max_len=max_len  # maximum input sentence length, will be padded to this size
        self.n_labels = n_labels
        self.lstm = nn.LSTM(input_size=self.EMBEDDING_DIM, hidden_size=self.LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=2 * self.LSTM_HIDDEN, out_features=n_labels)
        self.batch_size = batch_size
        self.pad_token, self.pad_label = padding

    def forward(self, inputs):
        '''
        Implements a forward pass through the Bi-LSTM.
        inputs are a batch (list) of sentences.
        '''
        if self.embedding_type == 'bert':
            word_embeds = self._get_bert_embeds(inputs)
        elif self.embedding_type == 'google':
            word_embeds = self._get_google_embeds(inputs)

        word_embeds = nn.Dropout(p=0.2)(word_embeds)
        lstm_result, _ = self.lstm(word_embeds)
        lstm_result = nn.Dropout(p=0.3)(lstm_result)
        tags = self.linear(lstm_result)
        log_probs = F.softmax(tags, dim=2)
        return log_probs
    
    def _get_google_embeds(self, inputs):
        embeddings = torch.Tensor()
        for sentence in inputs:
            sentence_embeds = torch.Tensor()
            for word in sentence:
                if GoogleEmbs.__contains__(word):
                    embed = GoogleEmbs.get_vector(word)
                    embed.setflags(write = True)
                    embed = torch.from_numpy(embed)
                else:
                    embed = torch.zeros(300)  # the word is not in the model dictionary, so use zero vector
                sentence_embeds = torch.cat((sentence_embeds, embed), dim=0)
            embeddings = torch.cat((embeddings, sentence_embeds), dim=0)
        return embeddings.view(len(inputs), -1, self.EMBEDDING_DIM)

    def _get_bert_embeds(self, inputs):
        embeddings = torch.Tensor()
        for sentence in inputs:

            input_ids = bert_tokenizer.convert_tokens_to_ids(sentence)
            sentence_embeds = bert_model(input_ids)[0][0]

            embeddings = torch.cat((embeddings, sentence_embeds), dim=0)
        return embeddings.view(len(inputs), -1, self.EMBEDDING_DIM)
    
    def fit(self, documents, labels, LEARNING_RATE=0.01, EPOCHS=3):

        padded_documents = pad_inputs(documents, self.pad_token)
        padded_labels = pad_inputs(labels, self.pad_label)
        padded_labels = [list(map(int, sentence)) for sentence in padded_labels]

        self.train()
        torch.manual_seed(1)
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        loss_func = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)  # ignores loss for padding tokens
        data_iterator = DataIterator(batch_size=self.batch_size)

        for epoch in range(EPOCHS):
            
            total_tags = 0
            matched_tags = 0
            epoch_loss = 0

            for i, batch in enumerate(data_iterator(padded_documents, padded_labels)):
                pred_tags = self.forward(inputs=batch.inputs)
                
                # probability distribution for each tag across all words
                pred_tags = pred_tags.view(-1, self.n_labels)
                
                # true label for each word
                targets = torch.tensor(batch.targets).flatten()
                batch_loss = loss_func(pred_tags, targets)
                epoch_loss += batch_loss.item()
                
                # optimization
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # proportion of matched tags
                for pred_tag, true_tag in zip(pred_tags, targets):
                    true_tag = true_tag.item()
                    
                    if true_tag == -100: continue  # ignore tag of padding tokens
    
                    pred_tag_idx = torch.argmax(pred_tag).item()
                    if pred_tag_idx == true_tag:
                        matched_tags +=1
                    total_tags += 1
                                
            print(f"Epoch {epoch} loss: {epoch_loss:.2f},  total tags matched: {matched_tags / total_tags * 100:.2f}%")
    
    def pad_inputs(collection: List[List[int]], pad_token):
        to_series = [pd.Series(el) for el in collection]
        enc_matrix = (pd.concat(to_series, axis=1)
                        .reindex(range(self.max_len))
                        .fillna(pad_token)
                        .T)
        return enc_matrix.values.tolist()
