import codecs
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import gensim.models
GoogleEmbs = gensim.models.KeyedVectors.load_word2vec_format(
                                './nlp_project/models/GoogleNews-50k.bin', binary=True)

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

    Pass true and predicted labels to pass_batch() each batch, 
    then use f1_score() for epoch level entity F1 (resets metrics).

    Note: Assumes consecutive entity labels are multi-word entities.
    Source for metrics: https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    """
    
    def __init__(self, pad_label=2):
        self.pad_label = pad_label
        self.metrics = np.array([0,0,0,0,0,0,0])  # initializing epoch-level [COR, PAR, INC, MIS, SPU, ACT, POS] array to 0
    
    def _pre_process(self, true_labels, pred_labels):
        """
        Inputs have shape (32, 100) and (32, 100, 3)
        Returns lists of ints (with padding removed).
        """
        true_labels = true_labels.detach().numpy()
        # print(f"before processing: {np.asarray(true_labels).shape}, {np.asarray(pred_labels.detach()).shape}")
        pad_mask = true_labels != self.pad_label
        true_labels = list(map(int,true_labels[pad_mask]))
        # print(f"during processing: {np.asarray(true_labels).shape}, {np.asarray(pred_labels.detach()).shape}")
        pred_labels = torch.argmax(pred_labels, dim=1, keepdim=False)
        pred_labels = pred_labels.detach().numpy()
        pred_labels = pred_labels[pad_mask]
        # print(f"processed shape: {np.asarray(true_labels).shape}, {np.asarray(pred_labels).shape}")
        return true_labels, pred_labels

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
            INC = ACT
            PAR = MIS = 0
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
        # print(f"batch shapes: {true_labels_batch.shape, pred_labels_batch.shape}")
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
        elif relaxed: recall = (COR+.5*PAR)/POS
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

class F1_error_evaluator(F1_evaluator):
    """
    Like F1_evaluator, but f1_score(verbose=True) also outputs a DataFrame of errors in epoch.
    Initiation requires sentences, BIOlabels and domains.
    """
    def __init__(self, sentences, BIOlabels, domains, pad_label=2):
        self.pad_label = pad_label
        self.metrics = np.array([0,0,0,0,0,0,0])  # initializing epoch-level [COR, PAR, INC, MIS, SPU, ACT, POS] array to 0
        self.BIOlabels = BIOlabels
        self.domains = domains
        self.sentences = sentences
        self.errors = pd.DataFrame({'error_type':[],'entity':[],'entity_BIO':[],'sentence':[],'sentence_BIO':[],'domain':[]}, dtype=str)
        self.sentence_index = 0

    def _pre_process(self, true_labels, pred_labels):
        """
        Inputs have shape (32, 100) and (32, 100, 3)
        Returns lists of ints (with padding removed).
        """
        true_labels = true_labels.detach().numpy()
        pad_mask = true_labels != self.pad_label
        true_labels = list(map(int,true_labels[pad_mask]))
        pred_labels = torch.argmax(pred_labels, dim=1, keepdim=False)
        pred_labels = pred_labels.detach().numpy()
        pred_labels = pred_labels[pad_mask]
        return true_labels, pred_labels

    def _get_sentence_metrics(self, true_labels, pred_labels):
        sentence = self.sentences[self.sentence_index]
        sentence_BIO_labels = self.BIOlabels[self.sentence_index]
        sentence_domain = self.domains[self.sentence_index]

        true_labels, pred_labels = self._pre_process(true_labels, pred_labels)  # Turn torch tensors into lists of ints

        true_ents = self._entities(true_labels)  # Get entity indeces
        pred_ents = self._entities(pred_labels)
        
        POS = len(true_ents)  # number of true entities 
        ACT = len(pred_ents)  # number of predicted entities
        correct_ents = [e for e in true_ents if e in pred_ents]
        COR = len(correct_ents)  # number of correctly predicted entities

        if POS == 0:  # no true entities
            SPU = ACT
            INC = ACT
            PAR = MIS = 0
            if ACT > 0:  # add spurious entities to errors
                for ent in pred_ents:
                    _BIO = [sentence_BIO_labels[i] for i in ent]
                    _entity_string = " ".join([sentence[i] for i in ent]) # merging entity words
                self.errors = pd.concat([pd.DataFrame([["SPU",_entity_string,_BIO,sentence,sentence_BIO_labels,sentence_domain]], columns=self.errors.columns), self.errors], ignore_index=True)
            self.sentence_index += 1
            return np.array([COR, PAR, INC, MIS, SPU, ACT, POS])
        
        # Comupting number of partial matches (PAR) and missed entities (MIS)
        PAR = MIS = 0
        partial_ents = []
        pred_entity_indeces = [x for e in pred_ents for x in e]
        non_predicted_entities = [e for e in true_ents if e not in pred_ents]
        for ent in non_predicted_entities:
            for index in ent:
                if index in pred_entity_indeces:
                    # part of the entity was predicted
                    PAR += 1
                    partial_ents.append(ent)
                    _BIO = [sentence_BIO_labels[i] for i in ent]
                    _entity_string = " ".join([sentence[i] for i in ent]) # merging entity words
                    self.errors = pd.concat([pd.DataFrame([["PAR",_entity_string,_BIO,sentence,sentence_BIO_labels,sentence_domain]], columns=self.errors.columns), self.errors], ignore_index=True)
                    break
            else:
                # none of the entity was predicted
                MIS += 1
                _BIO = [sentence_BIO_labels[i] for i in ent]
                _entity_string = " ".join([sentence[i] for i in ent]) # merging entity words
                self.errors = pd.concat([pd.DataFrame([["MIS",_entity_string,_BIO,sentence,sentence_BIO_labels,sentence_domain]], columns=self.errors.columns), self.errors], ignore_index=True)
        
        # SPU = predicted entities which are (wholly) nonexistant in true entities
        SPU = ACT - COR - PAR
        
        spurious_ents = []
        wrong_predicted_ents = [e for e in pred_ents if e not in true_ents]
        partial_ents_indeces = [x for e in partial_ents for x in e]
        for ent in wrong_predicted_ents:
            for i in ent:
                if i in partial_ents_indeces:
                    break
            else:
                spurious_ents.append(ent)

        for spu_ent in spurious_ents:
            _BIO = [sentence_BIO_labels[i] for i in spu_ent]
            _entity_string = " ".join([sentence[i] for i in spu_ent]) # merging entity words
            self.errors = pd.concat([pd.DataFrame([["SPU",_entity_string,_BIO,sentence,sentence_BIO_labels,sentence_domain]], columns=self.errors.columns), self.errors], ignore_index=True)

        INC = MIS + SPU  # INC = Incorrectly predicted entities (missing or spurious)

        self.sentence_index += 1
        return np.array([COR, PAR, INC, MIS, SPU, ACT, POS])  #  return metrics of sentence
        
    def _reset_variables(self):
        # epoch level metrics are reset
        self.metrics = np.array([0,0,0,0,0,0,0])
        self.BIOlabels = None
        self.domains = None
        self.errors = pd.DataFrame({'error_type':[],'entity':[],'entity_BIO':[],'sentence':[],'sentence_BIO':[],'domain':[]}, dtype=str)
        self.sentence_index = 0

    def f1_score(self, relaxed=False, verbose=False):
        """
        relaxed: whether relaxed-match evaluation is used instead of strict (default)
        verbose: whether to return (precision, recall, f1) and error DataFrame
        """
        COR, PAR, INC, MIS, SPU, ACT, POS = list(self.metrics)

        if POS == 0: recall = 0
        else: recall = COR/POS
        
        if ACT == 0: precision = 0
        elif relaxed: precision = (COR+.5*PAR)/ACT  # Relaxed F1 precision
        else: precision = COR/ACT  # Strict F1 precision
        
        if precision==0 and recall==0: f1 = 0
        else: f1 = 2*precision*recall / (precision+recall)

        self._reset_variables()  # Reset all the Evaluator's variables

        if verbose:
            return precision, recall, f1
        else:
            return f1

class Train1BiLSTM(torch.nn.Module):
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
    
    def fit(self, train, dev=None, epochs=3, print_metrics=False, learning_rate=0.05):
        
        documents, labels = train
        padded_documents, padded_labels = self._pad_data(documents, labels)  # Padding training data

        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        loss_func = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=self.pad_label)  # ignores loss for padding label
        Evaluator = F1_evaluator(self.pad_label)
        optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            self.train()
            for i, batch in enumerate(self.data_iterator(padded_documents, padded_labels)):
                pred_tags = self.forward(inputs=batch.inputs)
                # pred_tags = pred_tags.view(-1, self.n_labels) # probability distribution for each tag across all words in batch
                targets = torch.tensor(batch.targets)  # 
                # print(f"train batch shapes: {targets.shape, pred_tags.shape}")
                Evaluator.pass_batch(targets, pred_tags)  # passing batch labels to evaluator
                batch_loss = loss_func(pred_tags.permute(0,2,1), targets)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
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
                dev_f1, dev_metrics = self._dev_evaluate(x_dev, y_dev)
                self.dev_f1_log.append(dev_f1)
                if print_metrics:
                    COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f} \n dev metrics: {ACT} ACT, {POS} POS, {COR} COR, {INC} INC ({PAR} PAR, {MIS} MIS, {SPU} SPU)")
                else:
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f}")

    def _dev_evaluate(self, x_dev, y_dev):
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
            targets = torch.tensor(batch.targets)
            # print(f"dev batch shapes: {targets.shape, pred_dev.shape}")
            Evaluator.pass_batch(targets, pred_dev)
        dev_metrics = Evaluator.metrics
        dev_f1 = Evaluator.f1_score()
        return dev_f1, dev_metrics

    def evaluate(self, x_dev, y_dev, BIOlabels, domains, print_metrics=True, return_errors=True):
        """
        Evaluates model performance on supplied data.
        print_metrics set to print out f1, precision, recall and metrics by default.
        return_errors returns error DataFrame by default.
        """
        padded_dev_docs, padded_dev_labs = self._pad_data(x_dev, y_dev)  # Padding data
        self.eval()
        Evaluator = F1_error_evaluator(x_dev, BIOlabels, domains, pad_label=self.pad_label)
        for i, batch in enumerate(self.data_iterator(padded_dev_docs, padded_dev_labs)):
            with torch.no_grad():
                pred_dev = self.forward(batch.inputs)
            targets = torch.tensor(batch.targets)
            Evaluator.pass_batch(targets, pred_dev)
        dev_metrics = Evaluator.metrics
        error_df = Evaluator.errors
        precision, recall, f1 = Evaluator.f1_score(verbose=True)
        if print_metrics:
            print(f"F1: {f1:.3f} precision: {precision:.3f} recall: {recall:.3f}")
            COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
            print(f"Metrics: {ACT} ACT, {POS} POS, {COR} COR, {INC} INC ({PAR} PAR, {MIS} MIS, {SPU} SPU)")
        if return_errors:
            # error_df["general_label"] = error_df["BIO_label"][0]
            return error_df

class Train2BiLSTM(Train1BiLSTM):
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

        self.poly_data_iterator = PolyDataIterator(batch_size=self.batch_size)
        self.data_iterator = DataIterator(batch_size=self.batch_size)

        # Logs for performance in each training epoch
        self.train_f1_log = []
        self.dev_f1_log = []
    
    def fit(self, 
            train,
            train2, 
            dev=None,
            epochs=20, 
            print_metrics=False, 
            learning_rate=0.005,
            alpha=None):
        
        documents, labels = train
        pseudo_docs, pseudo_labels = train2

        # padding training data
        padded_documents, padded_labels = self._pad_data(documents, labels) 
        pseudo_docs, pseudo_labels = self._pad_data(pseudo_docs, pseudo_labels)

        loss_func = WeightedCrossEntropy(epochs, pad_label=self.pad_label)
        Evaluator = F1_evaluator(self.pad_label)
        optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            epoch_loss = 0
            self.train()
            for labeled, paraphrased in self.poly_data_iterator([padded_documents, pseudo_docs], 
                                                                [padded_labels, pseudo_labels]):
                pred_tags = self.forward(inputs=labeled[0])
                pred_pseudo_tags = self.forward(inputs=paraphrased[0])
                # probability distribution for each tag across all words
                # pred_tags = pred_tags.view(-1, self.n_labels)
                # pred_pseudo_tags = pred_pseudo_tags.view(-1, self.n_labels)
                # true label for each word
                targets = torch.tensor(labeled[1])  # .flatten()
                pseudo_targets = torch.tensor(paraphrased[1])  # .flatten()
                # passing all batch labels to evaluator
                Evaluator.pass_batch(torch.cat((targets, pseudo_targets), 0), 
                                     torch.cat((pred_tags, pred_pseudo_tags), 0))
                batch_loss = loss_func(pred_tags.permute(0,2,1), pred_pseudo_tags.permute(0,2,1), targets, pseudo_targets, alpha=(alpha or epoch / epochs))
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_metrics = Evaluator.metrics
            train_p, train_r, train_f1 = Evaluator.f1_score(verbose=True)
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
                dev_f1, dev_metrics = self._dev_evaluate(x_dev, y_dev)
                self.dev_f1_log.append(dev_f1)
                if print_metrics:
                    COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f} \n dev metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
                else:
                    print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f}")


class WeightedCrossEntropy(nn.Module):
    def __init__(self, epochs, pad_label):
        super(WeightedCrossEntropy, self).__init__()
        self.epochs = epochs
        self.pad_label = pad_label

    def forward(self, labeled_output, pseudo_output, labeled_target, pseudo_target, alpha=0.5):
        L = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.pad_label)
        labeled_loss = L(labeled_output, labeled_target) / len(labeled_target)
        pseudo_loss = L(pseudo_output, pseudo_target) / len(pseudo_target)
        return (1-alpha) * labeled_loss + alpha * pseudo_loss  # added 1-alpha term to normalize

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


# class old_Train2BiLSTM(torch.nn.Module):
#     def __init__(self,
#                 hidden_size=20,
#                 max_len=100,
#                 n_labels=3,
#                 batch_size=32,
#                 pad_token="<PAD>",
#                 pad_label=2,
#                 embedding_dim=300
#                 ):
#         super().__init__()
#         self.embedding_dim = embedding_dim  # length of embedding vectors
#         self.hidden_size = hidden_size  # number of LSTM cells
#         self.max_len=max_len  # maximum input sentence length, will be padded to this size
#         self.n_labels = n_labels
#         self.batch_size = batch_size
#         self.pad_token = pad_token
#         self.pad_label = pad_label

#         self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
#         self.linear = nn.Linear(in_features=2 * self.hidden_size, out_features=n_labels)

#         self.poly_data_iterator = PolyDataIterator(batch_size=self.batch_size)
#         self.data_iterator = DataIterator(batch_size=self.batch_size)

#         # Logs for performance in each training epoch
#         self.train_f1_log = []
#         self.dev_f1_log = []

#     def _pad_inputs(self, collection: List[List[int]], padding_token):
#         to_series = [pd.Series(el) for el in collection]
#         enc_matrix = (pd.concat(to_series, axis=1)
#                         .reindex(range(self.max_len))
#                         .fillna(padding_token)
#                         .T)
#         collection = enc_matrix.values.tolist()
#         return collection

#     def _pad_data(self, documents, labels):
#         padded_documents = self._pad_inputs(documents, self.pad_token)
#         padded_labels = self._pad_inputs(labels, self.pad_label)
#         padded_labels = [list(map(int,sentence)) for sentence in padded_labels]
#         return padded_documents, padded_labels

#     def forward(self, inputs):
#         '''
#         Implements a forward pass through the BiLSTM.
#         inputs are a batch (list) of sentences.
#         '''
#         word_embeds = self._get_google_embeds(inputs)
#         lstm_result, _ = self.lstm(word_embeds)
#         tags = self.linear(lstm_result)
#         log_probs = torch.nn.functional.softmax(tags, dim=2)
#         return log_probs
    
#     def _get_google_embeds(self, inputs):
#         embeddings = torch.Tensor()
#         for sentence in inputs:
#             sentence_embeds = torch.Tensor()
#             for word in sentence:
#                 if GoogleEmbs.__contains__(word):
#                     embed = GoogleEmbs.get_vector(word)
#                     embed.setflags(write = True)
#                     embed = torch.from_numpy(embed)
#                 else:
#                     embed = torch.zeros(300)  # the word is not in the model dictionary, so use zero vector
#                 sentence_embeds = torch.cat((sentence_embeds, embed), dim=0)
#             embeddings = torch.cat((embeddings, sentence_embeds), dim=0)
#         return embeddings.view(len(inputs), -1, self.embedding_dim)
    
#     def fit(self, 
#             documents, 
#             labels, 
#             dev=None,
#             pseudo=None, 
#             epochs=3, 
#             print_metrics=False, 
#             learning_rate=0.05,
#             alpha=None):
        
#         padded_documents, padded_labels = self._pad_data(documents, labels)  # Padding training data

#         pseudo_docs, pseudo_labels = pseudo
#         pseudo_docs, pseudo_labels = self._pad_data(pseudo_docs, pseudo_labels)  # Padding pseudo data

#         loss_func = WeightedCrossEntropy(epochs, pad_label=self.pad_label)
#         Evaluator = F1_evaluator(self.pad_label)
#         optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        
#         for epoch in range(epochs):
#             epoch_loss = 0
#             self.train()
#             for labeled, paraphrased in self.poly_data_iterator([padded_documents, pseudo_docs], 
#                                                                 [padded_labels, pseudo_labels]):
#                 pred_tags = self.forward(inputs=labeled[0])
#                 pred_pseudo_tags = self.forward(inputs=paraphrased[0])
#                 # probability distribution for each tag across all words
#                 pred_tags = pred_tags.view(-1, self.n_labels)
#                 pred_pseudo_tags = pred_pseudo_tags.view(-1, self.n_labels)
#                 # true label for each word
#                 targets = torch.tensor(labeled[1]).flatten()
#                 pseudo_targets = torch.tensor(paraphrased[1]).flatten()
#                 # passing all batch labels to evaluator
#                 Evaluator.pass_batch(torch.cat((targets, pseudo_targets), 0), 
#                                      torch.cat((pred_tags, pred_pseudo_tags), 0))
#                 batch_loss = loss_func(pred_tags, pred_pseudo_tags, targets, pseudo_targets, alpha=(alpha or epoch / epochs))
#                 epoch_loss += batch_loss.item()
#                 batch_loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#             train_metrics = Evaluator.metrics
#             train_p, train_r, train_f1 = Evaluator.f1_score(verbose=True)
#             self.train_f1_log.append(train_f1)

#             if dev is None:  # print performance and go to next epoch if no dev data is supplied
#                 if print_metrics:
#                     COR, PAR, INC, MIS, SPU, ACT, POS = list(train_metrics)
#                     print(f"Epoch {epoch}, train: {train_f1:.3f} \n train metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
#                 else:
#                     print(f"Epoch {epoch}, train: {train_f1:.3f}, loss: {epoch_loss:.3f}")
#             else:
#                 # Dev evaluation
#                 x_dev, y_dev = dev
#                 dev_f1, dev_metrics = self.evaluate(x_dev, y_dev, print_metrics=False, return_metrics=True)
#                 self.dev_f1_log.append(dev_f1)
#                 if print_metrics:
#                     COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
#                     print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f} \n dev metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
#                 else:
#                     print(f"Epoch {epoch}, train: {train_f1:.3f}, dev: {dev_f1:.3f}")

#     def evaluate(self, x_dev, y_dev, print_metrics=True, return_metrics=False):
#         """
#         Evaluates model performance on supplied data.
#         print_metrics set to print out metrics by default.
#         return_metrics (optionally) returns F1 and metrics.
#         """
#         padded_dev_docs, padded_dev_labs = self._pad_data(x_dev, y_dev)  # Padding data
#         self.eval()
#         Evaluator = F1_evaluator(self.pad_label)
#         for i, batch in enumerate(self.data_iterator(padded_dev_docs, padded_dev_labs)):
#             with torch.no_grad():
#                 pred_dev = self.forward(batch.inputs)
#             pred_dev = pred_dev.view(-1, self.n_labels) 
#             targets = torch.tensor(batch.targets).flatten()
#             Evaluator.pass_batch(targets, pred_dev)
#         dev_metrics = Evaluator.metrics
#         dev_p, dev_r, dev_f1 = Evaluator.f1_score(verbose=True)
#         if print_metrics:
#             COR, PAR, INC, MIS, SPU, ACT, POS = list(dev_metrics)
#             print(f"F1: {dev_f1:.3f} precision: {dev_p:.3f} recall: {dev_r:.3f}")
#             print(f"Metrics: {ACT} ACT, {POS} POS, {COR} COR, {PAR} PAR, {INC} INC ({MIS} MIS, {SPU} SPU)")
#         if return_metrics:
#             return dev_f1, dev_metrics


