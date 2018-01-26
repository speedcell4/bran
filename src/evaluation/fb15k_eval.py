import numpy as np
import random
import sys


class FB15KEvaluator(object):
    def __init__(self, model, fb15k_dir, string_int_maps, session, test_key='valid'):

        self.eval_type = 'FB15K'
        self.model = model
        self.session = session
        self.kb_str_id_map = string_int_maps['kb_str_id_map']
        self.kb_id_str_map = string_int_maps['kb_id_str_map']
        self.entity_str_id_map = string_int_maps['entity_str_id_map']
        self.entity_id_str_map = string_int_maps['entity_id_str_map']
        self.ep_str_id_map = string_int_maps['ep_str_id_map']
        self.ep_id_str_map = string_int_maps['ep_id_str_map']
        self.test_key = test_key
        self.facts = set()
        self.splits = {}

        print('Initializing FB15k eval...')
        # read in data
        for f_name in ['train', 'valid', 'test']:
            with open(fb15k_dir + '/' + f_name + '.txt', 'r') as f:
                # randomly sample k lines from the file
                lines = [l.strip().split('\t')[:3] for l in f.readlines()]
                # read in all facts
                filtered_lines = [(e1, e2, rel) for (e1, e2, rel) in lines if
                                  e1 in self.entity_str_id_map and e2 in self.entity_str_id_map]

                print('filtered : ' + str(len(lines) - len(filtered_lines)))
                self.splits[f_name] = [(self.entity_str_id_map[e1], self.entity_str_id_map[e2], self.kb_str_id_map[rel])
                                       for e1, e2, rel in filtered_lines]
                self.facts = self.facts.union(self.splits[f_name])

        self.entity_batch = [i for i in range(len(self.entity_str_id_map))]

        with open(fb15k_dir + '/text_emnlp_args.txt', 'r') as f:
            _train_ep_set = ['::'.join(l.strip().split('\t')[:2]) for l in f.readlines()]
            train_ep_set = set([self.ep_str_id_map[ep] for ep in _train_ep_set
                                if ep in self.ep_str_id_map])

        # map (e1_id, e2_id) -> ep_id
        self.e1_e2_ep_map = {(self.entity_str_id_map[ep_str.split('::')[0]],
                              self. entity_str_id_map[ep_str.split('::')[1]]): ep_id
                             for ep_id, ep_str in self.ep_id_str_map.iteritems()
                             if ep_id in train_ep_set
                             }
        self.ep_e1_e2_map = {ep: e1_e2 for e1_e2, ep in self.e1_e2_ep_map.iteritems()}

        # get only positive test triples with valid eps
        filtered_triples = [(e1, e2, rel) for e1, e2, rel in self.splits[self.test_key]
                            if (e1, e2) in self.e1_e2_ep_map
                            and self.e1_e2_ep_map[(e1, e2)] in train_ep_set]

        # map each positive ep in the test set to valid negative eps
        self.ep_batch_map = {self.e1_e2_ep_map[(pos_e1, pos_e2)]: [
            self.e1_e2_ep_map[(pos_e1, neg_e2)] for neg_e2 in self.entity_id_str_map.iterkeys()
            if (pos_e1, neg_e2) in self.e1_e2_ep_map and
            (pos_e1, neg_e2, pos_rel) not in self.facts and
            self.e1_e2_ep_map[(pos_e1, neg_e2)] in train_ep_set
            ]
            for pos_e1, pos_e2, pos_rel in filtered_triples}

    def eval(self, block=False, take=10000, min_neg_count=0):
        take = min(len(self.splits[self.test_key]), take)
        count = 0.
        total_rank = 0.
        total_hits_at_10 = 0.
        neg_lens = 0
        pos_counts = 0
        print('\n Evaluating %d entity pairs' % take)
        for i, (pos_e1, pos_e2, pos_rel) in enumerate(self.splits[self.test_key]):
            if (pos_e1, pos_e2) in self.e1_e2_ep_map and pos_counts <= take:
                pos_ep = self.e1_e2_ep_map[(pos_e1, pos_e2)]
                if pos_ep in self.ep_batch_map and len(self.ep_batch_map[pos_ep]) >= min_neg_count:
                    sys.stdout.write('\r ep number: %d' % i)
                    sys.stdout.flush()
                    pos_counts += 1
                    neg_eps = self.ep_batch_map[pos_ep]
                    neg_lens += len(neg_eps)
                    ep_batch = [pos_ep] + neg_eps
                    rel_batch = [pos_rel] * (len(ep_batch))
                    tail_rank = self.rank_ep(ep_batch, rel_batch)
                    total_rank += (1 / tail_rank)
                    count += 1
                    if tail_rank <= 10:
                        total_hits_at_10 += 1
        print('\n pos_eps: %d   neg eps : %d' % (pos_counts, neg_lens))

        mrr = 100 * (total_rank / count)
        hits_at_10 = 100 * (total_hits_at_10 / count)
        return mrr, hits_at_10

    def rank_ep(self, ep_batch, rel_batch):
        feed_dict = {self.model.ep_batch: ep_batch, self.model.kb_batch: rel_batch, self.model.text_update: False}
        scores = self.session.run([self.model.kb_ep_score], feed_dict=feed_dict)[0]
        # get rank of the positive triple
        n, rank = len(rel_batch), 0
        ranked_preds = np.squeeze(scores).argsort()[::-1][:n]
        while ranked_preds[rank] != 0 and rank < n:
            rank += 1
        return float(rank + 1)
