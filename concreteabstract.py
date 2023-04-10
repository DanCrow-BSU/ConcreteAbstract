from tqdm.notebook import tqdm
import pandas as pd
import warnings
from nltk.tree import Tree
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from collections import defaultdict as dd
from operator import itemgetter

class ConcreteAbstract:
    
    def __init__(self, word_vectors, concr_scores, word_net, pos_count = 10, neg_count = 20):
        self.word_vectors = word_vectors
        self.concr_scores = concr_scores
        self.wn = word_net
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.test_pct = 'tbd'
        self.min_rating = 'tbd'

        
    ########################################
    # Initiate Abstraction Tree
    ########################################
    
    def init_abstraction_tree(self, min_rating=8):
        """Initialize the abstraction tree with leaf nodes."""
        self.min_rating = min_rating

        wac_words = list(self.word_vectors.keys())
        wn_words = set(i for i in self.wn.words())
        wn_wac_words = wn_words & set(wac_words)

        concr_scores_subset = self.concr_scores[self.concr_scores.RATING >= min_rating]
        leaf_words = [w for w in tqdm(wn_wac_words) if w in concr_scores_subset.index]

        # Get Leaf Synsets...
        leaf_synsets = [self.wn.synsets(w)[0] for w in leaf_words]

        # Initiate Abstraction Tree
        embeddings = [self.word_vectors[w] for w in leaf_words]
        data = {
            "SYNSET" : leaf_synsets,
            "WORD" : leaf_words,
            "DIST2LEAF": [0]*len(leaf_synsets),
            "NUM_LEAVES": [1]*len(leaf_synsets),
            "HYPERNYM": [[]]*len(leaf_synsets),
            "HYPONYMS": [[]]*len(leaf_synsets),
            "EMBEDDING" : embeddings

        }
        self.abstraction_tree = pd.DataFrame(data)
        self.abstraction_tree.index = self.abstraction_tree.SYNSET

        # Get True Leaf Synsets
        ancestors = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in leaf_synsets:
                #print(set(s.closure(lambda s: s.hypernyms())))
                ancestors = ancestors.union(set(s.closure(lambda s: s.hypernyms())))

        # Remove leaves that are ancestors of other leaves
        true_leaf_synsets = set(leaf_synsets) - ancestors
        ansestor_leaves = set(leaf_synsets) - true_leaf_synsets
        self.abstraction_tree.drop(ansestor_leaves, inplace=True)

        # Remove leaves that have the same Synset
        self.abstraction_tree.drop_duplicates(subset='SYNSET', inplace=True)

        #return abstraction_tree

        
    ########################################
    # Grow abstraction tree
    ########################################
    
    def _update_dist2leaf(self, synset, dist):
        h_dist = self.abstraction_tree.loc[synset, 'DIST2LEAF']
        if h_dist > dist:
            return

        self.abstraction_tree.at[synset, 'DIST2LEAF'] = dist

        # go up the hypernym chain and set the distances
        h = self.abstraction_tree.loc[synset, 'HYPERNYM']
        if len(h) != 0:
            self._update_dist2leaf(h[0], dist+1)


    def _update_num_leaves(self, synset, num_leaves):
        orig_num_leaves = self.abstraction_tree.loc[synset, 'NUM_LEAVES']
        self.abstraction_tree.at[synset, 'NUM_LEAVES'] = orig_num_leaves + num_leaves

        # go up the hypernym chain and update the leaves
        h = self.abstraction_tree.loc[synset, 'HYPERNYM']
        if len(h) != 0:
            self._update_num_leaves(h[0], num_leaves)

    def grow_abstraction_tree(self):
        """Takes an initial abstraction tree (containing only leaves) and grows
        the rest of the tree."""

        synset_list = list(self.abstraction_tree['SYNSET'])

        # Note: synset_list will grow as we loop through it, so tqdm may reach 100% before it's done
        for s in tqdm(synset_list):
            h = s.hypernyms()

            if len(h) == 0:
                continue

            h = h[0]

            if h not in self.abstraction_tree.SYNSET:
                synset_list.append(h)
                self.abstraction_tree.loc[h] = [
                    h,    # SYNSET
                    None, # WORD
                    0,    # DIST2LEAF
                    0,    # NUM_LEAVES
                    [],   # HYPERNYM
                    [],   # HYPONYMS
                    None  # EMBEDDING
                ]

            # Set DIST2LEAF
            s_dist = self.abstraction_tree.loc[s, 'DIST2LEAF']
            self._update_dist2leaf(h, s_dist+1)

            # Set NUM_LEAVES
            s_num_leaves = self.abstraction_tree.loc[s, 'NUM_LEAVES']
            self._update_num_leaves(h, s_num_leaves)

            # Add hypernym to synset
            self.abstraction_tree.loc[s, 'HYPERNYM'] = [h]

            # Add synset to hypernym
            self.abstraction_tree.loc[h, 'HYPONYMS'].append(s)


    ########################################
    # Display abstraction tree
    ########################################
    
    display_tree_call_limit = 0

    def _build_display_tree_safe(self, root_synset, char_limit=20):
        """Builds a tree in a safer way by limiting the number of calls to itself."""

        self.display_tree_call_limit -= 1

        if self.display_tree_call_limit <= 0:
            warnings.warn("_build_display_tree_safe reached maximum number of calls.")
            return "MAX"[:char_limit]

        row = self.abstraction_tree.loc[root_synset]
        root_name = row['SYNSET'].lemmas()[0].name()[:char_limit]
        if len(row['HYPONYMS']) == 0:
            return root_name

        children = [self._build_display_tree_safe(h, char_limit) for h in row['HYPONYMS']]

        return Tree(root_name, children)


    def build_display_tree(self, root_synset, char_limit=20, call_limit=500):
        """Display an abstraction tree starting with the root_synset.
        Returns an nltk Tree structure.
        Do not use on big trees!"""
        self.display_tree_call_limit = call_limit
        return self._build_display_tree_safe(root_synset, char_limit)

    
    ########################################
    # Get Positive Synsets
    ########################################
    
    def _is_leaf(self, synset):
        if synset not in self.abstraction_tree.index:
            return False
        return self.abstraction_tree.loc[synset, 'DIST2LEAF'] == 0

    def _classifier_capable(self, synset):
        return self.abstraction_tree.loc[synset, 'NUM_LEAVES'] >= self.pos_count

    def _is_embedding_capable(self, synset):
        """Return true if the synset is capable of having an embedding."""
        if synset not in self.abstraction_tree.index:
            return False
        return self._is_leaf(synset) or self._classifier_capable(synset)

    # _dist2leaf gives us the furthest leaf.
    # This can be used as a "sort" of way to determine abstraction level
    def _dist2leaf(self, synset):
        return self.abstraction_tree.loc[synset, 'DIST2LEAF']

    def _is_closer_to_leaf(self, synset, dist):
        """Return true if the synset is closer to a leaf than dist."""
        return self._dist2leaf(synset) < dist

    def _get_hyponyms(self, synset):
        """Return a list of hyponyms, or itself if there are none."""
        if synset not in self.abstraction_tree.index:
            return None
        hypos = self.abstraction_tree.loc[synset, 'HYPONYMS']
        if len(hypos) == 0:
            return [synset]
        else:
            return hypos

    def _count_embedding_capable(self, synset_list):
        """Given a list of synsets, returns a count of how many are capable of having an embedding."""
        return sum((self._is_embedding_capable(s) == True)*1 for s in synset_list)

    def _expand_hyponym_list(self, synset_list):
        hypos = []
        for s in synset_list:
            hypos += self._get_hyponyms(s)
        return hypos

    def _remove_embedding_incapable(self, synset_list):
        synset_list = np.array(synset_list)
        return list(synset_list[list(map(self._is_embedding_capable, synset_list))])

    def find_positive_examples(self, synset, depth=100):
        pos = self._get_hyponyms(synset)
        for _ in range(depth):
            pos = self._expand_hyponym_list(pos)
            if self._count_embedding_capable(pos) >= self.pos_count:
                return self._remove_embedding_incapable(pos)

        raise Exception("Reached depth of {} without finding enough positive example: {}".format(depth, synset))
    
    
    ########################################
    # Find negative examples...
    ########################################
    
    def find_negative_examples(self, synset, pos_examples):
        # All synsets
        neg = np.array(self.abstraction_tree['SYNSET'])
        # Embedding capable synsets
        neg = neg[list(map(self._is_embedding_capable, neg))]
        # Use only more concrete words (words that are closer to a leaf)
        dist2leaf = self.abstraction_tree.loc[synset, 'DIST2LEAF']
        neg = neg[list(map(lambda x: self._is_closer_to_leaf(x, dist2leaf), neg))]
        # neg examples not in positive examples
        neg = set(neg) - set(pos_examples)
        neg_examples = random.sample(list(neg), k=min(self.neg_count, len(neg)))
        return neg_examples
    
    
    ########################################
    # Collect Examples
    ########################################
    
    # add positive and negative examples for a single hypernym
    def _add_positive_negative_examples(self, synset):
        pos = self.find_positive_examples(synset)
        self.abstraction_tree.at[synset, 'POSITIVE'] = pos
        neg = self.find_negative_examples(synset, pos)
        self.abstraction_tree.at[synset, 'NEGATIVE'] = neg
        
    def _get_classifier_capable(self):
        """Get a list of synsets capable of having a classifier."""
        return [s for s in self.abstraction_tree['SYNSET'] if self._classifier_capable(s)]
    
    # add positive and negative examples for all hypernyms
    def add_pos_neg_all(self):
        """Add positive and negative examples for each calssifier capable synset."""
        self.abstraction_tree['POSITIVE'] = [[]]*len(self.abstraction_tree)
        self.abstraction_tree['NEGATIVE'] = [[]]*len(self.abstraction_tree)
        for s in tqdm(self._get_classifier_capable()):
            self._add_positive_negative_examples(s)


    ########################################
    # Build Train/Test datasets
    ########################################
    
    def _build_train_test(self, synset, test_pct):
        pos_examples = self.find_positive_examples(synset)
        neg_examples = self.find_negative_examples(synset, pos_examples)
        X = pos_examples + neg_examples
        y = list(np.ones(len(pos_examples))) + list(np.zeros(len(neg_examples)))
        return train_test_split(X, y, test_size=test_pct, stratify=y)
    
    
    def fill_out_train_test(self, test_pct=0.3):
        self.test_pct = test_pct
        self.abstraction_tree['X_TRAIN'] = [[]]*len(self.abstraction_tree)
        self.abstraction_tree['X_TEST']  = [[]]*len(self.abstraction_tree)
        self.abstraction_tree['Y_TRAIN'] = [[]]*len(self.abstraction_tree)
        self.abstraction_tree['Y_TEST']  = [[]]*len(self.abstraction_tree)

        synsets = self._get_classifier_capable()
        for s in tqdm(synsets):
            X_train, X_test, y_train, y_test  = self._build_train_test(s, test_pct)
            self.abstraction_tree.at[s, 'X_TRAIN'] = X_train
            self.abstraction_tree.at[s, 'X_TEST']  = X_test
            self.abstraction_tree.at[s, 'Y_TRAIN'] = y_train
            self.abstraction_tree.at[s, 'Y_TEST']  = y_test
       
    
    ########################################
    # Train
    ########################################
    
    def _coefs(self, classifier):
        return classifier.coef_[0]

    def build_classifiers(self):
        self.abstraction_tree['CLASSIFIER']  = [None]*len(self.abstraction_tree)
        cc = self._get_classifier_capable()
        # Start with synsets close to leaves and work our way up to more abstract hypernyms
        cc.sort(key=self._dist2leaf)
        for ss in tqdm(cc):
            #print(ss)
            X_train = self.abstraction_tree.loc[ss, 'X_TRAIN']
            y_train = self.abstraction_tree.loc[ss, 'Y_TRAIN']
            X_train = list(self.abstraction_tree.loc[X_train, 'EMBEDDING'])

            lr = LogisticRegression(C=0.25, max_iter=1000)
            lr.fit(X_train, y_train)
            self.abstraction_tree.at[ss, 'CLASSIFIER'] = lr
            self.abstraction_tree.at[ss, 'EMBEDDING'] = self._coefs(lr)



    ########################################
    # Evaluate (vs negative examples) 
    ########################################
    
    def comp_random_baseline(self):
        return 0.5
    
    def comp_most_common_baseline(self):
        pos, tot = 0, 0
        for ss in self._get_classifier_capable():
            y = self.abstraction_tree.loc[ss, 'Y_TRAIN']
            pos += sum(y)
            tot += len(y)
            y = self.abstraction_tree.loc[ss, 'Y_TEST']
            pos += sum(y)
            tot += len(y)
        return max(pos/tot, 1-(pos/tot))
    
    def evaluate_vs_negative_examples(self):
        """Evaluate each classifier by itself using a set of negative examples.
        Return the combined accuracy score of all classifiers."""
        golds = []
        preds = []
        cc = self._get_classifier_capable()
        # Evaluate in order (Not stricly needed, but helpful)
        cc.sort(key=self._dist2leaf)
        for ss in cc:
            X_test = self.abstraction_tree.loc[ss, 'X_TEST']
            y_test = self.abstraction_tree.loc[ss, 'Y_TEST']
            X_test = list(self.abstraction_tree.loc[X_test, 'EMBEDDING'])

            c = self.abstraction_tree.loc[ss, 'CLASSIFIER']
            pred = c.predict_proba(X_test)
            pred = list(np.argmax(pred, axis=1))

            preds = preds + pred
            golds = golds + y_test
        return metrics.accuracy_score(golds, preds)
    
    
    ########################################
    # Evaluate (vs other classifiers) (i.e. Distractors)
    ########################################
    
    def _get_all_classifiers(self):
        classifiers = dd(None)
        for ss in self._get_classifier_capable():
            classf = self.abstraction_tree.loc[ss, 'CLASSIFIER']
            classifiers[ss] = classf

        return classifiers
    
    def _test_classifier_single(self, classifiers, x_ss, gold_ss, num_distractors=5):
        #gold_ss = wn.synset('conveyance.n.03')
        vec = self.abstraction_tree.loc[x_ss, 'EMBEDDING']
        probs = [(gold_ss, classifiers[gold_ss].predict_proba([vec])[0][1])]
        distractors=[]
        for _ in range(num_distractors):
            ss = random.choice(list(classifiers.keys()))
            if x_ss in self.abstraction_tree.loc[ss, 'POSITIVE']:
                continue
            if ss in distractors:
                continue
            distractors.append(ss)
            probs.append((ss, classifiers[ss].predict_proba([vec])[0][1]))
        best = max(probs, key=itemgetter(1))
        #print(x_ss, best[0], gold_ss)
        score = (best[0] == gold_ss)*1
        num_possibilities = len(probs) # To help calculate baseline
        return score, num_possibilities
    
    def _test_classifier_full(self, classifiers, gold_ss, num_distractors=5):
        test = self.abstraction_tree.loc[gold_ss, 'X_TEST']
        pos = self.abstraction_tree.loc[gold_ss, 'POSITIVE']
        X = set(test).intersection(pos)
        total_score = 0
        total_possibilities = 0
        total_tests = len(X)
        for x in X:
            score, num_possibilities = self._test_classifier_single(classifiers, x, gold_ss, num_distractors)
            total_score += score
            total_possibilities += num_possibilities
        return total_score, total_tests, total_possibilities
    
    # test/evaluate all classifiers
    def _evaluate_all_classifiers(self, classifiers, num_distractors=10):
        """Evaluates all classifiers vs a number of distractors.
        Returns a score and the computed random baseline."""
        
        grand_total_score, grand_total_tests, grand_total_possibilities = 0,0,0
        for gold_ss in tqdm(classifiers.keys()):
            #print(gold_ss)
            total_score, total_tests, total_possibilities = self._test_classifier_full(classifiers, gold_ss, num_distractors)
            #print(total_score, total_tests, total_possibilities)
            grand_total_score += total_score
            grand_total_tests += total_tests
            grand_total_possibilities += total_possibilities

        score = grand_total_score/grand_total_tests
        baseline = grand_total_tests/grand_total_possibilities

        return score, baseline
    
    def evaluate_vs_distractors(self, num_distractors=10):
        """Evaluates all classifiers vs a number of distractors.
        Returns a score and the computed random baseline."""
        
        classifiers = self._get_all_classifiers()
        return self._evaluate_all_classifiers(classifiers, num_distractors)
        
        
    ########################################
    # Extra Functions
    ########################################
    
    def get_classifier_count(self):
        return len(self._get_classifier_capable())
    
    def get_lemma_count(self):
        sum(len(ss.lemmas()) for ss in self._get_classifier_capable())
        
    def build_all(self, min_rating=8, verbose=True):
        if verbose:
            print("Initiate Abstraction Tree")
            self.init_abstraction_tree(min_rating)
            print("Grow Abstraction Tree")
            self.grow_abstraction_tree()
            print("Add Positive and Negative examples")
            self.add_pos_neg_all()
            print("Fill out Train and Test sets")
            self.fill_out_train_test()
            print("Build Classifiers")
            self.build_classifiers()
            print("Done")
        else:
            global tqdm
            tqdm_hold = tqdm
            tqdm = lambda x:x
            self.init_abstraction_tree(min_rating)
            self.grow_abstraction_tree()
            self.add_pos_neg_all()
            self.fill_out_train_test()
            self.build_classifiers()
            tqdm = tqdm_hold
        