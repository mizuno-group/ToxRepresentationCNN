from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

class RepeatedStratifiedGroupKFold:

    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self._n_splits = n_splits
        self.n_repeats = n_repeats
        self._random_state = random_state
        
    def split(self, X, y=None, groups=None):
        k = self._n_splits
        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = np.std(y_counts_per_fold / y_distr)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
            
        rnd = check_random_state(self._random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            groups_num = np.max(groups) + 1
            y_counts_per_group = defaultdict(lambda : np.zeros(labels_num))
            y_distr = np.zeros(labels_num)
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = np.zeros((k, labels_num))
            groups_per_fold = defaultdict(set)
            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)
            for _, (g, y_counts) in enumerate(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1]))):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)
            
            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices

class Fold:
    """
    Class for data split
    """
    def __init__(self, df, id_col, X_col, y_col, group_col = None,
                split_type = None, n_splits = None,
                random_state = None, load_path = None, save_path = None):
        """
        Fold(self, df, id_col, X_col, y_col, group_col = None,
            split_type = None, n_splits = None,
            random_state = None, load_path = None, save_path = None)
        df : DataFrame which restore train data
        id_col : The column name of ID
        X_col : The column list of X
        """
        self._n_splits = n_splits
        self._id_col = id_col
        self._X_col = X_col
        self._y_col = y_col
        self._group_col = group_col
        self._random_state = random_state
        if self._group_col:
            needed_cols = [self._id_col] + self._X_col + self._y_col + [self._group_col]
        else:
            needed_cols = [self._id_col] + self._X_col + self._y_col
        self._df = df.copy()[needed_cols]
        if load_path:
            folds = pd.read_csv(load_path) # folds must have id_col and "fold" as columns
            self._df = pd.merge(self._df, folds, on = self._id_col, how = "left")
            self._n_splits = self._df["fold"].max() + 1 # fold must be 0-indexed
        elif split_type == "KFold":
            self._set_fold(KFold(n_splits = n_splits,
                            random_state = random_state, shuffle=True).split(self._df))
        elif split_type == "StratifiedKFold":
            self._set_fold(StratifiedKFold(n_splits = n_splits,
                            random_state = random_state).split(self._df, self._df[self._y_col]))
        elif split_type == "GroupKFold":
            self._set_fold(GroupKFold(n_splits = n_splits).split(self._df,
                            self._df[self._y_col], self._df[self._group_col]))
        elif split_type == "RepeatedStratiedGroupKFold":
            targets = self._df[X_col].values
            self._df['combined_tar'] = (targets * (2 ** np.arange(len(X_col)))).sum(axis=1)
            self._df['combined_tar'] = LabelEncoder().fit_transform(self._df['combined_tar'])
            rskf = RepeatedStratifiedGroupKFold(n_splits=n_splits, random_state=random_state)
            groups = LabelEncoder().fit_transform(self._df[group_col])
            for i, (train_idx, valid_idx) in enumerate(rskf.split(self._df, self._df.combined_tar, groups)):
                self._df.loc[valid_idx, 'fold'] = int(i)
            # we will implement in future.
            
        if save_path:
            self._df[[id_col, 'fold']].to_csv(save_path, index = False)

    def __len__(self):
        return self._n_splits
    
    def __getitem__(self, i):
        return self._df[self._df['fold'] != i], self._df[self._df['fold'] == i]

    @property
    def df(self):
        return self._df

    @property
    def y_col(self):
        return self._y_col

    def _set_fold(self, split):
        for n, (train_index, val_index) \
            in enumerate(split):
            self._df.loc[val_index, 'fold'] = int(n)
    

class CrossValidation:

    def __init__(self, fold, metrics = []):
        self._fold = fold
        self._metrics = metrics
        self._oof = fold.df.copy()
        self._pred_col = [f'pred_{c}_loss' for c in self._fold.y_col] + \
            [f'pred_{c}_{mtr.name}' for mtr in self._metrics for c in self._fold.y_col]
        self._oof[self._pred_col] = np.nan

    def __getitem__(self, idx):
        return self._fold[idx]

    @property
    def metrics(self):
        return self._metrics

    def set_oof_pred(self, i, pred, name):
        self._oof.loc[self._oof['fold'] == i,[f'pred_{c}_{name}'
                for c in self._fold.y_col]] = pred
    
    def save_oof(self, save_path):
        self._oof.to_csv(save_path, index = False)

    def calculate_accuracy(self, mtr):
        pred_col = [f'pred_{c}_{mtr.name}' for c in self._fold.y_col]
        if mtr.idx is None:
            return mtr.func(self._oof[self._fold.y_col].values, self._oof[pred_col].values)
        else:
            return mtr.func(self._oof[self._fold.y_col].values[:, mtr.idx], self._oof[pred_col].values[:, mtr.idx])

