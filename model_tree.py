from node import Node
import pandas as pd
from data_utils import fix_columns
import sklearn

from sklearn import linear_model


class ModelTree:
    def __init__(self, X_train, y_train, min_samples_leaf=2, learner="LinearRegression"):
        try:
            getattr(sklearn.linear_model, learner)
        except:
            print("DUDE, USE A LEGIT SKLEARN LINEAR_MODEL LEARNER!")
        self.root = Node(learner, X_train, y_train, depth=0)
        self.min_samples_leaf = min_samples_leaf
        self.learner = learner

    def predict(self, sample, tree=None):
        if tree is None:
            tree = self.root
        if len(tree.children) == 0:  # Leaf
            data = fix_columns(pd.get_dummies(sample), pd.get_dummies(tree.samples).columns)
            return tree.model.predict(data)
        if tree.threshold is None:  # Categorical
            found = False
            for index, child in enumerate(tree.children):
                if sample[tree.split_by_feature].values[0] == child.nominal_value:
                    print("Found")
                    node = child
                    found = True
                    break
            if not found:
                print("Not found")
                return tree.model.predict(
                    fix_columns(pd.get_dummies(sample), pd.get_dummies(tree.samples).columns))
        else:  # Numeric
            if sample[tree.split_by_feature].values[0] >= tree.threshold:
                node = tree.children[0]  # right node
            else:
                node = tree.children[1]
        return self.predict(sample, node)

    def build(self, node=None):
        if node is None:
            node = self.root
        node = self.split_node(node)
        if len(node.children) == 0:
            return node
        print("Splitted node at depth {d} by feature {f} and threshold {t}".format(d=node.depth,
                                                                                   f=node.split_by_feature,
                                                                                   t=node.threshold))
        for index, children in enumerate(node.children):
            print("Node {i} with {n} samples".format(i=index, n=len(children.samples)))
            node.children[index] = self.build(children)

        return node

    def split_node(self, node):
        features = node.samples.columns
        splits = []
        for feature in features:
            split = self.split_by_mse(feature, node)
            if split is not None:
                splits.append(split)
        if len(splits) == 0:
            return node
        curr_split = min(splits, key=lambda x: x.loss)
        return curr_split

    def split_by_mse(self, feature, node):
        if node.samples[feature].dtype == 'O':
            split = self.nominal_split(feature, node.samples, node.labels, node.depth + 1)
            if split is None or len(split.children) < 2:
                return None
        else:
            attr_splits = list(set(node.samples[feature].values))
            splits = []
            for split in attr_splits:
                split_children = self.numeric_split(feature, node.samples, node.labels, split, node.depth + 1)
                if split_children is not None:
                    splits.append(split_children)
            if len(splits) == 0:
                return None
            split = min(splits, key=lambda x: x.loss)
            # We filter None due to splits with low number of samples in one of the sides
            if node.nominal_value is not None:
                split.nominal_value = node.nominal_value
                # Important to keep the nominal_value in case we splitted a categorical node
        return split

    def node_from_nominal_group(self, df, depth, nominal_value):
        samples = df.drop('Y', axis=1)
        labels = df['Y']
        node = Node(self.learner, samples, labels, depth, nominal_value=nominal_value)
        return node

    def nominal_split(self, feature, train, label, depth):
        df = train.copy()
        df['Y'] = label
        filtered_nominal_groups = df.groupby(feature).filter(lambda x: len(x) >= self.min_samples_leaf).groupby(feature)
        if len(filtered_nominal_groups) < 2:
            return None

        nodes = []
        wmse = 0
        for nominal_value, nominal_group in filtered_nominal_groups:
            node = self.node_from_nominal_group(nominal_group, depth, nominal_value)
            nodes.append(node)
            wmse += node.loss

        split = Node(self.learner, train, label, depth, split_by_feature=feature, children=nodes)
        loss = wmse / len(train)
        split.loss = loss
        return split

    def numeric_split(self, feature, train, label, threshold, depth):
        mask = train[feature] >= threshold
        right_node_samples, right_node_labels = train[mask], label[mask]
        left_node_samples, left_node_labels = train[~mask], label[~mask]
        if len(left_node_samples) < self.min_samples_leaf or len(right_node_samples) < self.min_samples_leaf:
            return None

        right_node = Node(self.learner, right_node_samples, right_node_labels, depth)
        left_node = Node(self.learner, left_node_samples, left_node_labels, depth)

        split = Node(self.learner, train, label, depth, children=[right_node, left_node], split_by_feature=feature,
                     threshold=threshold)

        wloss = (right_node.loss + left_node.loss) / len(train)
        split.loss = wloss
        return split
