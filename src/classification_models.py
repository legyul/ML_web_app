import pandas as pd
import numpy as np
import math
from collections import defaultdict
import statistics
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Models for the numeric dataset
class numeric:
    # Function to find the target column
    def find_target(data):
        numeric_columns = data.select_dtypes(include='number').columns.tolist()
        categorical_columns = data.select_dtypes(exclude='number').columns.tolist()
        return categorical_columns
    
    # Gausian Naive Bayes model
    class gausian_naive_bayes:
        def __init__(self):
            self.class_stats = defaultdict(lambda: defaultdict(list))
            self.class_freq = defaultdict(int)
            self.classes = set()
        
        def fit(self, X, y):
            if isinstance(X, pd.DataFrame):
                X = X.values.tolist()
            if isinstance(y, pd.Series):
                y = y.tolist()
    
            self.classes = set(y)

            # Classificate the data by class
            for label, features in zip(y, X):
                self.class_freq[label] += 1
                for i, value in enumerate(features):
                    self.class_stats[label][f'feature{i+1}_values'].append(value)
            
            # Calculate the mean and standard deviation of each class
            for label in self.classes:
                for i in range(len(X[0])):
                    values = self.class_stats[label][f'feature{i+1}_values']
                    self.class_stats[label][f'feature{i+1}_mean'] = statistics.mean(values)
                    self.class_stats[label][f'feature{i+1}_std'] = statistics.stdev(values) if len(values) > 1 else 1e-9
        
        def pdf(self, x, mean, std):
            if std == 0:
                return 1.0 if x == mean else 0.0
            
            coefficient = 1 / math.sqrt(2 * math.pi) * std
            exponent = math.exp(-0.5 * ((x - mean) / std) ** 2)
            
            return coefficient * exponent

        def predict(self, X):
            prediction = []
            total_samples = sum(self.class_freq.values())

            if isinstance(X, pd.DataFrame):
                X = X.values.tolist()
            
            for features in X:
                score = {}
                for label in self.classes:
                    score[label] = math.log(self.class_freq[label] / total_samples)
                
                for i, feature in enumerate(features):
                    mean = self.class_stats[label][f'feature{i+1}_mean']
                    std = self.class_stats[label][f'feature{i+1}_std']
                    score[label] += math.log(self.pdf(feature, mean, std) + 1e-9)
            
                probability = {label: math.exp(score[label]) for label in score}
                total_probability = sum(probability.values())
                percentage = {label: (prob / total_probability) * 100 for label, prob in probability.items()}

                predicted_label = max(percentage, key=percentage.get)
                prediction.append(predicted_label)
        
            return prediction
    
    # Decision Tree model
    class decision_tree:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def column_types(data):
            numeric_columns = data.select_dtypes(include='number').columns.tolist()
            categorical_columns = data.select_dtypes(exclude='number').columns.tolist()
            
            return numeric_columns, categorical_columns

        def numeric_column_statistics(data, numeric_columns):
            for col in numeric_columns:
                print(f"- {col}: mean = {data[col].mean():.2f}, min = {data[col].min()}, max = {data[col].max()}")

        def find_label(data, categorical_columns):
            counts = data[categorical_columns].value_counts()
            label = [item[0] for item in counts[counts > 1].index.tolist()]

            return label

        def entropy(y):
            classes, counts = np.unique(y, return_counts=True)
            probability = counts / len(y)
            entropy = -np.sum(probability * np.log2(probability))
            
            return entropy

        def gini_index(y):
            classes, counts = np.unique(y, return_counts=True)
            probability = counts / len(y)
            gini_index = 1 - np.sum(probability ** 2)

            return gini_index

        def indi_best_split(feature, target):
            unique_values = np.unique(feature)
            best_gain = -1
            best_split = None

            parent_entropy = numeric.decision_tree.entropy(target)

            for split in unique_values:
                left_mask = feature <= split
                right_mask = feature > split

                left_entropy = numeric.decision_tree.entropy(target[left_mask])
                right_entropy = numeric.decision_tree.entropy(target[right_mask])

                weighted_entropy = (len(target[left_mask]) / len(target)) * left_entropy + \
                                    (len(target[right_mask]) / len(target)) * right_entropy
                
                info_gain = parent_entropy - weighted_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_split = split
                
            return best_split, best_gain

        def all_best_split(X, y):
            best_feature = None
            best_split_value = None
            best_gain = -1

            for col in X.columns:
                #print(f"\nChecking feature: {col}")
                split, gain = numeric.decision_tree.indi_best_split(X[col].values, y)
                #print(f"- Split Value: {split}, Information Gain: {gain:.4f}")
            
            if gain > best_gain:
                best_gain = gain
                best_split_value = split
                best_feature = col
            
            return best_feature, best_split_value, best_gain
        
        @staticmethod
        def build_tree(X, y, depth=0):
            '''
            Train Decision tree model.
            X: 2D array or DataFrame containing the characteristics of each sample
            y: List of corresponding labels (class)
            '''
            # Condition of stopping processing
            if len(np.unique(y)) == 1:  # If there is same label
                return numeric.decision_tree(value=np.unique(y)[0])
            
            min_samples = max(10, int(0.05 * len(y)))
            if len(y) < min_samples:    # If less than minimum samples, then stop processing
                values, counts = np.unique(y, return_counts=True)

                return numeric.decision_tree(value=values[np.argmax(counts)])
            
            # Find best split
            best_feature, best_threshold, best_gain = numeric.decision_tree.all_best_split(X, y)

            if best_gain == 0:
                values, counts = np.unique(y, return_counts=True)
                return numeric.decision_tree(value=values[np.argmax(counts)])
            
            # Split the dataset
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold

            left_subtree = numeric.decision_tree.build_tree(X[left_mask], y[left_mask], depth + 1)
            right_subtree = numeric.decision_tree.build_tree(X[right_mask], y[right_mask], depth + 1)

            return numeric.decision_tree(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
        
        def predict(self, X):
            '''
            Predict the class labels for the given documents.
            X: 2D array or DataFrame containing the characteristics of each sample
            '''
            # When aproach to leaf node
            if self.value is not None:
                return self.value
            
            if X[self.feature] <= self.threshold:
                return self.left.predict(X)
            else:
                return self.right.predict(X)
    
    class RandomForest:
        def __init__(self, n_trees=None, max_depth=None, min_samples_split=2):
            self.n_trees = n_trees
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.trees = []
        
        @staticmethod
        def optimize_n_trees_depth(X, y):
            '''
            Use a validation set to explore the optimal number of trees and max depth
            '''
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            best_n_trees = 10
            best_max_depth = 3
            best_accuracy = 0

            for n_trees in range(10, 101, 10):
                for max_depth in range(3, 21, 2):
                    forest = numeric.RandomForest(n_trees=n_trees, max_depth=max_depth)
                    forest.fit(X_train, y_train)
                    predictions = forest.predict(X_val)
                    accuracy = accuracy_score(y_val, predictions)

                    print(f"Trees: {n_trees}, Max Depth: {max_depth}, Accuracy: {accuracy}")

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_n_trees = n_trees
                        best_max_depth = max_depth
            
            print(f"Optimal number of trees: {best_n_trees}, Optimal max depth: {best_max_depth}, Best Accuracy: {best_accuracy}")

            return best_n_trees, best_max_depth
        
        def fit(self, X, y):
            if self.n_trees is None or self.max_depth is None:
                self.n_trees, self.max_depth = self.optimize_n_trees_depth(X, y)
            
            self.trees = []

            # Bootstrap smapling
            for _ in range(self.n_trees):
                X_sample, y_sample = resample(X, y)
                if isinstance(X_sample, pd.DataFrame):
                    feature_indices = np.random.choice(range(X.shape[1]), size=int(np.sqrt(X.shape[1])), replace=False)
                    X_sample = X_sample.iloc[:, feature_indices]
                else:
                    feature_indices = np.random.choice(range(X.shape[1]), size=int(np.sqrt(X.shape[1])), replace=False)
                    X_sample = X_sample[:, feature_indices]

                tree = numeric.decision_tree()
                tree = tree.build_tree(X_sample, y_sample, depth=0)
                self.trees.append((tree, feature_indices))
        
        def predict(self, X):
            predictions = []
            for _, row in X.iterrows():
                tree_preds = []
                for tree, feature_indices in self.trees:
                    sample_features = row.iloc[feature_indices]
                    tree_preds.append(tree.predict(sample_features))
                
                predictions.append(max(set(tree_preds), key=tree_preds.count))
            
            return predictions



# Models for the language dataset
class LLM_models:
    # Preprocess functions for the language model
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)   # Split by word
        filtered_text = [word for word in word_tokens if word not in stop_words]    # Remove stop words
        
        processed_text = ' '.join(filtered_text)    # Return words by combining them again
        return processed_text
    
    # Bag of words function
    def bow(data):
        # Extract all words as a set
        vocabulary = set()

        for doc in data:
            words = doc.split() # Split the word with the space
            vocabulary.update(words)    # Add the word to the vocabulary

        # Create a word frequency vector for each document
        freq_vector = []
        for doc in data:
            words = doc.split()
            doc_vector = [words.count(word) for word in vocabulary]
            freq_vector.append(doc_vector)

        return np.array(freq_vector), list(vocabulary)
    
    class NaiveBayes:
        def __init__(self):
            self.class_counts = {}
            self.word_counts = {}
            self.total_documents = 0
            self.vocabulary = set()

        def train(self, x, y):
            '''
            Train Naive Bayes model.
            x: List of documents (Text)
            y: List of corresponding labels (class)
            '''
            self.total_documents = len(x)

            bow_vectors, self.vocabulary = LLM_models.bow(x)

            for doc, label, vector in zip(x, y, bow_vectors):
                self.class_counts[label] += 1
                for word, count in vector.items():
                    self.word_counts[label][word] += count
            # self.total_documents = len(x)
            # for doc, label in zip(x, y):
            #     if label not in self.word_counts:
            #         self.word_counts[label] = []
            #     self.class_counts[label] = self.class_counts.get(label, 0) + 1
            #     for word in doc.split():
            #         self.word_counts[(word, label)] = self.word_counts.get((word, label), 0) + 1
            #         self.vocabulary.add(word)
        
        def predict(self, X):
            '''
            Predict the class labels for the given documents.
            X: List of documents (Text)
            '''
            all_predicted_labels = []

            bow_vectors, _ = LLM_models.bow(X)

            for x, vector in zip(X, bow_vectors):
                scores = {}
                for label in self.class_counts.keys():
                    # Log probability
                    score = math.log(self.class_counts[label] / self.total_documents)

                    # Conditional probability for each word, the probability that the word is in that class
                    for word, count in vector.items():
                        word_count = self.word_counts[label].get(word, 0) + 1   # add-1 smoothing
                        score += math.log(word_count / (sum(self.word_counts[label].values()) + len(self.vocabulary)))

                    scores[label] = score
                
                # Calculate probability of each class
                probability = {label: math.exp(score) for label, score in scores.items()}

                # After calculating total probability, calculate the percentage about each class
                total_probability = sum(probability.values())
                percentage = {label: (prob / total_probability) * 100 for label, prob in probability.items()}

                # Predict the label with highest probability
                predicted_label = max(percentage, key=percentage.get)
                all_predicted_labels.append(predicted_label)
            
            return all_predicted_labels
            # all_predicted_labels = []
            # all_probabilities = []
        
            # for x in X:
            #     scores = {}
            #     for label in self.class_counts.keys():
            #         score = math.log(self.class_counts[label] / self.total_documents)
            #         for word in x.split():
            #             count = self.word_counts.get((word, label), 0) + 1
            #             score += math.log(count / (self.class_counts[label] + len(self.vocabulary)))
            #         scores[label] = score
        
            #     # Calculate the probability of each label
            #     probability = {label: math.exp(scores[label]) for label in scores}

            #     # Convert to probability to percentage
            #     total_probability = sum(probability.values())
            #     percentage = {label: (prob / total_probability) * 100 for label, prob in probability.items()}
        
            #     # Find highest probability of each label
            #     predicted_label = max(percentage, key=percentage.get)
            #     all_predicted_labels.append(predicted_label)
            #     all_probabilities.append(percentage)
        
            # return all_predicted_labels, all_probabilities

# Model evaluation functions
class evaluation:
    def evaluate_classifier(TP, TN, FP, FN):
        sensitivity = 0 if (TP + FN) == 0 else TP / (TP + FN)
        specificity = 0 if (TN + FP) == 0 else TN / (TN + FP)
        precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
        npv = 0 if (TN + FN) == 0 else TN / (TN + FN)
        accuracy = 0 if (TP + TN + FP + FN) == 0 else (TP + TN) / (TP + TN + FP + FN)
        f_score = 0 if (precision + sensitivity) == 0 else 2 * precision * sensitivity / (precision + sensitivity)
        return sensitivity, specificity, precision, npv, accuracy, f_score

    def calculate_metrics(test_labels, pred_labels, label):
        metrics = {l: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for l in label}
        for true_label, predicted_label in zip(test_labels, pred_labels):
            if isinstance(true_label, np.ndarray):
                true_label = true_label.item() if true_label.size == 1 else tuple(true_label)

            if true_label in label:
                if predicted_label == true_label:
                    metrics[true_label]['tp'] += 1
                else:
                    metrics[true_label]['fn'] += 1
                for l in label:
                    if l != true_label:
                        if predicted_label == l:
                            metrics[l]['fp'] += 1
                        else:
                            metrics[l]['tn'] += 1
                    
        return metrics

    def macro_average(metrics):
        precision_sum = 0
        sensitivity_sum = 0
        specificity_sum = 0
        f1_score_sum = 0

        for class_label, metric in metrics.items():
            TN = metric['tn']
            TP = metric['tp']
            FP = metric['fp']
            FN = metric['fn']
            precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
            sensitivity = 0 if (TP + FN) == 0 else TP / (TP + FN)
            specificity = 0 if (TN + FP) == 0 else TN / (TN + FP)
            f1_score = 0 if (precision + sensitivity) == 0 else 2 * precision * sensitivity / (precision + sensitivity)
            precision_sum += precision
            sensitivity_sum += sensitivity
            specificity_sum += specificity
            f1_score_sum += f1_score

        
        num_classes = len(metrics)
        macro_precision = precision_sum / num_classes
        macro_sensitivity = sensitivity_sum / num_classes
        macro_specificity = specificity_sum / num_classes
        macro_f1_score = f1_score_sum / num_classes
        return macro_precision, macro_sensitivity, macro_specificity, macro_f1_score

    def micro_average(metrics):
        sum_TN = sum([metric['tn'] for metric in metrics.values()])
        sum_TP = sum([metric['tp'] for metric in metrics.values()])
        sum_FP = sum([metric['fp'] for metric in metrics.values()])
        sum_FN = sum([metric['fn'] for metric in metrics.values()])
        
        micro_precision = 0 if (sum_TP + sum_FP) == 0 else sum_TP / (sum_TP + sum_FP)
        micro_sensitivity = 0 if (sum_TP + sum_FN) == 0 else sum_TP / (sum_TP + sum_FN)
        micro_specificity = 0 if (sum_TN + sum_FP) == 0 else sum_TN / (sum_TN + sum_FP)
        micro_f1_score = 0 if (micro_precision + micro_sensitivity) == 0 else 2 * micro_precision * micro_sensitivity / (micro_precision + micro_sensitivity)

        return micro_precision, micro_sensitivity, micro_specificity, micro_f1_score
