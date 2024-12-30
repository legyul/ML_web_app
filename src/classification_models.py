import pandas as pd
import math
from collections import defaultdict
import statistics
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Models for the numeric dataset
class numeric:
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

# Preprocess functions for the language model
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    processed_text = ' '.join(filtered_text)
    return processed_text

# Models for the language dataset
class LLM_models:
    class NaiveBayes:
        def __init__(self):
            self.class_counts = {}
            self.word_counts = {}
            self.total_documents = 0
            self.vocabulary = set()

        def train(self, x, y):
            self.total_documents = len(x)
            for doc, label in zip(x, y):
                if label not in self.word_counts:
                    self.word_counts[label] = []
                self.class_counts[label] = self.class_counts.get(label, 0) + 1
                for word in doc.split():
                    self.word_counts[(word, label)] = self.word_counts.get((word, label), 0) + 1
                    self.vocabulary.add(word)
        
        def predict(self, X):
            all_predicted_labels = []
            all_probabilities = []
        
            for x in X:
                scores = {}
                for label in self.class_counts.keys():
                    score = math.log(self.class_counts[label] / self.total_documents)
                    for word in x.split():
                        count = self.word_counts.get((word, label), 0) + 1
                        score += math.log(count / (self.class_counts[label] + len(self.vocabulary)))
                    scores[label] = score
        
                # Calculate the probability of each label
                probability = {label: math.exp(scores[label]) for label in scores}

                # Convert to probability to percentage
                total_probability = sum(probability.values())
                percentage = {label: (prob / total_probability) * 100 for label, prob in probability.items()}
        
                # Find highest probability of each label
                predicted_label = max(percentage, key=percentage.get)
                all_predicted_labels.append(predicted_label)
                all_probabilities.append(percentage)
        
            return all_predicted_labels, all_probabilities

# Model evaluation functions
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
