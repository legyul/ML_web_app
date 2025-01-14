import pandas as pd
import numpy as np
import math
from collections import defaultdict
import statistics
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV

class preprocess:
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
    
    def find_target_column(data):
        '''
        Automatically identify the target column in a dataset

        Parameters
        - data: pandas DataFrame

        Returns:
        - target_column: Name of the identified target column (or None if not found)
        '''
        # Column name analysis
        target_keywords = ['target', 'label', 'class', 'output', 'result', 'y']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                print(f"Target column identified by name: {col}")
                return col
        
        # Data type analysis
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].nunique() < 10:     # Categorical column
                print(f"Target column identified by data type or unique values: {col}")
                return col
        
        # Fall back
        print("No clear target column found. Please specify manually.")
        return None
    
    def map_target(target_column):
        '''
        Automatically map a target column to integers if it is not numeric

        Parameters
        - target_column: list, Series, or array-like containing target labels

        Returns:
        mapped_target: List of integer_mapped target values
        label_mapping: Dictionary mapping original labels to integers
        '''
        if isinstance(target_column, pd.DataFrame):
            if target_column.shape[1] > 1:
                raise ValueError("Target column must be a single column, but multiple columns were provided.")
            target_column = target_column.iloc[:, 0]

        # Convert to pandas Series for consistent processing
        target_column = pd.Series(target_column)

        # Check if the target column is numeric
        if pd.api.types.is_numeric_dtype(target_column):
            return target_column.tolist(), None     # If numeric, return the original target column without mapping
        
        # If not numeric, generate mapping
        unique_labels = sorted(target_column.unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        # Map the target column to integers
        mapped_target = target_column.map(label_mapping).tolist()

        return mapped_target, label_mapping

    def reverse_map(mapped, label_mapping):
        '''
        Convert integer-mapped target values back to their original labels.

        Parameters
        - mapped: List of integer-mapped target values
        - label_mapping: Dictionary mapping integers back to original labels

        Returns
        - List of original labels
        '''
        if label_mapping is None:
            raise ValueError("Reverse mapping is required to map back to original labels.")
        
        label_mapping = {v: k for k, v in label_mapping.items()}
        
        original_labels = [label_mapping[value] for value in mapped]
        return original_labels

# ============================================== Numeric ===========================================================
# Models for the numeric dataset
class numeric:
    # ------------------------------------------ Naive Bayes ---------------------------------------------------
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
    
    # ------------------------------------------ Decision Tree ---------------------------------------------------
    
    class decision_tree:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, mode='classification'):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.mode = mode

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
            '''
            Find the best split for a single feature in classification

            Parameters
            - feature: 1D array of feature values
            - y: Array-like target values

            Returns
            - best_split: The threshold value for the best split
            - best_gain: Information Gain due to the split
            '''
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
        
        def indi_best_split_regression(feature, y):
            '''
            Find the best split for a single feature in regression

            Parameters
            - feature: 1D array of feature values
            - y: Array-like target values

            Returns
            - best_split: The threshold value for the best split
            - best_gain: Reduction in MSE due to the split
            '''
            sorted_indices = np.argsort(feature)
            feature = feature[sorted_indices]
            y = y[sorted_indices]

            best_split = None
            best_gain = -float('inf')
            total_variance = np.var(y) * len(y)     # Total variance before split

            for i in range(1, len(y)):
                left_y = y[:i]
                right_y = y[i:]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                # Calculate variance for each split
                left_var = np.var(left_y) * len(left_y)
                right_var = np.var(right_y) * len(right_y)
                mse_reduction = total_variance - (left_var + right_var)

                # Update best split
                if mse_reduction > best_gain:
                    best_gain = mse_reduction
                    best_split = (feature[i-1] + feature[i]) / 2
            
            return best_split, best_gain

        def all_best_split(X, y, mode='classification'):
            '''
            Find the best split across all features.

            Parameters
            - X: DataFrame containing features
            - y: Array-like target values
            - mode: 'classification' or 'regression'

            Returns
            - best_feature: The feature providing the best split
            - best_split_value: The threshold value for the best split
            - best_gain: The gain (IG for classification, MSE reduction for regression)
            '''
            best_feature = None
            best_split_value = None
            best_gain = -float('inf')

            for col in X.columns:
                if mode == 'regression':
                    split, gain = numeric.decision_tree.indi_best_split_regression(X[col].values, y)
                else:
                #print(f"\nChecking feature: {col}")
                    split, gain = numeric.decision_tree.indi_best_split(X[col].values, y)
                #print(f"- Split Value: {split}, Information Gain: {gain:.4f}")
            
            if gain > best_gain:
                best_gain = gain
                best_split_value = split
                best_feature = col
            
            return best_feature, best_split_value, best_gain
        
        @staticmethod
        def build_tree(X, y, depth=0, mode='classification'):
            '''
            Train Decision tree model.
            X: 2D array or DataFrame containing the characteristics of each sample
            y: List of corresponding labels (class)
            mode: 'classification' or 'regression'
            '''
            # Condition of stopping processing
            if len(np.unique(y)) == 1:  # If there is same label
                if mode == 'regression':
                    return numeric.decision_tree(value=np.mean(y))  # Return mean value
                else:
                    return numeric.decision_tree(value=np.unique(y)[0])
            
            min_samples = max(10, int(0.05 * len(y)))
            if len(y) < min_samples:    # If less than minimum samples, then stop processing
                if mode == 'regression':
                    return numeric.decision_tree(value=np.mean(y))
                else:
                    values, counts = np.unique(y, return_counts=True)
                    return numeric.decision_tree(value=values[np.argmax(counts)])
            
            # Find best split
            best_feature, best_threshold, best_gain = numeric.decision_tree.all_best_split(X, y, mode=mode)

            if best_gain == 0:
                if mode == 'regression':
                    return numeric.decision_tree(value=np.mean(y))
                else:
                    values, counts = np.unique(y, return_counts=True)
                    return numeric.decision_tree(value=values[np.argmax(counts)])
            
            # Split the dataset
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold

            left_subtree = numeric.decision_tree.build_tree(X[left_mask], y[left_mask], depth + 1, mode=mode)
            right_subtree = numeric.decision_tree.build_tree(X[right_mask], y[right_mask], depth + 1, mode=mode)

            return numeric.decision_tree(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
        
        def predict(self, X):
            '''
            Predict the class labels for the given documents.
            X: 2D array or DataFrame containing the characteristics of each sample
            '''
            # When aproach to leaf node
            if self.value is not None:
                return self.value
            
            if self.mode == 'regression':
                if X[self.feature] <= self.threshold:
                    return self.left.predict(X)
                else:
                    return self.right.predict(X)
            else:
                if X[self.feature] <= self.threshold:
                    return self.left.predict(X)
                else:
                    return self.right.predict(X)
    
    # ------------------------------------------ Random Forest ---------------------------------------------------
    class RandomForest:
        def __init__(self, n_trees=None, max_depth=None, min_samples_split=2, mode='classification'):
            self.n_trees = n_trees
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.mode = mode
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
                print(f"tree_preds: {tree_preds}")
                if self.mode == 'regression':
                    mean_pred = np.mean(tree_preds)
                    rounded_pred = round(mean_pred)
                    predictions.append(np.mean(tree_preds))
                else:
                    predictions.append(max(set(tree_preds), key=tree_preds.count))
            
            return predictions
    
    # ---------------------------------------- Logistic Regression -------------------------------------------------
    class LogisticRegression:
        def __init__(self, learning_rate=0.001, max_epochs=1000, L2=0.01):
            '''
            Initialize the logistic regression model with parameters

            Parameters:
            - learning_rate (float): Step size for gradient descent (Default is 0.01)
            - max_epochs (int): Maximum number of epochs for training (Default is 500)
            '''
            self.learning_rate = learning_rate
            self.max_epochs = max_epochs
            self.L2 = L2
            self.num_class = 2
            self.w = None
            self.b = None

        @staticmethod
        def sigmoid(z):
            '''
            Sigmoid activation function.

            Parameters
            - z (numpy array): Input value or array to apply the sigmoid function to

            Returns
            - numpy array: Output of the sigmoid function applied element-wise to the input
            '''
            return 1 / (1 + np.exp(-z))
        
        @staticmethod
        def softmax(z):
            '''
            Softmax activation function

            Parameters
            - x (numpy array): Input array where each row represents a set of raw class scores (logits)
                for each sample in the dataset

            Returns
            - numpy array: A 2D array where each row represents the probability distribution over all classes for the corresponding sample
                with values between 0 and 1. The probabilities in each row sum to 1.
            ''' 
            # If z is 1D, calculate the max along axis 0; otherwise, use axis 1
            if len(z.shape) == 1:
                z = z.reshape(1, -1)  # Make it 2D with a single row if it's 1D
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
        def loss_computing(self, pred_probs, y, L2, class_weight=None):
            '''
            Compute the logistic regression loss with L2 regularization

            Parameters
            - pred_probs (numpy array): Predicted probabilities from softmax or sigmoid
            - y (numpy array): True labels (1D)
            - L2 (float): L2 regularization strength
            
            Returns
            - float: The calculated loss value
            '''
            epsilon = 1e-15

            num_samples, self.num_class = pred_probs.shape
            y_one_hot = np.zeros((num_samples, self.num_class))

            # Ensure y values are valid indices
            if np.max(y) >= self.num_class:
                raise ValueError(f"Invalid class label in y. Max class index({np.max(y)}) exceeds num_class ({self.num_class}).")

            y_one_hot[np.arange(num_samples), y] = 1
            
            # Compute cross-entropy loss
            clipped_probs = np.clip(pred_probs, epsilon, 1 - epsilon)
            if len(pred_probs.shape) > 1:   # Multi-class
                cross_entropy = -np.sum(y_one_hot * np.log(clipped_probs), axis=1)  # Per-sample loss
            else:   # Binary classification
                cross_entropy = - (y * np.log(clipped_probs) + (1 - y) * np.log(1 - clipped_probs))     # Per-sample loss
            
            # Apply class weights if provided
            if class_weight is not None:
                # Generate sample weights based on class weights
                sample_weights = np.array([class_weight[cls] for cls in y])
                weighted_loss = np.mean(sample_weights * cross_entropy)     # Weighted mean loss
            else:
                weighted_loss = np.mean(cross_entropy)      # Standard mean loss
            
            # Add L2 regularization term
            reg_loss = L2 * np.sum(self.w ** 2)
            total_loss = weighted_loss + reg_loss

            return total_loss
        
        def fit(self, X, y, patience=100, k=5, class_weight=None):
            '''
            Train a logistic regression model using gradient descent and early stopping.

            Parameters
            - X (DataFrame): Feature matrix with shape (num_samples, num_features)
            - y (Series): Labels with shape (num_samples)
            - patience (int): Number of epochs to wait for improvement before early stopping (default = 100)
            - k (int): Number of splits for k-fold cross-validation (default = 5)
            - class_weight (dict or None): Weights for balancing classes in loss computation (default = None)

            Returns
            - tuple
                - w (numpy array): Trained weights
                - b (float): Trained bias
                - train_losses (list): List of training losses for each epoch
                - val_losses (list): List of validation losses for each epoch
            '''
            # Convert DataFrame to Numpy array for processing
            X = np.array(X)
            y = np.array(y).flatten()
            self.num_class = len(np.unique(y))
            y = y - y.min()

            # Initialize parameters
            m, n = X.shape
            self.w = np.zeros((n, self.num_class))
            self.b = np.zeros(self.num_class)
            train_losses = []
            val_losses = []

            # Set up k-fold cross-validation
            kf = KFold(n_splits=k, shuffle=True)

            # Early stopping initialization
            best_val_loss = float('inf')                    # Set the initial best validation loss to infinity 
            epochs_wout_improvement = 0                     # Counter for patience
            best_w, best_b = self.w, self.b                 # Best weights and bias to restore after early stopping

            # Training loop over epochs
            for epoch in range(self.max_epochs):
                epoch_train_loss = 0
                epoch_val_loss = 0

                # Perform k-fold cross-validation
                for train_idx, val_idx in kf.split(X):
                    train_X, val_X = X[train_idx], X[val_idx]
                    train_y, val_y = y[train_idx], y[val_idx]

                    # Forward pass: Compute prediction
                    train_z = np.dot(train_X, self.w) + self.b
                    pred_probs_train = (self.sigmoid(train_z) if self.num_class == 2 else self.softmax(train_z))

                    val_z = np.dot(val_X, self.w) + self.b
                    pred_probs_val = (self.sigmoid(val_z) if self.num_class == 2 else self.softmax(val_z))

                    # Compute losses
                    train_loss = self.loss_computing(pred_probs_train, train_y, self.L2, class_weight)
                    val_loss = self.loss_computing(pred_probs_val, val_y, self.L2, class_weight)

                    epoch_train_loss += train_loss
                    epoch_val_loss += val_loss

                    # One-hot encode for multi-class gradient computation
                    if self.num_class > 2:
                        train_y_one_hot = np.zeros((len(train_y), self.num_class))
                        train_y_one_hot[np.arange(len(train_y)), train_y] = 1
                    
                    else:       # Binary classification
                        train_y_one_hot = train_y.reshape(-1, 1)
                    
                    # Compute gradients
                    dw = np.dot(train_X.T, (pred_probs_train - train_y_one_hot)) / len(train_y) + self.L2 * self.w
                    db = np.sum(pred_probs_train - train_y_one_hot, axis=0) / len(train_y)

                    # Update weights and bias using gradient descent
                    self.w -= self.learning_rate * dw
                    self.b -= self.learning_rate * db
                
                # Calculate average loss for this epoch
                train_losses.append(epoch_train_loss / k)
                val_losses.append(epoch_val_loss / k)

                if val_losses[-1] < best_val_loss:
                    best_val_loss = val_losses[-1]
                    best_w, best_b = self.w.copy(), self.b
                    epochs_wout_improvement = 0     # Reset patience counter
                
                else:
                    epochs_wout_improvement += 1
                
                # Check patience
                if epochs_wout_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch} with validation loss: {epoch_val_loss:.4f}")
                    break       # Stop training if no improvement in 'patience' epochs

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Training loss: {train_losses[-1]:.4f}, Validation loss: {val_losses[-1]:.4f}")
            
            return best_w, best_b, train_losses, val_losses
        
        def predict(self, X):
            '''
            Predict class labels for input data using the trained logistic regression model

            Parameters
            - X (numpy array or DataFrame): Input feature matrix (2D)
            - w (numpy array): Model weights (1D)
            - b (float): model bias (scalar)

            Returns
            - numpy array: Predicted class labels (1D)
            '''
            if self.w is None or self.b is None:
                raise ValueError("Model not trained yet. Call fit() before predict.")
            
            # Compute the raw class scores (logits)
            z = np.dot(X, self.w) + self.b

            # Check if it is binary or multi-class
            if self.num_class == 2:     # Binary classification
                # Apply sigmoid to get probability
                pred_probs = self.sigmoid(z)

                # Predictions are class 1 if probability >= 0.5, else class 0
                predictions = (pred_probs >= 0.5).astype(int)

            else:       # Multi-class classification
                # Apply softmax to get class probabilities
                soft_z = self.softmax(z)

                # Predictions are the class with the highest probability
                predictions = np.argmax(soft_z, axis=1)

            return predictions
        
        def predict_proba(self, X):
            '''
            Predict class probabilities for input data using the trained logistic regression model

            Parameters:
            - X (numpy array or DataFrame): Input feature maxtrix (2D)

            Returns:
            - numpy array: Predicted class probabilities (2D)
            '''
            z = np.dot(X, self.w) + self.b

            if self.num_class == 2:     # Binary classification
                pred_probs = self.sigmoid(z)
                return np.column_stack((1 - pred_probs, pred_probs))        # Return both class probabilities
            
            else:       # Multi-class classification
                soft_z = self.softmax(z)
                return soft_z       # Return probabilities for each class
        
        def get_params(self, deep=True):
            '''
            Return hyperparameters of the model
            '''
            return {'learning_rate': self.learning_rate,
                    'max_epochs': self.max_epochs,
                    'L2': self.L2}
        
        def set_params(self, **params):
            '''
            Set hyperparameters of the model
            '''
            for param, value in params.items():
                setattr(self, param, value)
            
            return self

# ========================================= Language ==============================================================
# Models for the language dataset
class LLM_models:

    # Preprocess functions for the language model
    def LM_preprocess(text):
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

# ============================================= Tuning ===========================================================
class tuning:
    def tuen_hyperparameters(model, param_dist, train_X, train_y, test_X, test_y, n_iter=100, cv=5, scoring=None, random_state=42):
        '''
        Optimize the hyperparameters of a given model using RandomizedSearchCV and evaluate its performance

        Parameters
        - model: The model to optimize (LogisticRegression, RandomForestClassifier, etc)
        - param_dist (dictionary): The search space for hyperparameters
        - train_X (features): Training data
        - train_y (labels): Training data
        - test_X (features): Test data
        - test_y (labels): Test data
        - n_iter: Number of hyperparameter combination to try
        - cv: Number of cross-validation folds
        - scoring: Performance evaluation metrics (default is None, but multiple metrics can be used)
        - random_state: Random seed (default is 42)

        Returns:
        - Best hyperparameters, model performance metrics
        '''   
        # Using RandomizedSearchCV to optimize hyperparameters
        random_search = RandomizedSearchCV(estimator=model,
                                        param_distributions=param_dist,
                                        n_iter=n_iter,
                                        cv=cv,
                                        scoring=scoring,
                                        n_jobs=-1,
                                        random_state=random_state,
                                        refit='accuracy')    # Choose the best model based on 'accurary'
        
        # Fit the model
        random_search.fit(train_X, train_y)

        # Output the best hyperparameters
        best_params = random_search.best_params_
        print(f"Best parameters: {best_params}")

        # Predict using the best model
        y_pred = random_search.best_estimator_.predict(test_X)
        y_pred_proba = random_search.best_estimator_.predict_proba(test_X)

        if y_pred_proba.ndim == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)

        test_y = np.array(test_y)

        # Evaluate using different metrics
        accuracy = accuracy_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred, average='weighted')
        roc_auc = roc_auc_score(test_y, y_pred_proba, multi_class='ovr')

        print(f"Best Parameters: {best_params}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score (Weighted): {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Best Model: {random_search.best_estimator_}")

        return best_params, accuracy, f1, roc_auc, random_search.best_estimator_


# ============================================== Evaluation ================================================
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
