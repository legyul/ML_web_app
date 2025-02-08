import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
import re
from .common import setup_global_logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from joblib import Parallel, delayed
import gc
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
from logger_utils import logger


class preprocess:
    def column_types(data):
        numeric_columns = data.select_dtypes(include='number').columns.tolist()
        categorical_columns = data.select_dtypes(exclude='number').columns.tolist()
        
        return numeric_columns, categorical_columns
    
    def check_target_type(y):
        _, categofical_columns = preprocess.column_types(pd.DataFrame(y))
        return 'categorical' if len(categofical_columns) > 0 else 'numerical'

    def numeric_column_statistics(data, numeric_columns):
        for col in numeric_columns:
            print(f"- {col}: mean = {data[col].mean():.2f}, min = {data[col].min()}, max = {data[col].max()}")
    
    def is_text_column(column_data):
        '''
        Detects if a column primarily contains text or natural language

        Parameters
        - column_data (Series): one column of the dataset

        Return
        - bool: True if the column is likely a language based column, else False
        '''
        column = column_data.fillna('').astype(str)
        avg_word_count = column.apply(lambda x: len(x.split())).mean()

        return avg_word_count > 2
        
    def is_continuous_data(X, y):
        '''
        Automatically determines if the data is continuous

        Parameters
        - X (numpy array or DataFrame): Input feature
        - y (Series or DataFrame): Target class

        Returns
        - bool: True if the dataset can be used with a regression model, False if classification is needed
        '''
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0].values

        # Check if the target (y) is continuous
        unique_values = len(np.unique(y))
        return unique_values > 1 and np.issubdtype(y.dtype, np.number)

    def detect_id_columns(data):
        '''
        Automatically detect ID columns based on high cardinality (unique values)
        and column names indicating potential ID columns (e.g., "id", "user", "code", "customer").

        Parameters:
        - data (DataFrame): The input DataFrame to analyze.

        Returns:
        - list: A list of column names identified as ID columns.
        '''
        id_columns = []

        for col in data.columns:
            if data[col].dtype == 'object':
                if data[col].str.len().mean() > 100:
                    continue

            # Check if the column is numeric
            if data[col].dtype in ['int64', 'float64']:
                # Check if it has many unique values (likely to be an ID column)
                if data[col].nunique() > len(data) * 0.8:  # If unique values > 80% of the dataset length
                    id_columns.append(col)

            # For object type columns, we can check if they have unique values
            elif data[col].dtype == 'object' and data[col].nunique() > len(data) * 0.8:
                if not data[col].str.contains(r'http|www|#|@').any():
                    id_columns.append(col)

        return id_columns

    def detect_text_data(data, target_column=None):
        '''
        Determines if a dataset is primarily a text dataset, excluding the target column

        Parameters
        - data: pandas DataFrame
        - target_column: Name of the target column (if known, else None)

        Returns
        - is_text_data: True if the dataset is primarily text based
        - text_columns: List of columns identified as text based
        '''
        # Exclude target column if provided
        data_to_check = data.drop(columns=[target_column], errors='ignore')

        # Check each column for text based characteristics
        text_columns = [col for col in data_to_check.columns if preprocess.is_text_column(data_to_check[col])]

        # Determine if most of the dataset is text based
        is_text_data = len(text_columns) > 0 and (len(text_columns) / len(data_to_check.columns) > 0.6)
        return is_text_data, text_columns
    
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
            if col.lower() in target_keywords:
                logger.debug(f"Target column identified by name: {col}")
                return col
        
        # Check for categorical columns and return it
        numeric_columns, categorical_columns = preprocess.column_types(data)
        
        if categorical_columns:
            logger.debug(f"Target column identified as categorical: {categorical_columns}")
            return categorical_columns

        # Check for categorical-like columns
        categori_candidate = []
        for col in data.columns:
            unique_values = data[col].nunique()
            total_values = len(data[col])

            if unique_values < total_values * 0.05:
                categori_candidate.append(col)
        
        # If there is only one categorical candidate, assume it as target
        if len(categori_candidate) == 1:
            logger.debug(f"Target column identified by low cardinality: {categori_candidate[0]}")
            return categori_candidate[0]
        
        # Check for columns with imbalanced class distributions
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].nunique() < 10:
                value_counts = data[col].value_counts(normalize=True)
                
                if value_counts.max() > 0.5:
                    logger.debug(f"Target column identified by imbalanced distribution: {col}")
                    return col
        
        # Check for text-based target columns (sentiment or other categories)
        for col in data.columns:
            # If the column contains string labels like "Positive", "Neutral", "Negative", etc.
            value_counts = data[col].value_counts(normalize=True)
            if any(val in ['Positive', 'Neutral', 'Negative', 'Extremely Positive', 'Extremely Negative'] for val in value_counts.index):
                logger.debug(f"Target column identified as sentiment or categorical: {col}")
                return col
                
        # Use is_text_column function to detect text_based target columns
        text_candidates = [col for col in data.columns if preprocess.is_text_column(data[col])]

        if len(text_candidates) == 1:
            logger.debug(f"Target column identified as text-based: {text_candidates[0]}")
            return text_candidates[0]

        logger.error("No clear target column found.")
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
        # If target_column is a list or numpy array, convert to pandas Series for consistent processing
        if isinstance(target_column, (list, np.ndarray)):
            target_column = pd.Series(target_column)
        
        # If it's a DataFrame with more than one column, raise an error
        if isinstance(target_column, pd.DataFrame):
            if target_column.shape[1] > 1:
                error_message = "Target column must be a single column, but multiple columns were provided."
                logger.error(error_message)
                raise ValueError(error_message)
            target_column = target_column.iloc[:, 0]

        # Check if the target column is numeric
        if pd.api.types.is_numeric_dtype(target_column):
            return target_column.tolist(), None     # If numeric, return the original target column without mapping
        
        # If not numeric, generate mapping
        unique_labels = sorted(target_column.unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        # Map the target column to integers
        mapped_target = target_column.map(label_mapping)

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
            error_message = "Reverse mapping is required to map back to original labels."
            logger.error(error_message)
            raise ValueError(error_message)
        
        label_mapping = {v: k for k, v in label_mapping.items()}
        
        original_labels = [label_mapping[value] for value in mapped]
        return original_labels
    
    ## Need to be fix to work with text dataset
    def preprocess_text_columns(data, top_k_features=100):
        '''
        Detect text data and preprocess the detected columns using LM_preprocess and TF-IDF

        Parameters
        - data (DataFrame): Input dataset

        Returns
        - processed_data (DataFrame): DataFrame with language columns preprocessed and vertorized
        - target_column (str): Target column name
        - language_columns (list): List of language columns names
        - bow_vocab (list): Vocabulary generated by the Bag of Words process
        '''
        data = data.copy()
        data.dropna(how='any')

        # Automatically detect ID columns and exclude them
        id_columns = preprocess.detect_id_columns(data)
        logger.info(f"Detected ID columns: {id_columns}")

        # Find the target columns
        target_column = preprocess.find_target_column(data)
        logger.info(f"Target column detected: {target_column}")

        # Detect text columns
        is_text_data, text_columns = preprocess.detect_text_data(data, target_column=target_column)

        if not is_text_data:
            logger.info("No text data detected.")
            return data, target_column, None, []
        
        logger.info(f"Detected text columns: {text_columns}")

        # Exclude ID columns and target column from the text columns
        text_columns = [col for col in text_columns if col not in id_columns and col != target_column]

        # If text_columns is empty after filtering, print a message
        if not text_columns:
            logger.warning("No valid text columns remaining after filtering ID and target columns.")
            return data, target_column, [], []
        
        gc.collect()

        vectorizer = Text.TextVectorizer()
        bow_vocab = set()

        for col in text_columns:
            logger.info(f"[INFO] Preprocessing and vectorizing column: {col}")

            # Text preprocess
            data[col] = data[col].astype(str).apply(Text.preprocess)

            # Generate vocabulary
            vectorizer.fit(data[col].tolist())

            # Transform BoW
            bow_matrix = vectorizer.transform(data[col].tolist())

            # Transform TF-IDF
            tfidf_matrix = vectorizer.compute_tfidf(bow_matrix)

            # Feature selection: Select top K features based on chi-squared test
            # Select K important features for each text entry (using word importance from TF-IDF)
            feature_selector = SelectKBest(chi2, k=top_k_features)
            selected_features = feature_selector.fit_transform(tfidf_matrix, data[target_column])

            # Apply the result to DataFrame
            data[col] = selected_features

            # Update vocabulary
            bow_vocab.update(vectorizer.vocabulary.keys())

            gc.collect()
        
        logger.info(f"[INFO] Preprocesing complete. Vocabulary size: {len(bow_vocab)}")

        return data, target_column, text_columns, bow_vocab

    
# ============================================== Numeric ===========================================================
# Models (Naive Bayes, Decision Tree, Random Forest, Logistic Regression) for the numeric dataset
class numeric:
    # ------------------------------------------ Naive Bayes ---------------------------------------------------
    # Gausian Naive Bayes model
    class gausian_NaiveBayes:
        def __init__(self):
            '''
            Initialize the Gaussian Navie Bayes model

            Attributes
            - class_stats (dict): Stores the mean and standard deviation of each feature for each class
            - class_freq (dict): Stores the frequency of each class in the training data
            - classes (set): Unique set of classes in the dataset
            '''
            self.class_stats = defaultdict(dict)
            self.class_freq = defaultdict(int)
            self.classes = set()
            self.num_class = 2
        
        def fit(self, X, y):
            '''
            Train the model by calculating mean and standard deviation for each feature of each class

            Parameters
            - X (numpy array or DataFrame): Feature matrix of shape (num_samples, num_features)
            - y (numpy array or Series): Target labels of shape (num_samples)
            '''
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
    
            self.classes = set(y)
            classes, counts = np.unique(y, return_counts=True)
            self.num_class = len(classes)
            
            # Calculate the mean and standard deviation of each class
            for label in self.classes:
                # Filter rows belonging to the current class
                X_class = X[y == label]
                self.class_freq[label] = len(X_class)
                
                # Clacluate mean and std for each feature in the current class
                for i in range(X.shape[1]):
                    self.class_stats[label][f'feature{i+1}_mean'] = np.mean(X_class[:, i])
                    std = np.std(X_class[:, i])
                    self.class_stats[label][f'feature{i+1}_std'] = std if std > 1e-9 else 1e-9
        
        def pdf(self, x, mean, std):
            '''
            Calculate the probability density function (PDF) value for a given data point

            Parameters
            - x (float): The value to evaluate
            - mean (float): Mean of the distribution
            - std (float): Standard deviation of the distribution

            Returns
            - float: Probability density value for the given x
            '''
            if std == 0:        # Handle zero standard deviation
                return 1.0 if x == mean else 0.0
            
            coefficient = 1 / math.sqrt(2 * math.pi) * std
            exponent = math.exp(-0.5 * ((x - mean) / std) ** 2)
            
            return coefficient * exponent

        def predict_proba(self, X):
            '''
            Predict the probabilities of each class for the given input data

            Parameters
            - X (numpy array or DataFrame): Feature matrix of shape (num_samples, num_features)

            Returns:
            - numpy array: Predicted probabilities for each class, shape (num_samples, num_classes)
            '''
            if isinstance(X, pd.DataFrame):
                X = X.values

            probabilities = []
            total_samples = sum(self.class_freq.values())

            
            for features in X:
                score = {}
                for label in self.classes:
                    # Calculate log prior probability
                    score[label] = math.log(self.class_freq[label] / total_samples)
                
                    # Add log likelihood for each feature
                    for i, feature in enumerate(features):
                        mean = self.class_stats[label][f'feature{i+1}_mean']
                        std = self.class_stats[label][f'feature{i+1}_std']
                        score[label] += math.log(self.pdf(feature, mean, std) + 1e-9)
                
                # Convert log scores to probabilities
                max_score = max(score.values())
                probability = {label: math.exp(score[label] - max_score) for label in score}
                total_probability = sum(probability.values())
                probabilities.append([probability[label] / total_probability for label in self.classes])

            probabilities = np.array(probabilities)

            return probabilities
        
        def predict(self, X):
            '''
            Predict the class labels for the given input data

            Parameters
            - X (numpy array or DataFrame): Feature matrix of shape (num_samples, num_features)

            Returns
            - list: Predicted class labels of shape (num_samples)
            '''
            probabilities = self.predict_proba(X)       # Get probabilities for each class

            # Choose the class with the highest probability
            predicted_indices = np.argmax(probabilities, axis=1)
            class_list = sorted(list(self.classes))
            predictions = [class_list[idx] for idx in predicted_indices]
            predictions = np.array(predictions).astype(int)

            return predictions
        
                
    # ------------------------------------------ Decision Tree ---------------------------------------------------
    
    class DecisionTree:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, mode='classification', num_class=None):
            '''
            DecisionTree class for classification and regression

            Parameters
            - feature (str): The feature used for splitting at this node
            - threshold (float): The threshold value for splitting the data
            - left (DecisionTree): The left subtree
            - right (DecisionTree): The right subtree
            - value (float or int): Value of the prediction at a leaf node
            - mode (str): Mode of the tree, either 'classification' or 'regression' (defalut = 'classification')
            '''
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.mode = mode
            self.root = None
            self.num_class = num_class
            
        def entropy(self, y):
            '''
            Calculate the entropy for a set of target values

            Parameters
            - y (Series): Target values

            Returns
            - float: Entropy value
            '''
            classes, counts = np.unique(y, return_counts=True)
            probability = counts / len(y)
            entropy = -np.sum(probability * np.log2(probability + 1e-9))    # Added small constant to avoid log(0)
            
            return entropy

        def gini_index(self, y):
            '''
            Calculate the Gini index for a set of target values

            Parameters
            - y (Series): Target values

            Returns
            - float: Gini index value
            '''
            classes, counts = np.unique(y, return_counts=True)
            probability = counts / len(y)
            gini_index = 1 - np.sum(probability ** 2)

            return gini_index
        
        @staticmethod
        def _parallel_fit_subtree(tree, X, y, depth, min_gain, n_jobs):
            '''
            Helper function to fit a subtree in parallel

            Parameters:
            - tree (DecisionTree): Subtree to fit
            - X (DataFrame): Input features
            - y (Series): Target labels
            - depth (int): Current depth
            - min_gain (float): Minimum gain threshold
            - n_jobs (int): Number of jobs for parallel processing

            Returns:
            - DecisionTree: Trained subtree
            '''
            tree.fit(X, y, depth, min_gain, n_jobs)
            return tree
        
        def fit(self, X, y, depth=0, min_gain=0.01, n_jobs=-1):
            '''
            Build the decision tree recursively

            Parameters
            - X (numpy array or DataFrame): Input features
            - y (Series): Target values
            - depth (int): Current depth of the tree (default = 0)
            - min_gain (folat): Minimum information gain required to split a node (default = 0.01)
            - n_jobs (int): Number of jobs for parallel processing (default = -1)
            ''' 
            unique_classes = np.unique(y)
            
            if self.num_class is None:
                self.num_class = len(unique_classes)
            
            best_feature, best_threshold, best_gain = self._find_best_split(X, y)

            if best_gain == -float('inf'):
                self.value = np.bincount(y).argmax() if self.mode == 'classification' else np.mean(y)
                return

            # Stopping condition: All labels are the same
            if len(unique_classes) == 1:
                self.value = y.iloc[0] if isinstance(y, pd.Series) else y[0]
                logger.debug(f"Stopping at leaf: class={self.value}, num_class={self.num_class}")
                return
            
            # Stopping condition: Not enough samples
            min_samples = max(10, int(0.05 * len(y)))
            if len(y) < min_samples:
                self.value = y.mode()[0] if self.mode == 'classification' else y.mean()
                return
            
            # Find the best split
            best_feature, best_threshold, best_gain = self._find_best_split(X, y)
            if best_gain < min_gain:
                self.value = y.mode()[0] if self.mode == 'classification' else y.mean()
                logger.debug(f"Stopping due to low gain: value={self.value}, num_class={self.num_class}")
                return
            
            # Perform the split
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                # Handle case where one split is empty
                self.value = y.mode()[0] if self.mode == 'classification' else y.mean()
                logger.debug(f"Stopping due to empty split: value={self.value}, num_class={self.num_class}")
                return

            # Create left and right subtrees
            self.feature = best_feature
            self.threshold = best_threshold
            self.left = numeric.DecisionTree(mode=self.mode, num_class=self.num_class)
            self.right = numeric.DecisionTree(mode=self.mode, num_class=self.num_class)

            results = Parallel(n_jobs=n_jobs)(
                delayed(self._parallel_fit_subtree)(tree, X[mask], y[mask], depth + 1, min_gain, n_jobs)
                for tree, mask in [(self.left, left_mask), (self.right, right_mask)]
            )

            self.left, self.right = results

            logger.debug(f"Tree built at depth {depth} with feature={self.feature} and threshold={self.threshold}")

        def _find_best_split(self, X, y):
            '''
            Find the best feature and threshold to split the data

            Parameters
            - X (DataFrame): Input features
            - y (Series): Target values

            Returns
            - best_feature (str): The feature providing the best split
            - best_threshold (float): The threshold value for the best split
            - best_gain (float): The information gain for the best split
            '''
            best_gain = -float('inf')
            best_feature = None
            best_threshold = None

            # Parent node entropy
            parent_entropy = self.entropy(y)

            # Iterate over all features
            for feature in X.columns:
                unique_values = np.unique(X[feature])
                if len(unique_values) > 50:
                    unique_values = np.quantile(unique_values, np.linspace(0.1, 0.9, 10))
                for value in unique_values:
                    left_mask = X[feature] <= value
                    right_mask = X[feature] > value

                    left_entropy = self.entropy(y[left_mask])
                    right_entropy = self.entropy(y[right_mask])

                    # Calculate weighted entropy
                    weighted_entropy = ((len(y[left_mask]) / len(y)) * left_entropy + (len(y[right_mask]) / len(y)) * right_entropy)

                    # Calculate information gain
                    gain = parent_entropy - weighted_entropy

                    # Update the best split if gain is higher
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = value
            
            return best_feature, best_threshold, best_gain
        
        def predict(self, X):
            '''
            Predict the output for the given input data.

            Parameters
            - X (numpy array or DataFrame): Input feature

            Returns
            - list: Predicted values for all samples
            - float/int: Predicted value
            '''
            predictions = np.array([self._predict_single(row) for _, row in X.iterrows()])
            
            return predictions

        def _predict_single(self, row):
            '''
            Helper function to predict a single sample

            Parameters
            - row (Series or 1D array): Single sample

            Returns
            - float or int: Predicted value
            '''
            # If reach a leaf node, return the node value
            if self.value is not None:
                return self.value
            
            if row[self.feature] <= self.threshold:
                return self.left._predict_single(row)
            else:
                return self.right._predict_single(row)

        def predict_proba(self, X, num_class=None):
            '''
            Predict the class probabilities for the given input data

            Parameters
            - X (numpy array or DataFarme): Input feature

            Returns
            - list: Predicted probabilities for all samples
            - numpy array: Predicted distribution over classes for classification, or predicted value for regression
            '''
            prob_results = []

            for _, row in X.iterrows():
                prob_results.append(self._predict_proba_single(row))
            
            return prob_results

        def _predict_proba_single(self, row):
            '''
            Helper function to predict probabilities for a single sample

            Parameters
            - row (Series or 1D array): Single sample

            Returns
            - numpy array: Predicted probability distribution
            - list: Predicted probability distribution or value
            '''
            # If reach a leaf node, return the probability distribution
            if self.value is not None:
                # logger.debug(f"Leaf Node: value={self.value}, num_class={self.num_class}")
                
                if not (0 <= int(self.value) < self.num_class):
                    raise ValueError(f"[ERROR] Invalid class value: {self.value}. Expected range: 0 to {self.num_class}")
                # Return probability distribution
                prob = np.zeros(self.num_class)
                prob[int(self.value)] = 1.0
                return prob

            if self.feature is None or self.threshold is None:
                error_message = "Reached an intermidate node with no valid splits. Tree may not be properly trained."
                # logger.error(error_message)
                raise ValueError(error_message )

            if row[self.feature] <= self.threshold:
                return self.left._predict_proba_single(row)
            else:
                return self.right._predict_proba_single(row)
        
        def print_tree(self, node=None, depth=0):
            '''
            Recursively print the tree structure for debugging purposes.
            '''
            if node is None:
                node = self
            
            # Print the current node
            print(f"{'|   ' * depth}Node: feature={node.feature}, threshold={node.threshold}, value={node.value}, mode={node.mode}")

            # Recursively print the left and right subtrees
            if node.left is not None:
                print(f"{'|   ' * depth}Left:")
                self.print_tree(node.left, depth + 1)
            
            if node.right is not None:
                print(f"{'|  ' * depth}Right:")
                self.print_tree(node.right, depth + 1)
    
    # ------------------------------------------ Random Forest ---------------------------------------------------
    class RandomForest:
        
        def __init__(self, n_trees=None, max_depth=25, min_samples_split=2, mode='classification', random_state=None):
            '''
            Initialize the RandomForest model

            Parameters
            - n_trees (int): Number of trees in the forest (default = None, will be optimized)
            - max_depth (int): Maximum depth of each tree (default = 40, will be optimized)
            - min_samples_split (int): Minimum samples required to split a node (default = 2)
            - mode (str): Either 'classification' or 'regression'
            '''
            self.n_trees = n_trees
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.mode = mode
            self.trees = []
            self.num_class = 2
            self.random_state = random_state
    
        def optimize_n_trees_depth(self, X, y, n_jobs=-1, random_state=42):
            '''
            Use a validation set to explore the optimal number of trees and max depth

            Parameters
            - X (numpy array or DataFrame): Input feature
            - y (Series): Target labels
            - n_jobs (int): Number of jobs for parallel processing (default = -1 for all processors)
            - random_state (int): Random seed (default = 42)

            Returns
            - best_n_trees (int): The optimal number of trees
            - best_max_depth (int): The optimal max depth
            '''
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

            best_n_trees = None
            best_max_depth = None
            best_score = -float('inf')

            def train_and_evaluate(n_trees, max_depth):
                try:
                    forest = numeric.RandomForest(n_trees=n_trees, max_depth=max_depth, mode=self.mode)
                    forest.fit(X_train, y_train)
                    predictions = forest.predict(X_val)
                    score = accuracy_score(y_val, predictions) if self.mode == 'classification' else r2_score(y_val, predictions)
                    return score, n_trees, max_depth
            
                except Exception as e:
                    logger.error(f"Error in train_and_evaluate with n_trees={n_trees} and max_depth={max_depth}: {e}")
                    return None, n_trees, max_depth
                
            # Parallel execution for different combinations of n_trees and max_depth
            results = Parallel(n_jobs=n_jobs)(
                delayed(train_and_evaluate)(n_trees, max_depth)
                for n_trees in range(10, 101, 10)
                for max_depth in range(3, 21, 2)
            )
            results = [result for result in results if result[0] is not None]

            if not results:
                logger.error("No valid results from the train and evaluate process. Setting default values.")
                return 10, 5

            for score, n_trees, max_depth in results:
                if score> best_score:
                    best_score = score
                    best_n_trees = n_trees
                    best_max_depth = max_depth
            
            logger.info(f"Optimal number of trees: {best_n_trees}, Optimal max depth: {best_max_depth}, Best Accuracy: {best_score}")
            return best_n_trees, best_max_depth
        
        def fit(self, X, y, n_jobs=-1):
            '''
            Train the RandomForest model by fitting multiple decision trees
            If optimize_hyperparameters=True, it will automatically find the best n_trees and max_depth

            Parameters
            - X (numpy array or DataFrame): Input feature
            - y (Series): Target labels
            - n_jobs (int): Number of jobs for parallel processing (default=-1 for all processors)
            - random_state (int): Random seed (default = 42)
            '''
            if isinstance(X, pd.DataFrame):
                # If already a DataFrame, return as is
                pass
            elif isinstance(X, np.ndarray):
                # If numpy array, convert to DataFrame
                X = pd.DataFrame(X)
            elif isinstance(X, list):
                # If list, convert to DataFrame
                X = pd.DataFrame(X)
            
            classes, counts = np.unique(y, return_counts=True)
            self.num_class = len(classes)

            np.random.seed(self.random_state)
            
            if self.n_trees is None or self.max_depth is None:
                best_n_trees, best_max_depth = self.optimize_n_trees_depth(X, y, n_jobs=n_jobs)

                self.n_trees = best_n_trees
                self.max_depth = best_max_depth
            
            self.trees = Parallel(n_jobs=n_jobs)(
                delayed(self._train_tree)(X, y, i) for i in range(self.n_trees)
            )

            logger.info(f"Training compled. {len(self.trees)} trees trained and {self.num_class}.")

        def _train_tree(self, X, y, seed):#random_state, tree_idx):
            '''
            Train a single Decision Tree for the RandomForest.

            Parameters:
            - X (DataFrame or numpy array): Input features
            - y (Series or numpy array): Target labels
            - random_state (int): Random seed
            - tree_idx (int): Index of the tree being trained (used for randomization)
            
            Returns:
            - tree: The trained DecisionTree
            '''
            np.random.seed(seed)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X.iloc[indices], y.iloc[indices]
            tree = numeric.DecisionTree(mode=self.mode, num_class=self.num_class)
            tree.fit(X, y)
            return tree

        def predict(self, X):
            '''
            Predict the target values for the given input features using the trained RandomForest

            Parameters
            - X (numpy array or DataFrame): Input feature
            
            Returns
            - predictions (list): Predicted values for each sample
            '''
            if not self.trees:
                error_message = "The RandomForest has not been trained. Call 'fit' first."
                logger.error(error_message)
                raise ValueError(error_message)
            
            logger.info("Making predictions...")
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)

            predictions = [tree.predict(X) for tree in self.trees]

            if self.mode == 'classification':
                return np.array([Counter(pred).most_common(1)[0][0] for pred in zip(*predictions)])
            else:
                return np.mean(predictions, axis=0)
            
        def predict_proba(self, X):
            '''
            Predict the class probabilities for the given input features using the trained RandomForest

            Parameters
            - X (numpy array or DataFrame): Input features

            Returns
            - numpy array: Probability distribution over classes for each sample in classification or predicted value for regression
            '''
            if not self.trees:
                raise ValueError("The RandomForest has not been trained. Call 'fit' first.")
            
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            logger.info("Predicting probabilities...")
            
            probabilities = [tree.predict_proba(X, self.num_class) for tree in self.trees]
            return np.mean(probabilities, axis=0)
            
        def check_trees(self):
            '''
            Check the status of all trees in the forest for debugging purposes
            '''
            for i, (tree, feature_indices) in enumerate(self.trees):
                if tree is None:
                    logger.error(f"Tree {i} is None!")
                else:
                    logger.debug(f"Tree {i}: {tree}, Feature Indices: {feature_indices}")
        
        def get_params(self, deep=True):
            '''
            Return the parameters of the model for RandomizedSearchCV or GridSearchCV.

            Parameters:
            - deep (bool): Whether to return parameters of sub-estimators (default is True)
            
            Returns:
            - params (dict): Dictionary of model parameters
            '''
            params = {
                'n_trees': self.n_trees,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'mode': self.mode,
                'random_state': self.random_state
            }

            if deep:
                params.update({'tree_params': {
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                }})
            
            return params
        
        def set_params(self, **params):
            '''
            Set parameters for the model.

            Parameters:
            - **params (dict): Dictionary of parameters to set
            '''
            for param, value in params.items():
                if param in self.__dict__:
                    setattr(self, param, value)
                elif param == 'tree_params' and 'tree_params' in params:
                    tree_params = params['tree_params']
                    for tree in self.trees:
                        tree.set_params(**tree_params)
            return self


    # ---------------------------------------- Logistic Regression -------------------------------------------------
    class LogisticRegression:
        def __init__(self, learning_rate=0.001, max_epochs=1000, L2=0.01, num_class=2):
            '''
            Initialize the logistic regression model with parameters

            Parameters:
            - learning_rate (float): Step size for gradient descent (Default is 0.01)
            - max_epochs (int): Maximum number of epochs for training (Default is 500)
            '''
            self.learning_rate = learning_rate
            self.max_epochs = max_epochs
            self.L2 = L2
            self.num_class = num_class
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
            - z (numpy array): Input array where each row represents a set of raw class scores (logits)
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

            #num_samples, self.num_class = pred_probs.shape
            y_one_hot = np.zeros((len(y), self.num_class))

            # Ensure y values are valid indices
            if np.max(y) >= self.num_class:
                error_message = f"Invalid class label in y. Max class index({np.max(y)}) exceeds num_class ({self.num_class})."
                logger.error(error_message)
                raise ValueError(error_message)

            y_one_hot[np.arange(len(y)), y] = 1
            
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
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

            # Early stopping initialization
            best_val_loss = float('inf')                    # Set the initial best validation loss to infinity 
            epochs_wout_improvement = 0                     # Counter for patience
            best_w, best_b = self.w.copy(), self.b.copy()                 # Best weights and bias to restore after early stopping

            # Training loop over epochs
            for epoch in range(self.max_epochs):
                epoch_train_loss = 0
                epoch_val_loss = 0

                # Perform k-fold cross-validation
                for train_idx, val_idx in skf.split(X, y):
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
                    logger.info(f"Early stopping triggered at epoch {epoch} with validation loss: {epoch_val_loss:.4f}")
                    break       # Stop training if no improvement in 'patience' epochs

                if epoch % 100 == 0:
                    logger.info(f"Epoch {epoch}, Training loss: {train_losses[-1]:.4f}, Validation loss: {val_losses[-1]:.4f}")
            
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
                error_message = "Model not trained yet. Call fit() before predict."
                logger.error(error_message)
                raise ValueError(error_message)
            
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
    
    
# ========================================= Text ==============================================================
# Models for the text dataset
class Text:
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)  
        return text

    class TextVectorizer:
        def __init__(self):
            self.vocabulary = {}
            self.inverse_vocabulary = []
            self.document_count = 0
            self.word_document_freq = {}
        
        def fit(self, documents):
            self.document_count = len(documents)

            for doc in documents:
                seen_words = set()
                for word in doc.split():
                    if word not in self.vocabulary:
                        self.vocabulary[word] = len(self.inverse_vocabulary)
                        self.inverse_vocabulary.append(word)
                    if word not in seen_words:
                        self.word_document_freq[word] = self.word_document_freq.get(word, 0) + 1
                        seen_words.add(word)
        
        def transform(self, documents):
            num_docs = len(documents)
            num_words = len(self.vocabulary)
            bow_matrix = np.zeros((num_docs, num_words), dtype=int)

            for i, doc in enumerate(documents):
                for word in doc.split():
                    if word in self.vocabulary:
                        bow_matrix[i, self.vocabulary[word]] += 1
            
            return bow_matrix
        
        def compute_tfidf(self, bow_matrix):
            # Calculate TF
            tf = bow_matrix / np.maximum(bow_matrix.sum(axis=1, keepdims=True), 1)

            # Calculate IDF
            idf = np.zeros(len(self.vocabulary))
            for word, idx in self.vocabulary.items():
                idf[idx] = np.log((self.document_count + 1) / (self.word_document_freq[word] + 1)) + 1    # Add-1 smoothing
            
            tfidf_matrix = tf * idf

            return tfidf_matrix

# ============================================= Tuning ===========================================================
class tuning:
    
    def tune_hyperparameters(model, param_dist, X, y, n_iter=100, cv=5, random_state=42, n_jobs=-1):
        '''
        Optimize the hyperparameters of a given model using RandomizedSearchCV and evaluate its performance

        Parameters
        - model: The model to optimize (LogisticRegression)
        - param_dist (dictionary): The search space for hyperparameters
        - X (numpy array or DataFrame): Input features
        - y (numpy array or Series): Input labels
        - n_iter: Number of hyperparameter combination to try
        - cv: Number of cross-validation folds
        - random_state: Random seed (default is 42)
        - n_jobs: Number of jobs for parallel processing (default = -1 for all processors)

        Returns:
        - Best hyperparameters, model performance metrics
        '''   
        try:
            y, label_map = preprocess.map_target(y)
            num_class = len(np.unique(y))
            if hasattr(model, 'num_class'):
                model.num_class = num_class

            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=random_state)

            # Using RandomizedSearchCV to optimize hyperparameters
            random_search = RandomizedSearchCV(estimator=model,
                                            param_distributions=param_dist,
                                            n_iter=n_iter,
                                            cv=cv,
                                            scoring='accuracy',
                                            n_jobs=n_jobs,
                                            random_state=random_state,
                                            refit=True)    # Choose the best model based on 'accurary'
            

            # Fit the model
            random_search.fit(train_X, train_y)

            # Output the best hyperparameters
            best_params = random_search.best_params_

            # Predict using the best model
            best_model = random_search.best_estimator_
            test_predictions = best_model.predict(test_X)

            # Evaluate metrics
            performance_metrics = {}
            accuracy = accuracy_score(test_y, test_predictions)
            f1 = f1_score(test_y, test_predictions, average='weighted')


            if hasattr(best_model, 'predict_proba') and callable(getattr(best_model, 'predict_proba')):
                test_probabilities = best_model.predict_proba(test_X)
                if test_probabilities.ndim == 1:
                    test_probabilities = test_probabilities.reshape(-1, 1)
                roc_auc = roc_auc_score(test_y, test_probabilities, multi_class='ovr', average='weighted')
            else:
                roc_auc = None       # No predict_proba available

            performance_metrics['Accuracy'] = accuracy
            performance_metrics['F1 Score'] = f1
            performance_metrics['ROC AUC'] = roc_auc

            logger.info("Scores with tuned hyperparameters: ")
            logger.info(f"Accuracy: {accuracy: .4f}")
            logger.info(f"F1 Score (Weighted): {f1: .4f}")
            logger.info(f"ROC AUC: {roc_auc: .4f}" if roc_auc is not None else "ROC AUC not available.")

            logger.info("Hyperparameter tuning complete.")

            return best_params, performance_metrics, best_model
    
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            return None, None, None
    
    
# ============================================== Select Model ================================================
# Select the best model class
class select_model:
    # k-Fold Cross-Validation function
    def cross_validation(model_class, X, y, k=5, mode='classification', n_jobs=-1):
        '''
        Perform k-fold cross-validation for a given model and dataset with ROC-AUC curve score for classification,
        R^2 score for regression using joblib

        Parameters
        - model: Classification models (eg, Naive Bayes, Decision Tree, Random Forest, Logistic Regression, etc.)
        - X (numpy array or DataFrame): Feature matrix of shape (num_samples, num_features)
        - y (numpy array or Series): Target labels of shape (num_samples)
        - k (int): Number of folds for cross-validation (default = 5)
        - mode (str): 'classification' or 'regression' (default = 'classification')
        - n_jobs (int): Number of jobs for parallel processing (default = -1)

        Returns
        - float: The average score across all folds
        '''
        try:
            # Initialize k-fold cross-validation splitter
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            logger.debug(f"Current model (cross_validation_joblib): {model_class}")

            def train_and_evaluate(train_idx, val_idx):
                # Split the data
                if isinstance(X, pd.DataFrame):
                    train_X, val_X = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    train_X, val_X = X[train_idx], X[val_idx]
                
                if isinstance(y, pd.Series):
                    train_y, val_y = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    train_y, val_y = y[train_idx], y[val_idx]

                # Initialize the model
                if isinstance(model_class, type):  # If it's a class
                    model = model_class()
                else:  # If it's already an instance
                    model = model_class
                
                model.fit(train_X, train_y)

                if mode == 'regression':
                    predictions = model.predict(val_X)
                    return r2_score(val_y, predictions)
                elif mode == 'classification':
                    prob_y = model.predict_proba(val_X)
            
                    if hasattr(model, "num_class") and model.num_class == 2:
                        return roc_auc_score(val_y, prob_y[:, 1])
                    else:
                        return roc_auc_score(val_y, prob_y, multi_class='ovr', average='weighted')
            
            scores = Parallel(n_jobs=n_jobs)(
                delayed(train_and_evaluate)(train_idx, val_idx)
                for train_idx, val_idx in skf.split(X, y)
            )
            
            avg_score = np.mean([score for score in scores if score is not None])        # Filter out failed folds

            logger.debug(f"Cross-validation results for all folds: {scores}")

            return avg_score
        except Exception as e:
            logger.error(f"Error during cross-validation with joblib: {e}")
            return None

    # Model selection function
    def model_selection(models, X, y, mode='classification', k=5):
        '''
        Seleect the best model for the given dataset using k-fold cross-validation and Dask for parallel computation

        Parameters
        - models (dict): Dictionary of models to be used
        - X (numpy array or DataFrame): Feature matrix of shape (num_samples, num_features)
        - y (numpy array or Series): Target labels of shape (num_samples)
        - k (int): Number of folds for cross-validation (default=5)

        Returns
        - tuple
            - best_model: The model with the best performance based on the ROC-AUC curve
            - best_score (float): The average score of the best model across all folds
        '''
        logger.info("Starting model selection...")
        start_time = time.time()
        try:
            y_type = preprocess.check_target_type(y)
            if y_type == 'categorical':
                y, label_map = preprocess.map_target(y)
            else:
                y, label_map = y, None
            
            # Run all tasks in parallel
            results = []
            for model_name, model in models.items():
                # Skip models that do not match the detected mode
                if ('classification' in model_name and mode == 'regression') or \
                ('regression' in model_name and mode == 'classification'):
                    logger.info(f"Skipping model: {model_name} (not applicable for mode: {mode})")
                    continue

                logger.info(f"Processing model: {model_name}")
                model_start_time = time.time()

                if model_name in [
                    'Decision Tree classification',
                    'Decision Tree regression',
                    'Random Forest classification',
                    'Random Forest regression',
                    'Tuned Logistic Regression'
                    ]:

                    if model_name == 'Tuned Logistic Regression':
                        logger.info(f"Applying scaler to Tuned Logistic Regression.")
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                        train_X = pd.DataFrame(train_X).reset_index(drop=True)
                        test_X = pd.DataFrame(test_X).reset_index(drop=True)
                        train_y = train_y.reset_index(drop=True)
                        test_y = test_y.reset_index(drop=True)

                        model.fit(train_X, train_y)
                    
                    else:
                        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

                        train_X = pd.DataFrame(train_X).reset_index(drop=True)
                        test_X = pd.DataFrame(test_X).reset_index(drop=True)
                        train_y = train_y.reset_index(drop=True)
                        test_y = test_y.reset_index(drop=True)

                        model.fit(train_X, train_y)

                    if mode == 'classification':
                        if isinstance(model, Pipeline):
                            inner_model = None
                            for step_name in ['Decision Tree classification', 'Decision Tree regression', 'Random Forest classification', 'Random Forest regression', 'Tuned Logistic Regression']:
                                if step_name in model.named_steps:
                                    inner_model = model.named_steps[step_name]
                                    break
                        else:
                            inner_model = model

                        prob_y = model.predict_proba(test_X)
                        if hasattr(inner_model, 'num_class') and inner_model.num_class == 2:
                            score = roc_auc_score(test_y, prob_y)
                            logger.info(f"[{model_name}] Classification ROC-AUC score (binary): {score: .4f}")
                        else:
                            score = roc_auc_score(test_y, prob_y, multi_class='ovr', average='weighted')
                            logger.info(f"[{model_name}] Classification ROC-AUC score (multi-class): {score: .4f}")
                    else:
                        predictions = model.predict(test_X)
                        score = r2_score(test_y, predictions)
                        logger.info(f"[{model_name}] Regression R^2 score: {score: .4f}")
                    
                    logger.debug(f"{model_name} evaluation score: {score}")
                    results.append((model_name, model, score))
                
                else:
                    # Perform cross-validation fro other models
                    logger.info(f"Scheduling cross-validation fro model: {model_name}")
                    score = select_model.cross_validation(model, X, y, k, mode=mode)
                    results.append((model_name, model, score))
                    logger.debug(f"{model_name} cross-validation score: {score}")
                
                model_end_time = time.time() - model_start_time
                logger.info(f"Model {model_name} execution time: {model_end_time: .2f} seconds")

            # Find the best model based on score
            used_model_name = []
            best_model = None
            best_score = -float('inf')
            best_model_name = None
            for model_name, model, score in results:
                used_model_name.append(model_name)
                if score is not None and score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name
            
            total_time = time.time() - start_time
            logger.info(f"Best Model: {best_model}, Best Score: {best_score: .4f}")
            logger.info(f"Model selection completed in {total_time: .2f} seconds.")
            
            return best_model_name, used_model_name, best_model, best_score, label_map, y_type
        
        except Exception as e:
            logger.error(f"Error during model selection {model_name}: {e}")
            return None, None, None, None, None, None


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

def build_model_dict(X, y):
    '''
    Build the models dictionary to use model_selection

    Parameters
    - X (DataFrame): Input feature
    - y (DataFrame or Series): Input labels

    Returns
    - models (dict): Dictionary of the models
    '''
    dt_classification = numeric.DecisionTree(mode='classification')
    dt_regression = numeric.DecisionTree(mode='regression')

    # Use Pipeline
    rf_classification_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('rf_model', numeric.RandomForest(mode='classification'))
    ])

    rf_regression_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('rf_model', numeric.RandomForest(mode='regression'))
    ])

    logistic_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('logistic_model', numeric.LogisticRegression())
    ])

    # Logistic Regression hyperparameters tuning
    logi_model = numeric.LogisticRegression()

    param_dist = {
                    'L2': uniform(0.01, 10),
                    'learning_rate': uniform(0.0001, 0.01),
                    'penalty': ['12'],
                    'solver': ['lbfgs', 'liblinear']
                }
    
    LR_best_params, metrics, tuned_logistic = tuning.tune_hyperparameters(
            model=logi_model,
            param_dist=param_dist,
            X=X,
            y=y,
            n_iter=50,
            cv=3,
            random_state=42
        )
    
    models = {
        'Naive Bayes': numeric.gausian_NaiveBayes(),
        'Decision Tree classification': dt_classification,
        'Decision Tree regression': dt_regression,
        'Random Forest classification': rf_classification_pipeline,
        'Random Forest regression': rf_regression_pipeline,
        'Logistic Regression': logistic_pipeline,
        'Tuned Logistic Regression': tuned_logistic
    }

    return models, LR_best_params, metrics


class BestModel:
    def __init__(self, model, label_mapping=None):
        self.model = model
        self.label_mapping = label_mapping
    
    def fit(self, X, y):
        y, label_mapping = preprocess.map_target(y)
        self.label_mapping = label_mapping
        self.model.fit(X, y)
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return preprocess.reverse_map(predictions, self.label_mapping)

def individual_model(model_choice, X, y, mode='classification'):
    '''
    Train and Evaluate individual model depend on user choice

    Parameters
    - model_choice (str): User choose model
    - retression (bool): If true, 'regression', if false, 'classification'

    Returns
    - model: Trained model
    - accuracy: Accuracy score
    - score: R^2 score for regression, ROC-AUC score for classification
    '''
    y_type = preprocess.check_target_type(y)
    if y_type == 'categorical':
        y, label_map = preprocess.map_target(y)
    else:
        y, label_map = y, None
    
    model = None

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == 'Naive Bayes':
        model = numeric.gausian_NaiveBayes()
    elif model_choice == 'Decision Tree':
        model = numeric.DecisionTree(mode=mode)
    elif model_choice == 'Random Forest':
        model = numeric.RandomForest(mode=mode)
    elif model_choice == 'Logistic Regression':
        model = numeric.LogisticRegression()
    else:
        error_message = f"Invalid model choice: {model_choice}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info(f"Training {model_choice}...")

    start_time = time.time()

    try:
        model.fit(train_X, train_y)
    except Exception as e:
        logger.error(f"Error training {model_choice}: {e}")
        raise

    training_time = time.time() - start_time
    logger.info(f"Training {model_choice} completed in {training_time: .2f} seconds.")

    logger.info(f"Evaluating {model_choice}...")

    try:
        predictions = model.predict(test_X)
    except Exception as e:
        logger.error(f"Error predicting with {model_choice}: {e}")
        raise

    accuracy = accuracy_score(test_y, predictions)

    if mode == 'regression':      # Regression
        score = r2_score(test_y, predictions)
        logger.info(f"[{model_choice}] Regression R^2 score: {score: .4f}")
    else:               # Classification
        prob_y = model.predict_proba(test_X)
        if hasattr(model, 'num_class') and model.num_class == 2:
            score = roc_auc_score(test_y, prob_y)
            logger.info(f"[{model_choice}] Classification ROC-AUC score (binary): {score: .4f}")
        else:
            score = roc_auc_score(test_y, prob_y, multi_class='ovr', average='weighted')
            logger.info(f"[{model_choice}] Classification ROC-AUC score (multi-class): {score: .4f}")
    
    logger.info(f"{model_choice} evaluation completed with accuracy: {accuracy: .4f}, score: {score: .4f}")
    
    return model, accuracy, score, y_type, label_map
