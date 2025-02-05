from utils import SentimentExample
from typing import List
from collections import Counter
import time
from tokenizers import Tokenizer
import numpy as np
from tqdm import tqdm

import gensim.downloader as api


class FeatureExtractor(object):
    """
    Base class for feature extraction. Takes a text and returns an indexed list of features.
    """

    def extract_features(self, text: str) -> Counter:
        """
        Extract features from a text.
        :param text: Text to featurize.
        :return: A feature vector (e.g., Counter mapping token ids to counts).
        """
        raise Exception("Don't call me, call my subclasses")


class CountFeatureExtractor(FeatureExtractor):
    """
    Extracts count features from text. The tokenizer returns token ids and we count their occurrences.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenizer)

    def extract_features(self, text: str) -> Counter:
        tokens = self.tokenizer.tokenize(text)
        feature_counter = Counter()
        for tok in tokens:
            tok_id = self.tokenizer.token_to_id.get(tok)
            if tok_id is None:
                tok_id = self.tokenizer.token_to_id.get((tok,))
            if tok_id is not None:
                feature_counter[tok_id] += 1
        return feature_counter


class CustomFeatureExtractor(FeatureExtractor):
    """
    Custom feature extractor that extracts features from text using a custom approach.
    In this example, tokens are first lowercased and only tokens longer than 2 characters are counted.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenizer)

    def extract_features(self, text: str) -> Counter:
        """
        Custom feature extraction:
         1. Tokenize the text.
         2. Lowercase tokens and filter out tokens with length <= 2.
         3. Map the remaining tokens to token ids and count occurrences.
        """
        tokens = self.tokenizer.tokenize(text)
        feature_counter = Counter()
        for tok in tokens:
            tok_lower = tok.lower()
            if len(tok_lower) > 2:
                tok_id = self.tokenizer.token_to_id(tok_lower)
                if tok_id is not None:
                    feature_counter[tok_id] += 1
        return feature_counter


class MeanPoolingWordVectorFeatureExtractor(FeatureExtractor):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        print("Loading word2vec model...")
        self.word_to_vector_model = api.load("glove-twitter-25")
        print("Word2vec model loaded")

    def __len__(self):
        # the glove-twitter word vectors have 25 dimensions
        return 25

    def get_word_vector(self, word) -> np.ndarray:
        """
        Get the word vector for a word from the loaded model.
        If the word is not in the vocabulary, return None.
        """
        if word in self.word_to_vector_model:
            return self.word_to_vector_model[word]
        else:
            return None

    def extract_features(self, text: List[str]) -> Counter:
        """
        Extract mean pooling word vector features from a text.
        Steps:
         1. Tokenize the text.
         2. For each word, obtain its word vector.
         3. Average all available word vectors to get a mean vector.
         4. Convert the mean vector to a Counter mapping (as required by the framework).
           (Normally, you would use the vector directly, but we need a Counter here.)
        """
        tokens = self.tokenizer.tokenize(text)
        vectors = []
        for word in tokens:
            vec = self.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
        if len(vectors) == 0:
            # If no word vectors found, return a zero vector in Counter format.
            mean_vector = np.zeros(len(self), dtype=np.float64)
        else:
            mean_vector = np.mean(vectors, axis=0)
        # Convert the mean vector to a Counter mapping index to value.
        return Counter({i: mean_vector[i] for i in range(len(mean_vector))})


class SentimentClassifier(object):
    """
    Base class for sentiment classifiers.
    """

    def predict(self, text: List[str]) -> int:
        """
        :param text: List of words in the text.
        :return: 0 for negative or 1 for positive sentiment.
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    A trivial sentiment classifier that always predicts the positive class.
    """

    def predict(self, text: List[str]) -> int:
        return 1


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.
    """
    if x < 0:
        return np.exp(x) / (1 + np.exp(x))
    return 1 / (1 + np.exp(-x))


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Logistic regression classifier that uses a featurizer to convert text into feature vectors.
    It learns a binary classifier with weights and bias.
    """

    def __init__(self, featurizer: FeatureExtractor):
        """
        Initialize the logistic regression classifier with weights and bias set to zero.
        """
        self.featurizer = featurizer
        self.weights = np.zeros(len(self.featurizer), dtype=np.float64)
        self.bias = 0

    def predict(self, text: str) -> int:
        """
        Predict the sentiment of a text.
        Steps:
         1. Extract features from the text.
         2. Compute the score as a dot product of weights and features plus bias.
         3. Compute the sigmoid of the score.
         4. Return 1 if the sigmoid output is >= 0.5, otherwise return 0.
        """
        features = self.featurizer.extract_features(text)
        score = self.bias
        for feat_id, feat_value in features.items():
            score += self.weights[feat_id] * feat_value
        prob = sigmoid(score)
        return 1 if prob >= 0.5 else 0

    def set_weights(self, weights: np.ndarray):
        """
        Set the weights of the model.
        """
        self.weights = weights

    def set_bias(self, bias: float):
        """
        Set the bias of the model.
        """
        self.bias = bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def training_step(self, batch_exs: List[SentimentExample], learning_rate: float, l2_lambda: float = 0.001):
        """
        Perform a single training step on a batch of examples.  Update the weights and bias using the gradient of the loss.
        """
        grad_w = np.zeros_like(self.weights, dtype=np.float64)
        grad_b = 0.0
        misclassified_count = 0


        for ex in batch_exs:
            feats = self.featurizer.extract_features(ex.words)
            score = self.bias + sum(self.weights[feat_id] * feat_value for feat_id, feat_value in feats.items())
            p = sigmoid(score)
            predicted_label = 1 if p >= 0.5 else 0

            #  skip if predicted label is correct
            if predicted_label == ex.label:
                continue

            # cal
            error = predicted_label - ex.label
            grad_b += error
            for feat_id, feat_value in feats.items():
                grad_w[feat_id] += error * feat_value
            misclassified_count += 1

        if misclassified_count > 0:
            # L2 regularization
            grad_w -= np.round(l2_lambda * self.weights,1)

            # Update weights and bias
            self.bias -= learning_rate * grad_b
            self.weights -= learning_rate * grad_w / misclassified_count


def get_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Calculate the accuracy of predictions.
    """
    num_correct = sum(1 for i in range(len(predictions)) if predictions[i] == labels[i])
    return num_correct / len(predictions)


def run_model_over_dataset(
    model: SentimentClassifier, dataset: List[SentimentExample]
) -> List[int]:
    """
    Run the model over the entire dataset and return the list of predictions.
    """
    predictions = []
    for ex in dataset:
        predictions.append(model.predict(ex.words))
    return predictions


def train_logistic_regression(
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    feat_extractor: FeatureExtractor,
    learning_rate: float = 0.01,
    batch_size: int = 10,
    epochs: int = 10,
) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    Steps:
     1. Initialize the model and variables to track the best dev accuracy.
     2. Use an exponential decay learning rate scheduler.
     3. For each epoch, shuffle the training examples and update the model in mini-batches.
     4. Evaluate on the dev set and save the model parameters if improved.
     5. After training, restore the best model parameters.
    """
    # Initialize model
    model = LogisticRegressionClassifier(feat_extractor)
    best_dev_acc = 0.0
    best_weights = None
    best_bias = 0.0

    # Exponential decay learning rate scheduler
    scheduler = lambda epoch: learning_rate * (0.95 ** epoch)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Shuffle the training examples (create a new randomly ordered list)
        shuffled_train_exs = np.random.permutation(train_exs)
        cur_learning_rate = scheduler(epoch)

        # Iterate over mini-batches
        for i in range(0, len(shuffled_train_exs), batch_size):
            batch_exs = shuffled_train_exs[i: i + batch_size]
            model.training_step(batch_exs, cur_learning_rate)

        # Evaluate on the development set
        dev_preds = run_model_over_dataset(model, dev_exs)
        dev_labels = [ex.label for ex in dev_exs]
        cur_dev_acc = get_accuracy(dev_preds, dev_labels)

        # Save best model if performance improved
        if cur_dev_acc > best_dev_acc:
            best_dev_acc = cur_dev_acc
            best_weights = np.copy(model.get_weights())
            best_bias = model.get_bias()

        # Log metrics to the progress bar
        metrics = {"best_dev_acc": best_dev_acc, "cur_dev_acc": cur_dev_acc, "epoch": epoch}
        pbar.set_postfix(metrics)

    # Restore the best model parameters
    if best_weights is not None:
        model.set_weights(best_weights)
        model.set_bias(best_bias)

    return model


def train_model(
    args,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    tokenizer: Tokenizer,
    learning_rate: float,
    batch_size: int,
    epochs: int,
) -> SentimentClassifier:
    """
    Main entry point for training the sentiment classifier.
    Depending on args, it instantiates a feature extractor and trains either a trivial or logistic regression model.
    """
    # Initialize feature extractor
    if args.feats == "COUNTER":
        feat_extractor = CountFeatureExtractor(tokenizer)
    elif args.feats == "WV":
        feat_extractor = MeanPoolingWordVectorFeatureExtractor(tokenizer)
    elif args.feats == "CUSTOM":
        feat_extractor = CustomFeatureExtractor(tokenizer)
    else:
        raise Exception("Unknown feature type")

    # Train the model based on specified type
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "LR":
        model = train_logistic_regression(
            train_exs,
            dev_exs,
            feat_extractor,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
        )
    else:
        raise Exception("Pass in TRIVIAL or LR to run the appropriate system")
    return model
