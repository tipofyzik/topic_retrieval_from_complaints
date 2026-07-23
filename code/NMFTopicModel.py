from sklearn.decomposition import NMF
import numpy as np



class NMFTopicModel:
    """
    Extracts topics from text data using Non-negative Matrix Factorization.

    Attributes:
        __nmf (NMF): The sklearn NMF model used for topic extraction.
    """

    def __init__(self, n_topics: int = 10, random_state: int = 0):
        """
        Initializes the NMFTopicModel with the specified number of topics.

        Args:
            n_topics (int): Number of topics to extract from the dataset.
            random_state (int): Random state for reproducible results.
        """
        self.__nmf = NMF(n_components=n_topics, random_state=random_state, 
                         init="nndsvda")

    def fit(self, tfidf_matrix) -> None:
        """
        Fits the NMF model on TF-IDF representations.

        Args:
            tfidf_matrix: Sparse matrix containing TF-IDF representations
                          of the dataset.

        Returns:
            None
        """
        self.__nmf.fit(tfidf_matrix)

    def fit_transform(self, tfidf_matrix) -> np.ndarray:
        """
        Fits the NMF model and transforms documents into topic distributions.

        Args:
            tfidf_matrix: Sparse matrix containing TF-IDF representations
                          of the dataset.

        Returns:
            numpy.ndarray: Matrix containing topic distribution for each document.
        """
        return self.__nmf.fit_transform(tfidf_matrix)

    def get_topics(self, feature_names: list[str], 
                   n_words: int = 10) -> dict[str, list[str]]:
        """
        Returns the most representative words for each extracted topic.

        Args:
            feature_names (list[str]): List of words representing the vocabulary.
            n_words (int): Number of top words to include for each topic.

        Returns:
            dict[str, list[str]]: Dictionary containing topic names as keys
                                  and corresponding words as values.
        """
        self.__topics = {}
        for topic_idx, topic in enumerate(self.__nmf.components_):
            words = [feature_names[i] 
                     for i in topic.argsort()[:-n_words - 1:-1]
                     ]
            self.__topics[f"Topic {topic_idx + 1}"] = words
        return self.__topics

    def print_topics(self) -> None:
        """
        Prints the stored topics and their most representative words.

        Returns:
            None
        """
        for topic_name, words in self.__topics.items():
            print(f"{topic_name}:")
            print(", ".join(words))
            print()