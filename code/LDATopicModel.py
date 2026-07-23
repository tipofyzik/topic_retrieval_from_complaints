from sklearn.decomposition import LatentDirichletAllocation
import numpy as np



class LDATopicModel:
    """
    Extracts topics from text data using Latent Dirichlet Allocation.

    Attributes:
        __lda (LatentDirichletAllocation): The sklearn LDA model used for topic extraction.
    """

    def __init__(self, n_topics: int = 10, random_state: int = 0):
        """
        Initializes the LDATopicModel with the specified number of topics.

        Args:
            n_topics (int): Number of topics to extract from the dataset.
            random_state (int): Random state for reproducible results.
        """
        self.__lda = LatentDirichletAllocation(n_components=n_topics,
                                               random_state=random_state,
                                               learning_method="online")

    def fit(self, bow_matrix) -> None:
        """
        Fits the LDA model on Bag of Words representations.

        Args:
            bow_matrix: Sparse matrix containing Bag of Words representations
                        of the dataset.

        Returns:
            None
        """
        self.__lda.fit(bow_matrix)

    def fit_transform(self, bow_matrix) -> np.ndarray:
        """
        Fits the LDA model and transforms documents into topic distributions.

        Args:
            bow_matrix: Sparse matrix containing Bag of Words representations
                        of the dataset.

        Returns:
            numpy.ndarray: Matrix containing topic distribution for each document.
        """
        return self.__lda.fit_transform(bow_matrix)

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

        for topic_idx, topic in enumerate(self.__lda.components_):
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
