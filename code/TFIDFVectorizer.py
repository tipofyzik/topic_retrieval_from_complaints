from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import scipy



class TFIDFVectorizer:
    """
    Converts preprocessed texts into TF-IDF vector representations.

    Attributes:
        __vectorizer (TfidfVectorizer): The sklearn vectorizer used to build the vocabulary
                                        and transform texts into TF-IDF representations.
    """

    def __init__(self, max_features: int = 10000, 
                 min_df: int = 5, max_df: float = 0.95):
        """
        Initializes the TFIDFVectorizer with parameters for vocabulary creation.

        Args:
            max_features (int): Maximum number of features to keep in the vocabulary.
            min_df (int): Minimum document frequency required for a word to be included.
            max_df (float): Maximum document frequency allowed for a word to be included.
        """
        self.__vectorizer = TfidfVectorizer(max_features=max_features, 
                                            min_df=min_df, max_df=max_df)

    def fit_transform(self, texts: pd.Series) -> scipy.sparse.csr_matrix:
        """
        Learns the vocabulary from texts and transforms them into TF-IDF vectors.

        Args:
            texts (list[str]): List of preprocessed texts used to build the vocabulary.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix containing TF-IDF representations.
        """
        return self.__vectorizer.fit_transform(texts)

    def transform(self, texts: pd.Series) -> scipy.sparse.csr_matrix:
        """
        Transforms texts into TF-IDF vectors using the previously learned vocabulary.

        Args:
            texts (list[str]): List of preprocessed texts to transform.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix containing TF-IDF representations.
        """
        return self.__vectorizer.transform(texts)

    def get_feature_names(self) -> list[str]:
        """
        Returns the names of features learned during vocabulary creation.

        Returns:
            list[str]: List of feature names representing the vocabulary.
        """
        return self.__vectorizer.get_feature_names_out().tolist()