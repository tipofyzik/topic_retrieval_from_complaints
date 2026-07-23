import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class TopicAnalyzer:
    """
    Analyzes and visualizes topic modeling results.

    Supports comparison and visualization of topic models such as
    Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF).
    """

    def __init__(self):
        """
        Initializes the TopicAnalyzer.
        """
        pass

    def topics_to_dataframe(self, topics: dict[str, list[str]]) -> pd.DataFrame:
        """
        Converts extracted topics into a pandas DataFrame.

        Args:
            topics (dict[str, list[str]]): Dictionary containing topic names
                                           and their representative words.

        Returns:
            pd.DataFrame: DataFrame representation of topics.
        """
        data = []

        for topic_name, words in topics.items():
            data.append(
                {
                    "Topic": topic_name,
                    "Top words": ", ".join(words)
                }
            )
        return pd.DataFrame(data)

    def save_topics(self, topics: dict[str, list[str]],
                    filename: str) -> None:
        """
        Saves extracted topics into a CSV file.

        Args:
            topics (dict[str, list[str]]): Extracted topics.
            filename (str): Path to the output CSV file.

        Returns:
            None
        """
        topics_df = self.topics_to_dataframe(topics)
        topics_df.to_csv(filename, index=False)

    def plot_topic_distribution(self, document_topic_matrix: np.ndarray,
                                filename: str, title: str = "Topic distribution") -> None:
        """
        Creates and saves a horizontal bar chart showing the percentage
        distribution of dominant topics.

        Each document is assigned to the topic with the highest probability.

        Args:
            document_topic_matrix (np.ndarray):
                Matrix containing topic probabilities for each document.
                Shape:
                (documents, topics).
            filename (str): Path to save the plot.
            title (str): Plot title.

        Returns:
            None
        """
        dominant_topics = document_topic_matrix.argmax(axis=1)

        topic_counts = np.bincount(dominant_topics)
        topic_percentages = (topic_counts / len(dominant_topics) * 100)

        topic_labels = [f"Topic {i + 1}" 
                        for i in range(len(topic_percentages))
                        ]
        distribution = pd.DataFrame({
            "Topic": topic_labels,
            "Percentage": topic_percentages
            })
        distribution = distribution.sort_values(
            by="Percentage",
            ascending=True
        )

        plt.figure(figsize=(8, 5))
        bars = plt.barh(distribution["Topic"], distribution["Percentage"])
        plt.xlabel("Percentage of documents (%)")
        plt.ylabel("Topics")
        plt.title(title)

        # Add percentage labels
        for bar, value in zip(bars, distribution["Percentage"]):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center"
            )

        plt.xlim(0, max(distribution["Percentage"]) + 10)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def plot_topic_confidence(self, document_topic_matrix: np.ndarray,
                              filename: str, title: str = "Average topic confidence",
                              normalize: bool = False) -> None:
        """
        Creates and saves a bar chart showing the average
        confidence with which documents are assigned to each topic.

        Confidence is calculated as the average probability (or normalized
        topic weight) of the dominant topic across all documents assigned
        to that topic.

        Args:
            document_topic_matrix (np.ndarray): Matrix containing topic 
                                                probabilities for each document.
                Shape:(documents, topics).
            filename (str):Path to save the plot.
            title (str):Plot title.
            normalize (bool): If True, normalizes topic weights per document.
                              Recommended for NMF models.

        Returns:
            None
        """
        if normalize:
            row_sums = document_topic_matrix.sum(axis=1, keepdims=True)

            document_topic_matrix = np.divide(
                document_topic_matrix,
                row_sums,
                out=np.zeros_like(document_topic_matrix),
                where=row_sums != 0
            )

        dominant_topics = document_topic_matrix.argmax(axis=1)
        confidences = document_topic_matrix.max(axis=1)

        n_topics = document_topic_matrix.shape[1]

        average_confidence = []

        for topic in range(n_topics):
            topic_confidences = confidences[dominant_topics == topic]

            if len(topic_confidences) == 0:
                average_confidence.append(0.0)
            else:
                average_confidence.append(float(topic_confidences.mean()))

        distribution = pd.DataFrame({
            "Topic": [f"Topic {i + 1}" for i in range(n_topics)],
            "Confidence": average_confidence
        }).sort_values("Confidence")

        plt.figure(figsize=(8, 5))

        bars = plt.barh(
            distribution["Topic"],
            distribution["Confidence"].to_numpy()
        )

        plt.xlabel("Average confidence")
        plt.ylabel("Topics")
        plt.title(title)

        for bar, value in zip(bars, distribution["Confidence"]):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center"
            )

        plt.xlim(0, 1.05)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def get_dominant_topic(self, document_topic_matrix: np.ndarray) -> int:
        """
        Returns the most prevalent topic index.

        Each document is assigned to the topic
        with the highest probability.

        Args:
            document_topic_matrix (np.ndarray): Topic probability matrix.

        Returns:
            int: Dominant topic index.
        """
        document_topics = document_topic_matrix.argmax(axis=1)
        return int(np.bincount(document_topics).argmax())

    def plot_dominant_topic_keywords(self, document_topic_matrix: np.ndarray,
                                     texts: pd.Series, filename: str,
                                     n_words: int = 10,
                                     title: str = "Most frequent words in dominant topic") -> None:
        """
        Creates and saves a horizontal bar chart showing the most
        frequently occurring words in documents assigned to the
        dominant topic.

        The dominant topic is determined as the topic assigned to
        the largest number of documents.

        Args:
            document_topic_matrix (np.ndarray): Topic probability matrix.
            texts (pd.Series): Original preprocessed texts.
            filename (str): Output image path.
            n_words (int): Number of words to display.
        """
        dominant_topic = self.get_dominant_topic(document_topic_matrix)
        document_topics = (document_topic_matrix.argmax(axis=1))

        selected_texts = texts[
            document_topics == dominant_topic
        ]

        words = (" ".join(selected_texts).split())

        word_frequency = (pd.Series(words).value_counts().head(n_words).sort_values())


        plt.figure(figsize=(8,5))
        word_labels = word_frequency.index.tolist()
        word_counts = word_frequency.to_numpy(dtype=int)
        plt.barh(word_labels, word_counts)
        plt.xlabel("Number of occurrences")
        plt.ylabel("Words")
        plt.title(f"{title} (Topic {dominant_topic + 1})")
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
