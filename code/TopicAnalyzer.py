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

    def plot_topic_keywords(
        self,
        filename: str,
        mode: str,
        feature_names: list[str],
        n_words: int = 10,
        title: str = "Topic keywords",
        document_topic_matrix=None,
        document_term_matrix=None,
        model_components=None,
        topic_index=None
    ) -> None:
        """
        Creates and saves a horizontal bar chart showing topic keywords.

        Two modes are supported:

        - "model":
            Extracts top keywords directly from LDA/NMF model components.
            Shows terms with the highest topic weights.

        - "frequency":
            Extracts the most frequent terms from documents assigned
            to a dominant topic. Supports n-grams generated by the vectorizer.

        Args:
            filename (str):
                Path where the plot will be saved.

            mode (str):
                Keyword extraction method:
                "model" or "frequency".

            feature_names (list[str]):
                Vocabulary terms generated by CountVectorizer or TfidfVectorizer.

            n_words (int):
                Number of keywords displayed.

            title (str):
                Plot title.

            document_topic_matrix (np.ndarray):
                Matrix containing document-topic probabilities/weights.
                Required for "frequency" mode.

            document_term_matrix:
                Document-term matrix created by vectorizer.
                Required for "frequency" mode.

            model_components:
                Topic components from LDA or NMF model.
                Required for "model" mode.
                Shape:
                (n_topics, n_features)

            topic_index (int):
                Topic number to visualize.
                Required for "model" mode.
                If None, dominant topic is selected.

        Returns:
            None
        """

        if mode not in ["model", "frequency"]:
            raise ValueError(
                "mode must be either 'model' or 'frequency'"
            )

        # ----------------------------
        # Model based keywords (LDA/NMF)
        # ----------------------------
        if mode == "model":

            if model_components is None:
                raise ValueError(
                    "model_components required for model mode"
                )

            # Select dominant topic if topic_index is not provided
            if topic_index is None:
                if document_topic_matrix is None:
                    raise ValueError(
                        "document_topic_matrix required when topic_index is None"
                    )

                topic_index = self.get_dominant_topic(
                    document_topic_matrix
                )

            topic_weights = model_components[topic_index]

            keyword_indices = (
                topic_weights
                .argsort()[-n_words:]
            )

            keywords = [
                feature_names[i]
                for i in keyword_indices
            ]

            values = [
                topic_weights[i]
                for i in keyword_indices
            ]

            data = pd.Series(
                values,
                index=keywords
            ).sort_values()

        # ---------------------------------
        # Frequency based keywords
        # ---------------------------------
        else:

            if document_topic_matrix is None:
                raise ValueError(
                    "document_topic_matrix required for frequency mode"
                )

            if document_term_matrix is None:
                raise ValueError(
                    "document_term_matrix required for frequency mode"
                )

            dominant_topic = self.get_dominant_topic(
                document_topic_matrix
            )

            document_topics = (
                document_topic_matrix.argmax(axis=1)
            )

            selected_matrix = document_term_matrix[
                document_topics == dominant_topic
            ]

            term_counts = np.asarray(
                selected_matrix.sum(axis=0)
            ).ravel()

            data = (
                pd.Series(
                    term_counts,
                    index=feature_names
                )
                .sort_values(
                    ascending=False
                )
                .head(n_words)
                .sort_values()
            )

            topic_index = dominant_topic


        # ----------------------------
        # Plot
        # ----------------------------

        plt.figure(figsize=(8, 5))

        word_labels = data.index.tolist()
        word_counts = data.to_numpy(dtype=int)
        bars = plt.barh(word_labels, word_counts)
        # bars = plt.barh(
        #     data.index,
        #     data.values
        # )

        plt.xlabel(
            "Weight" if mode == "model"
            else "Occurrences"
        )

        plt.ylabel("Terms")

        plt.title(
            f"{title} (Topic {topic_index + 1})"
        )

        max_value = max(data.values)
        for bar, value in zip(bars, data.values):
            plt.text(
                bar.get_width() + max_value * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center"
            )

        plt.xlim(
            0,
            max_value * 1.15
        )

        plt.tight_layout()

        plt.savefig(
            filename,
            bbox_inches="tight"
        )

        plt.close()

    def plot_topic_overlap_heatmap(
            self,
            topics: dict[str, list[str]],
            filename: str,
            title: str = "Topic keyword overlap"
        ) -> None:
        """
        Creates a normalized heatmap showing topic similarity.

        Values are calculated using Jaccard similarity:
        intersection / union.

        0.0 = no overlap
        1.0 = identical keyword sets
        """

        import seaborn as sns


        topic_names = list(topics.keys())

        topic_sets = {
            topic: set(words)
            for topic, words in topics.items()
        }


        similarity_matrix = []

        for topic_a in topic_names:

            row = []

            for topic_b in topic_names:

                intersection = len(
                    topic_sets[topic_a]
                    .intersection(topic_sets[topic_b])
                )

                union = len(
                    topic_sets[topic_a]
                    .union(topic_sets[topic_b])
                )

                if union == 0:
                    similarity = 0
                else:
                    similarity = intersection / union

                row.append(similarity)

            similarity_matrix.append(row)


        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=topic_names,
            columns=topic_names
        )


        plt.figure(figsize=(8, 6))


        sns.heatmap(
            similarity_df,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={
                "label": "Jaccard similarity"
            }
        )


        plt.title(title)
        plt.xlabel("Topics")
        plt.ylabel("Topics")

        plt.tight_layout()

        plt.savefig(
            filename,
            bbox_inches="tight",
            dpi=300
        )

        plt.close()