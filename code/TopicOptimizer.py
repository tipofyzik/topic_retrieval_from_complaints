from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import pandas as pd



class TopicOptimizer:
    """
    Optimizes the number of topics for LDA and NMF models.

    LDA is evaluated using coherence score.
    NMF is evaluated using reconstruction error.
    """

    def __init__(self):
        """
        Initializes TopicOptimizer.
        """
        pass

    def optimize_lda(self, bow_matrix, topic_range: list[int],
                     filename_plot: str) -> None:

        results = []

        for n_topics in topic_range:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=0,
                learning_method="online"
            )
            lda.fit(bow_matrix)
            perplexity = lda.perplexity(bow_matrix)
            results.append(
                {
                    "n_topics": n_topics,
                    "perplexity": perplexity
                }
            )

        results_df = pd.DataFrame(results)
        plt.figure(figsize=(8,5))
        plt.plot(
            results_df["n_topics"],
            results_df["perplexity"],
            marker="o"
        )
        plt.xlabel("Number of topics")
        plt.ylabel("Perplexity")
        plt.title("LDA topic optimization")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            filename_plot,
            bbox_inches="tight"
        )
        plt.close()



    def optimize_nmf(self, tfidf_matrix, topic_range: list[int],
                     filename_plot: str) -> None:
        """
        Finds the optimal number of NMF topics using reconstruction error.

        Args:
            tfidf_matrix:
                TF-IDF document matrix.

            topic_range (list[int]):
                Range of topic numbers to test.

            filename_csv (str):
                Path to save results.

            filename_plot (str):
                Path to save plot.

        Returns:
            int:
                Selected number of topics.
        """
        results = []

        for n_topics in topic_range:
            nmf = NMF(
                n_components=n_topics,
                random_state=0,
                beta_loss="frobenius",
                init="nndsvda",
                max_iter=500
            )
            nmf.fit(tfidf_matrix)
            results.append(
                {
                    "n_topics": n_topics,
                    "reconstruction_error": nmf.reconstruction_err_
                }
            )

        results_df = pd.DataFrame(results)

        plt.figure(figsize=(8,5))
        plt.plot(
            results_df["n_topics"],
            results_df["reconstruction_error"],
            marker="o"
        )
        plt.xlabel("Number of topics")
        plt.ylabel("Reconstruction error")
        plt.title("NMF topic optimization")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            filename_plot,
            bbox_inches="tight"
        )
        plt.close()
