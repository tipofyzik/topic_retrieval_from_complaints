import pandas as pd
import json
import time
import os 

from RawDataAnalyzer import RawDataAnalyzer
from DataPreprocessor import DataPreprocessor
from TopicOptimizer import TopicOptimizer
from BoWVectorizer import BoWVectorizer
from LDATopicModel import LDATopicModel
from TFIDFVectorizer import TFIDFVectorizer
from NMFTopicModel import NMFTopicModel
from TopicAnalyzer import TopicAnalyzer
# https://www.kaggle.com/datasets/sherrytp/consumer-complaints


with open("./config.json", "r") as f:
    config = json.load(f)

analyze = config["analyze"]
preprocess = config["preprocess"]
optimize_number_of_topics = config["optimize_number_of_topics"]
bow_and_lda = config["bow_and_lda"]
tfiidf_and_nmf = config["tfiidf_and_nmf"]
analyze_topics = config["analyze_topics"]

lda_topics = None
nmf_topics = None
lda_document_topics = None
nmf_document_topics = None


if __name__ == "__main__":
    raw_data_analyzer = RawDataAnalyzer()
    important_columns = ['Product', 'Sub-product', 'Issue', 'Sub-issue', 
                            'Consumer complaint narrative', 'Company']
    if analyze:
        dataset = pd.read_csv("./data/complaints.csv")
        dataset = dataset[important_columns]
        cleaned_data = raw_data_analyzer.clean_data(data = dataset, 
                                                    columns_to_clean = important_columns)
        raw_data_analyzer.analyze(data = cleaned_data)

        # Choosing a company with the most requests
        top_company = cleaned_data["Company"].value_counts().idxmax()
        data = cleaned_data[cleaned_data["Company"] == top_company][important_columns][:50000]
        data.to_csv("./data/cleaned_complaints.csv", index=False)
    else:
        data = pd.read_csv("./data/cleaned_complaints.csv")
        raw_data_analyzer.analyze(data = data)



    if preprocess:
        data_preprocessor = DataPreprocessor(batch_size=100, spacy_n_process=4)
        preprocessed_data = data_preprocessor.preprocess_data_batch(dataset = data, 
                                                                    column_to_preprocess = "Consumer complaint narrative")
        preprocessed_data = raw_data_analyzer.clean_data(data = preprocessed_data, 
                                                    columns_to_clean = important_columns)
        preprocessed_data.to_csv("./data/preprocessed_complaints.csv", index=False)
    else:
        preprocessed_data = pd.read_csv("./data/preprocessed_complaints.csv")
    # preprocessed_data = preprocessed_data[:1000]



    if bow_and_lda:
        start_time = time.time()

        bow_vectorizer = BoWVectorizer()
        bow_matrix = bow_vectorizer.fit_transform(
            texts = preprocessed_data["Consumer complaint narrative"]
        )

        if optimize_number_of_topics:
            optimizer = TopicOptimizer()
            optimizer.optimize_lda(
                bow_matrix,
                topic_range = list(range(1,15)),
                filename_plot="./results/lda_perplexity.png"
            )

        lda_topic_extractor = LDATopicModel(
            n_topics=5
        )
        lda_document_topics = lda_topic_extractor.fit_transform(
            bow_matrix=bow_matrix)
        lda_topics = lda_topic_extractor.get_topics(
            feature_names=bow_vectorizer.get_feature_names()
        )
        lda_topic_extractor.print_topics()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {int(execution_time // 3600)}h {int((execution_time % 3600) // 60)}m {int(execution_time % 60)}s")



    if tfiidf_and_nmf:
        start_time = time.time()

        tfidf_vectorizer = TFIDFVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            texts = preprocessed_data["Consumer complaint narrative"]
        )

        if optimize_number_of_topics:
            optimizer = TopicOptimizer()
            optimizer.optimize_nmf(
                tfidf_matrix=tfidf_matrix,
                topic_range=list(range(1, 15)),
                filename_plot="./results/nmf_reconstruction_error.png"
            )

        nmf_topic_extractor = NMFTopicModel(
            n_topics=5
        )
        nmf_document_topics = nmf_topic_extractor.fit_transform(
            tfidf_matrix=tfidf_matrix)
        nmf_topics = nmf_topic_extractor.get_topics(
            tfidf_vectorizer.get_feature_names()
        )
        nmf_topic_extractor.print_topics()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {int(execution_time // 3600)}h {int((execution_time % 3600) // 60)}m {int(execution_time % 60)}s")



    if analyze_topics:
        topic_analyzer = TopicAnalyzer()
        os.makedirs("./results", exist_ok = True)
        if lda_topics and lda_document_topics is not None:
            topic_analyzer.save_topics(
                topics=lda_topics,
                filename="./results/lda_topics.csv"
            )
            topic_analyzer.plot_topic_distribution(
                document_topic_matrix=lda_document_topics,
                title="Most prevalent LDA topics",
                filename="./results/lda_topic_distribution.png"
            )
            topic_analyzer.plot_topic_confidence(
                document_topic_matrix=lda_document_topics,
                filename="./results/lda_topic_confidence.png",
                title="LDA topic assignment confidence"
            )
            topic_analyzer.plot_dominant_topic_keywords(
                document_topic_matrix=lda_document_topics,
                texts=preprocessed_data["Consumer complaint narrative"],
                filename="./results/lda_topic_keywords.png"
            )

        if nmf_topics and nmf_document_topics is not None:
            topic_analyzer.save_topics(
                topics=nmf_topics,
                filename="./results/nmf_topics.csv"
            )
            topic_analyzer.plot_topic_distribution(
                document_topic_matrix=nmf_document_topics,
                title="Most prevalent NMF topics",
                filename="./results/nmf_topic_distribution.png"
            )
            topic_analyzer.plot_topic_confidence(
                document_topic_matrix=nmf_document_topics,
                filename="./results/nmf_topic_confidence.png",
                title="NMF topic assignment confidence",
                normalize=True
            )
            topic_analyzer.plot_dominant_topic_keywords(
                document_topic_matrix=nmf_document_topics,
                texts=preprocessed_data["Consumer complaint narrative"],
                filename="./results/nmf_topic_keywords.png"
            )
