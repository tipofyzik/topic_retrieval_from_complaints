# Retrieving topics from financial customer complaints
1. Do not forget to download dataset and put it to the folder "data" inside the "code" folder: [Financial Consumer Complaints Dataset](https://www.kaggle.com/datasets/sherrytp/consumer-complaints)  
2. Results are on Kaggle: https://www.kaggle.com/code/tipofyzik/data-analysis-task  
In the section "output" the following graphics can be found for both combinations BoW+LDA and TF-IDF+NMF:
- The document percentage distribution across dominant topics
- The bar chart showing the average confidence with which documents are assigned to each topic
- The most frequently occurring n-grams in documents assigned to the most dominant topic
- Top keywords directly from LDA/NMF model components. Shows terms with the highest topic weights
    - Additionally, tables with topic keywords can be found there
- Topic overlap heatmaps, showing topic similarity
- Graphics representing the elbow method for both LDA and NMF models

# How to use
1. Download the code folder and install all the requirements.
2. Download the data as mentioned in the previous section, point 1.
3. The following JSON configuration enables data analysis, preprocessing, LDA/NMF topic modeling, and result visualization. For further information, read the **config_explanation.txt**.  
```json
{
    "analyze": 1,
    "preprocess": 1,
    "optimize_number_of_topics": 0,
    "bow_and_lda": 1,
    "tfiidf_and_nmf": 1,
    "analyze_topics": 1
}
```
4. Parameters, such as ngram range, number of topics, etc. can be found in the **main.py** file. Default parameters are
```python
ngram_range = (2, 3)
n_words = 15
n_topics = 4
```
5. To run the system just open the **main.py** file after downloading everything and run it. For the first time, run the program with the parameters setups mentioned above.

## BoW + LDA results and topic interpretation
<table>
  <tr>
    <td width="33%">
      <img width="790" height="490" alt="lda_perplexity" src="https://github.com/user-attachments/assets/d8d41f85-2f2b-46bc-aa03-578969e46d8b" />
    </td>
    <td width="33%">
      <img width="790" height="490" alt="lda_topic_distribution" src="https://github.com/user-attachments/assets/77a322a9-70b7-4361-bf12-4f98f7a8abe1" />
    </td>
    <td width="33%">
      <img width="790" height="490" alt="lda_topic_confidence" src="https://github.com/user-attachments/assets/b8938e8a-b4bd-47ef-b8bb-15ee799a0500" />
    </td>
  </tr>
</table>



| Topic | Interpretation |
|---|---|
| Topic 1 | **General credit reporting issues.** This topic is centered around credit reports, credit scores, and identity theft. It represents general complaints about inaccurate credit information, credit bureaus, and requests to correct or remove incorrect records. |
| Topic 2 | **Credit inquiry and account disputes.** The keywords indicate disputes related to hard inquiries, account numbers, inaccurate information, collection agencies, and auto financing. Customers mainly report incorrect account details and unauthorized credit inquiries. |
| Topic 3 | **FCRA compliance and legal disputes.** This topic contains terms associated with consumer reporting agencies, privacy rights, and the Fair Credit Reporting Act. It reflects complaints where consumers refer to legal obligations and alleged violations of credit reporting regulations. |
| Topic 4 | **Identity theft and fraudulent account handling.** The dominant keywords relate to identity theft, fraudulent accounts, blocked information, and consumer reporting agencies. These complaints focus on resolving fraud-related records and preventing unauthorized accounts from affecting consumers' credit histories. |

## TF-IDF + NMF results and topic interpretation

## NMF topic interpretation
<table>
  <tr>
    <td width="33%">
      <img width="790" height="490" alt="nmf_reconstruction_error" src="https://github.com/user-attachments/assets/ef63fccc-3fbb-488b-b628-4482e2e2fcd9" />
    </td>
    <td width="33%">
      <img width="790" height="490" alt="nmf_topic_distribution" src="https://github.com/user-attachments/assets/0b9e70e3-0b8e-480e-a18a-e17dcfbb4f39" />
    </td>
    <td width="33%">
      <img width="790" height="490" alt="nmf_topic_confidence" src="https://github.com/user-attachments/assets/74c5641d-20e5-474d-a90e-2d7136734604" />
    </td>
  </tr>
</table>


| Topic | Interpretation |
|---|---|
| Topic 1 | **Consumer reporting agency compliance and account disputes.** This topic focuses on interactions with consumer reporting agencies, account information updates, written instructions, and compliance-related issues under fair credit reporting regulations. |
| Topic 2 | **General credit reporting and identity theft issues.** This topic represents common complaints about credit reports, credit bureaus, identity theft, inaccurate information, credit scores, and requests to remove incorrect records. |
| Topic 3 | **Credit score impact and consumer frustration.** This topic captures complaints about unexpected credit score drops, repayment history, credit utilization, and dissatisfaction with companies handling consumer protection issues. |
| Topic 4 | **Failure to investigate and credit record correction requests.** This topic describes disputes where consumers request investigations, removal of incorrect items, account deletion, and possible legal action due to unresolved credit report issues. |
