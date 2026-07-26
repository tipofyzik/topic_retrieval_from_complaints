# Retrieving topics from financial customer complaints
1. Do not forget to download dataset and put it to the folder "data" inside the "code" folder: [Financial Consumer Complaints Dataset](https://www.kaggle.com/datasets/sherrytp/consumer-complaints)  
2. Results are on Kaggle: https://www.kaggle.com/code/tipofyzik/data-analysis-task  
In the section "output" the following graphics can be found for both combinations BoW+LDA and TF-IDF+NMF:
- The document percentage distribution across dominant topics,
- The bar chart showing the average confidence with which documents are assigned to each topic,
- The most frequently occurring n-grams in documents assigned to the most dominant topic,
- Top keywords directly from LDA/NMF model components. Shows terms with the highest topic weights,
- Topic overlap heatmaps, showing topic similarity,
- Graphics representing the elbow method for both LDA and NMF models. 


## LDA topic interpretation
| Topic | Interpretation |
|---|---|
| Topic 1 | **General credit reporting issues.** Complaints about credit reports, credit scores, credit bureaus, identity theft, and incorrect credit information. |
| Topic 2 | **Credit inquiry and account disputes.** Issues related to unauthorized inquiries, inaccurate account information, collection agencies, and credit report corrections. |
| Topic 3 | **FCRA compliance and legal disputes.** Complaints involving consumer reporting agencies, privacy rights, and violations of the Fair Credit Reporting Act. |
| Topic 4 | **Identity theft and fraudulent account handling.** Problems related to identity theft, blocked information, fraudulent accounts, and credit history corrections. |

Topic 1 – General credit reporting issues.
This topic is centered around credit reports, credit scores, and identity theft. It represents general complaints about inaccurate credit information, credit bureaus, and requests to correct or remove incorrect records.
Topic 2 – Credit inquiry and account disputes.
The keywords indicate disputes related to hard inquiries, account numbers, inaccurate information, collection agencies, and auto financing. Customers mainly report incorrect account details and unauthorized credit inquiries.
Topic 3 – Legal compliance with the Fair Credit Reporting Act (FCRA).
This topic contains terms associated with consumer reporting agencies, privacy rights, and the Fair Credit Reporting Act. It reflects complaints where consumers refer to legal obligations and alleged violations of credit reporting regulations.
Topic 4 – Identity theft and fraudulent account handling.
The dominant keywords relate to identity theft, fraudulent accounts, blocked information, and consumer reporting agencies. These complaints focus on resolving fraud-related records and preventing unauthorized accounts from affecting consumers' credit histories.
