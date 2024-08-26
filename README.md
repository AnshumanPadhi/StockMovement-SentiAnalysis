# StockMovement-SentiAnalysis

## 1. _Introduction_

_Over time, the data have increased exponentially. Digitalization in all areas generates a large number of data per second. Today, this large amount of data is an unstructured type, i.e. there are no predefined formats.The text is the most generated type of unstructured data and  Whether it’s a customer’s product review or a news headline, everything is based mainly on text. Companies and enterprises are using these data to make strategic decisions._

_These decisions may include the modification of existing policies, the creation of new products, or taking strategic decisions to choose the right set of Stocks, etc. Since the generated text is human-readable, it cannot be understood by machines. Therefore, to make it machine-friendly, the concept of natural language processing was introduced._ 

_To understand the emotion behind a text, NLP technique is used to find patterns & to generate sentiment scores to define whether it is a positive text or a negative text._

_The Objective of the Project is to identify the sentiments through online news headlines whether the mood is positive about a specific stock or negative/ no reaction which further identifies the Stock Market Performance for the Stock._ 

_In the dataset, we have top headlines for specific companies. Based on these headlines there are labels of values zero and one. Zero basically means that stock price will have a negative impact and One means that stock price will have a positive impact._

_About the problem and the dataset used._

- _The data set in consideration is a combination of the world news and stock price shifts._

- _News headline data ranges from Apr’2022 to July24  from Finviz_

- _Stock data from Apr' 22 to July 24 was scrapped from Yahoo finance._

- _There are multiple rows & one column of top news headlines for each day in the data frame._

- _Sentiment score is represented on a scale of -1 to 1, with the low end of the scale indicating negative responses and the high end of the scale indicating positive responses._

**Objectives:**

- _Predict the Stock movement (UP/Down/No-Change) of a stock based on the associated News Headlines._

- _Evaluate the performance of the predictive models._

- _Understand and get accustomed to a Data Science Pipeline_


## _Data Collection and Preprocessing_

### _Data Sources_

- _Historical stock prices from NSE API_

- _News headlines from Finviz ( NASDAQ)_


### _DataSet Details_

_The Stock which we would be aiming here are NASDAQ & NSE Stocks._


#### **_Key details of the dataset are as follows:_**

**_Stock Symbols_**_: The dataset includes 11 different stock symbols, encompassing a wide range of companies from various sectors. This diversity ensures that the analysis can cover a broad spectrum of the stock market._

**_Date Range:_** _The data covers a period of twenty seven months, from Apr’2022 to July24. This period provides a sufficient time frame to analyze recent trends and movements in the stock market._

**_Data Fields:_** _For each stock symbol, the dataset includes the following fields:_

**_News Headline Data_** 

_Index:Unique Sequence_ 

_Date & Time : MM-DD-YY HH:MM format_

_Headlines: News headlines text_ 

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdk3AxucoyAuA0MffL8ZO5RpHpxb0L6fGPln7q4Dzkj205tBwIVXt1IID7DJgFwK20PAq42XUPasuxndPND1Xwfs0SadwZy1ot0mlfBxZO8Cnvkup8BJ3leKYN8ZT2gVMvmDlSkaOEwXeEnxlbhlOmt4JY?key=iDsHD7FiYhNJdhYVK1OmNw)__

**_Stock Market Data:_** 

_Date: The trading date._

_Open: The opening price of the stock on that date._

_High: The highest price reached by the stock on that date._

_Low: The lowest price reached by the stock on that date._

_Close: The closing price of the stock on that date._

_Adj Close: The adjusted closing price, accounting for corporate actions like dividends and stock splits._

_Volume: The number of shares traded on that date._

_Data Format: The data is stored in a CSV file format,_ 


### **_Data Preprocessing_**

_1. Lowercasing: Convert all text to lowercase to ensure uniformity and avoid the duplication of words due to capitalization._

_2. Tokenization: Divide the text into units of meaningful data called tokens. This step is essential as it allows the model to analyze important segments that make sense together.._

_3. Removing Punctuation: Remove punctuation marks such as commas, periods, exclamation marks, etc., as they do not add much value for sentiment analysis, rather they become a source of noise as their usage are inconsistent and varies a lot from person to person._

_4. Removing Stopwords: Remove common words that do not contribute much to sentiment analysis, such as "the," "a," "is," etc. Due to their high frequency in the language, they skew the results as they become some of the most significant words in the model which is not the case._

_5. Stemming and Lemmatization: Stemming is the process of reducing a word length by removing characters from the end. A good example is removing ‘d’ from ‘bathed’. But in most scenarios, we tend to overstem so ‘university’ can go to ‘univers’ which is not sensible and doesn’t mean anything. A better alternative is Lemmatization. In Lemmatization, we change a word to its root word like ‘bathing’ to ‘bathe’. Stemming or Lemmatization is important as this basically works like dimensionality reduction._ 

_6. Handling Contractions: Expand contractions to ensure that words are in their full form. Words like ‘can’t’ or ‘don’t’ not only have special characters, it makes the text inconsistent. Expanding contractions brings uniformity to data, reduction to ambiguity and improves tokenization as well._

_7. Removing Numbers and Special Characters: Remove numbers and special characters, as they generally don't contribute much to sentiment analysis._

_8. Handling Emojis and Emoticons: Convert emojis and emoticons to meaningful words to include their sentiments in the analysis. A common approach is to tag them with associated emotions that they represent._ 

_9. Part-of-Speech (POS) Tagging: Identify the part of speech of each word (noun, verb, adjective, etc.) for better context analysis._

_10. Spell Checking and Correction: Identify and correct spelling errors to improve the accuracy of sentiment analysis._

## 2. _Technique Used / Methodology_

**1)   Sequence Diagram**

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcX5aNTTV0IKzoYAUcTlK7NPUx_gWmxurc3W4gvWm4MnHKXfyAIWld5i4Yf7nBu45nUkdVtbGRltPg45KMZ-81GxsjvucfTQbNOKKII9pAAZSVSLkJtS85YocmMdSbTGgnaArxIAsCWH7_XU195DBSSK5yG?key=iDsHD7FiYhNJdhYVK1OmNw)


### **B)  Exploratory Data Analysis (EDA)**

_The dataset comprises news headlines for various stock symbols from the Nifty 50 list, with details on publication times and corresponding stock symbols. An initial examination reveals that the headlines vary significantly in length, ranging from 22 to 307 characters and containing between 4 and 38 words. This variation suggests a mix of brief updates and detailed reports. A bar chart of the number of articles per stock symbol shows uniform media coverage for most stocks, with nearly 100 articles each, except for ZCAR, which has only 17 articles._

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfOoXNCUQnyzvvpmcgkIscmd5kAiPMsw35QX3t32c05XrMKRZJT5yRw8ZD6MHsj_7tcHyKtBrJe1sfgUndNeEIsrUQCb7CEX8XqsAv_cuxchUdu6doFv8GYKc5zyWIqb0kfD1a-nOXFJWsOu9fywpfjfWbR?key=iDsHD7FiYhNJdhYVK1OmNw)__

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcBv12fZo7-FqzdNZ_SCyAEsG-Oe1jgG9VMui885CbZCEZwTOjEMSQVVGMNBzUpoyUr0Qz54VAz7sbKa0iQydZFhqLujP7-41fXCP1-Y1tlzvCMjBkQyhJ0hyFFTEgd9BuKsgiwrSsm4XYNoBPkKftyrvc?key=iDsHD7FiYhNJdhYVK1OmNw)

_A word cloud visualization highlights the most frequently occurring terms, such as "India," "stock," "Infosys," "Wipro," "Dr Reddy," and "Yatra Online," indicating a strong focus on these entities in the news coverage._ 


### **_C)_** __**Techniques Used**

1. _Web Scraping with Python Language_ 

_Step 1: Install Required Libraries_

_Step 2: Import Libraries_

_Step 3: Choose a Financial News Website_

_Step 4: Send a GET Request to the Website_

_Step 5: Parse the HTML Content_

_Step 6: Locate the News Articles Inspect the website’s HTML structure to identify the elements containing the news articles you want to scrape. Typically, you’ll look for HTML tags like \<div>, \<ul>, or \<li> with specific classes or attributes that wrap the news articles._

_Step 7: Extract News Article Data- This code snippet locates all the news articles on the page and extracts their headlines, summaries, and links._

_Step 8: Data Storage_ 

Output : 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXd4hElWNwJPLMNizcYaHm9k9gF8u40nkbRPSNhIVdTCqBd7UN_JbfbM6feEoVhGP8E2TgEOaqZ6PKbMZ26an54kVMDmFqZTQYHst4y-GybOC9sH_T28ag9Dj0YHDWl9OZfEVFcAKGSQlSaF4wLD3JUC9MA?key=iDsHD7FiYhNJdhYVK1OmNw)

  

Code Snippet : 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXekBPgx3gtx1wGCof3solPlIyzxo9M1SjkY-0F7woqBO5atwE_Sjcov0fwDbCKHhphljWSRUID3FXev1qGgs2OEKqmay9ReWYdM4weVAOgPm7RtnEzulgbtnN28KXOqsjcg_U5gh5Wpi5Z6zH1tR7LZ1yNW?key=iDsHD7FiYhNJdhYVK1OmNw)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfvehnapOn3Ix-ErD0hvoZcqKDn8pz3iZWCaeg3znaKRmlJOOK1MkNmBmgh4rGnlifKXtG_kMWqcvKtXTLVuXaDk9r46MbMQue4Q0QCinZ6U9MmtBGQwEEhe-bOfoua35Obcu-npb2ZiEv993knSODHaJ8?key=iDsHD7FiYhNJdhYVK1OmNw)


## 2.  _VADER SentimentIntensityAnalyser to calculate Sentiment Score_

**_VADER_** _is a long-form for Valence Aware Dictionary and sEntiment Reasoner, a rule-based sentiment analysis tool.VADER calculates text emotions and determines whether the text is positive, neutral or negative._

_In this method, the Sentiment Intensity Analyser uses the_ **_VADER Lexicon_**_.This analyzer calculates text sentiment and produces four different classes of output scores:_ **_positive, negative, neutral, and compound._** 

_A compound score is the aggregate of the score of a word, or precisely, the sum of all words in the lexicon, normalized between -1 and 1._

_Note: It is not recommended to use general text preprocessing techniques prior to calculation because this may affect VADER results._

_Step 1 : Importing Libraries including SentimentIntensityAnalyzer_

_Step 2 : Reading Data_

_Step 3 : VADER Sentiment Scoring_

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf54tMGFWceh50htmMzXYktii9eRzasOkbruNsEngDOwZYO5zw3xZSvaHtc6W_Ppa5ls8BNPSVD38Ha5_YoXc-Lnsb01OltSh6gOB0UMLeWo973rBd79ZeaD6FqZpeJq87t28EO7mgKBW3NhEWil8eKT5g?key=iDsHD7FiYhNJdhYVK1OmNw)__

____

_Step 4 : Run the polarity score on the entire dataset_

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdOQrhfvv4XoT78Px0bYica8m3s5X9GmkOx6BRtQeBaMcuKrqkbowoFz3GoNbtGU-W5oMmO_BDn6SYhBEmcH0ZtvGm7p2tBEiz-WvLdJM5wd5xtX8-RwO0j_-HOBIuG4NhlpxveMdVrbVFTKh1Zu-RRESs?key=iDsHD7FiYhNJdhYVK1OmNw)****

 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXejYs086KpVF_u4Y2tuVbMEAC7d5PXY2OJDjPUGqfLFYRloSCFGixCUGAr75Atd354H0KrAUt6Q6sKC9h8lsI5T4gw2n9IpyDQ2I1S5k4CZpInzthQvprzzmdTmgg_tXZCRC4kCDIV93U0Qw_W39Y7lVD8L?key=iDsHD7FiYhNJdhYVK1OmNw)

____

_Code Snippet :_  

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfqrWVDrYdmQA9IQs1wEp5_JnVX8qgcw1tooV2E4HNhfLGmyMtL-t9x8cWCPIkVfYopkfqsfFpCuG6cZmqVeLgCm8jCYBxvL4FQGH546jh2O9yfagZueAYaTXgUsu0obFVd8StqasQDDSfenWzAFlV3dKb3?key=iDsHD7FiYhNJdhYVK1OmNw)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXc5MbJpJhCo6iWBz9csxYDaG4l11X8yF9GgA5X3R3IDafs0_I-O19zWjN4Lq1xeIcjwRtCR9mZy7esEzKC-HrHfdUpx3XeMyz3PJt5o3AwflWf-obpm4Ars9GayBQ-YX2iCCABJdXoFHWriTbl8LtP5FNzq?key=iDsHD7FiYhNJdhYVK1OmNw)

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcdax9SNLhpTN31JMRj8v-EnRoi8_RyhVMrVikzudCevGGbtIOll3Y971FJOgDwXgPg8al13D-yEatxHI5lFFR5fPejSuPP3aD-jHN5g3TAZ1sbE2ptweaOXDDWj-36t8dUbBqa3FbV_PvUpZVXMkcG6gnX?key=iDsHD7FiYhNJdhYVK1OmNw)__

**3. _roBERTa_** _Pre trained model to calculate sentiment score._

**_roBERTa_** _(short for “Robustly Optimized BERT Approach”) is a variant of the BERT (Bidirectional Encoder Representations from Transformers) model, which was developed by researchers at Facebook AI._ 

_roBERTa was trained on a dataset of 160GB of text, which is more than 10 times larger than the dataset used to train BERT.Additionally, RoBERTa uses a dynamic masking technique during training that helps the model learn more robust and generalizable representations of words._

___Step 1 : Importing Libraries_ 

_Step 2 : Reading Data_

_Step 3 : AutoTokenization_ 

_Step 4: Roberta Score_

_Step 5: Run the polarity score on the entire dataset_

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe-g6DcYxbCGoaAWsHLOP636V_0CLC0d8rcInSBrtVa0t4qiTwY8vbNXEGtC_tb_KiMeNC1mTsYg8JHfjrD03c2eFepKqIKQ5QMzP8HX7BAS1R9NDK9SWzjkua151bibpHu_IAsS7lII8e8q7EB8HxP8m4?key=iDsHD7FiYhNJdhYVK1OmNw)__

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf4wwoclfowOVx51fmwfTzufv3G-cJ_YYx1esSMEgH7SwGPEtV7DI4RA7IFRpDciSvGPvD7mFDn8q05kBEk6SQYULdM9okzSNkfFQSOxEdLB_xA7V0-zO8ttLmVRcpkpwyHs90FDk2P0pCDQib8CzheOWfM?key=iDsHD7FiYhNJdhYVK1OmNw)__

 __

_Code Snippet:_![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXedluPOTr584Y77ssXTM0uiPkBc66oBQdITbcUDPd9qWVTCc53BdOtFEIHmc8BEqjs2qwetO1qrVAYytfg_9RIX7XC4-J3xJwhWPolRJ3YGmi6-P2ympNSsAvhG7460GhzXf6yqa0mI_KGo7ZDy2BqnBAMS?key=iDsHD7FiYhNJdhYVK1OmNw)__

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdXSSEd1tGBNzYy7FVEzd0jaJC2FpuFwcGscr0rfe-SmrTKDZ5kmLlDSFRN6c4zxW1Pl8iPMJk3Cpl03GL0xVk7HM8h0YaI0nonOtrZhr-QHWkzYXpjGApzEpnHvrpvMhaCiphDsTgIcI_DWF4bSVbfQ-G8?key=iDsHD7FiYhNJdhYVK1OmNw)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdqhjEcZ87ZB9AHwc1zXAWSUaNcytJWZ5lS8lBYsOfrc5i4pit_QbRwdi8jQvAvTF_eAqDVA8isykAVoOerY21sxf8yGz8MciZ8ZrbdpSHCpU7ZX91Glxo6tTgewDVaM3bPx37e1Jy-gz8yYkqpAEJRmF8F?key=iDsHD7FiYhNJdhYVK1OmNw)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeOhkPH-4_6U0hY3LSkcX_8MZADneQc6F6NwkZi4ZDOjWiiB5YAqWTNvSpegB_mLnpm4Mxd11lFmPWBWZo4CJmvS1ulJP5w7ezgm94_ze_vU4a8P_lOUA1PJXM3WbTwFFWgiZrzR_s9ZjjYiZQV0gQAlyV-?key=iDsHD7FiYhNJdhYVK1OmNw)__

## 3.  _Results_

_Sentiment Score through VADER and roBERTa model._

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXff-0ti13nmyM8Ca3QMy6a1LhGq7xHHzbjLUm0Gz3TPl_VGQUvmpsvjMQT9poMwQtRtuXWGYSl0RHapvYaMP0LwoYrwQOcFew2fPc_Jp4gb3XRfXJNwBqU8mY3Icd_6guN8ERZZdnhYSQl8Q2yH9s7XK6bu?key=iDsHD7FiYhNJdhYVK1OmNw)

_Comparison between VADER and roBERTa_

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcjJBG7z7rjXlaAIbVJJs8hkBqoT0bT1T4H1pBNM0gRPv3zoj5-nVCDSwFEpYnG5qZwx02XN09zLXwkikgFU1FkhsrCiFOTr1_rFz_kuWGJt4ovkqobS8eQxS09AL5Ez0xg33wG9bJlDAItPLVVkp64KQb4?key=iDsHD7FiYhNJdhYVK1OmNw)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdCpEYHAzq2mm1WNz6wNINJcB1lwqdT3pur_ssZQtODbAGJt2EwfxlkNwTpfZU6B6qom5CiiQPh0rlpgFQoi68wII3aSH1jwVAOwDifSiFsFhMD1OuwFjR8Muc02SdY-W80vSYbO12cUrMs3ZbGs4bqtDNP?key=iDsHD7FiYhNJdhYVK1OmNw)

## 4.  _Discussion of the results_

_Insight_

_VADER (lexicon-based)_ 

_VADER model has its own dictionary (lexicons) of words or emojis with positive or negative weights. The algorithm count the number of positive and negative words in the given text. If the number of positives is more than the negatives, they return a positive sentiment. If both are equal, they return a neutral sentiment. Rules or dictionaries of words can be customized._ 

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXewCMW-Yt8hBZ5KG3L-GgQotYG8PW5PCRq_3LFECK120-3DO54lHmtz9LOHlpcdfQDly4WtE4cFEe0Sx3fEB_DGp0aicwIJvOZ11-h6EB-wfFsQnSgkx_L8xRoeMcwQoquzQRC1s5eIpfWhj12zVGKBssaI?key=iDsHD7FiYhNJdhYVK1OmNw)__

_Model Performance_

_In many cases ,it has been observed that the VADER model is not able to find many positive or negative words in the given text and hence you can see the neutral sentiment is more in comparison with positive or negative._

_The VADER model is easy to implement & quite faster in giving sentiment results._ 

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfSUqL7Jxfzcwfcca8oGJVFIiMsIud9KPOo9xS3UR9q76XU-GRcj2Z1xl_7BOqZBEN4P_v7yYkWEjkSM1L1iUA8Mtwu-JkROE_wjZLbVfWwETuLVWLdDQ_C5CEfvG5gcqw83Qob3R3On1RRJLmIJmhXRco?key=iDsHD7FiYhNJdhYVK1OmNw)__

_The Pie Chart shows how many of them are labeled negative, neutral, and positive. According to the rule-based VADER model, positive is 10% and negative is 5% whereas neutral is contributing a big chunk of 85%._

_VADER model completed the task of 1011 news headline in 0.355131 seconds_

_Insight_

roBERTa Pre Trained Model

_roBERTA is a deep learning-based algorithm which uses the TRANSFORMER package. This model was pre trained with 124 million tweets and fine tuned for sentiment analysis. The models do not require any training, which means they can be directly applied to textual data._

_Model Performance_

 _RoBERTa implementation is a bit more complicated, particularly because it does not provide a compound score and the same needs to be calculated manually._

_roBERTa, in reality catches the deep meaning of a text rather than individual words, Also in many cases it is able to give the sentiment score of negative or positive where VADER was giving neutral score._

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeU8sJ7WaxokIKxUvEBAuFmTVtiIY9Io2y7Y5H55WTodJCKjaQ1nc2zmHr3Rv695djK0gYtE7vxljtw0Z24Q4s_4t_y536LS-FUVLBZHvY6IPdu-85HjYkh57EQTK-GB0z0QFPBwwLtW1Xc_G6sqKPMVNtH?key=iDsHD7FiYhNJdhYVK1OmNw)__

_The Pie Chart shows how many of them are labeled negative, neutral, and positive. According to the deep learning based roBERTA model, positive is 27% and negative is 7% whereas neutral is contributing 66%._

_roBERTA is completed the task of 1011 news headline in 156.3862 seconds_

_Challenges_

1. _Social media do not follow grammar rules, and contain many slang words and emojis, all of these make them more complicated & difficult to identify the sentiment without any pretrained rule._

2. _Though roBERTA is giving better results of the sentiment score but time taken by the algorithm is way too high at 156.3 seconds versus 0.355 seconds by VADER model._ 

3. _Finding News headlines : Accuracy and speed of associating a Stock with the relevant News Headline._

4. _Price movement Prediction : Accuracy of the (UP/Down/No-Change) classifier._

## 5)  _Conclusion_

**_Summary_**

_In this activity , we learned about two different ways of calculating the Sentiment Scores for given 1011 news headlines related to these 13 stocks ._ 

_The first method was rule-based(lexicons) method and the score is evaluated by itself and the second method was the roBERTA pre-trained model which is trained on a dataset of 160GB of text. So you don't need to pre trained it again ._

_Each of these methods has its own pros and cons. We even tried creating our own dictionary of words to calculate sentiment score but the results were not accurate._

_VADER no doubt is very fast in completing the task in 0.355 seconds whereas roBERTA takes more time in completing the task of 1011 news headlines (156 seconds) but in the end what we need is to focus on the accuracy of sentiment scores, where roBERTA is way ahead in identifying the emotions behind the sentence and not just only the words._ 

**_Future Work_**

1. _Incorporating more features (e.g., sentiment analysis from news articles)._

2. _Experimenting with different machine learning models (e.g., Transformer models)._

3. _Enhancing model robustness to adapt to rapidly changing market conditions._

4. _Evaluation_

   1. _Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)._

   2. _Comparing the model's predictions with actual stock prices._

## 6)  _Appendix_ 


#### _Output of Data Download_

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcw2lOo7wXfT97VzntSmsC1hb7GzytdWxyxVXR6YLCeWl4ru-vViil1xMouYK6tbdCVOxTZUWDrrlWsrlaZ2z_6c5QqLHYw9Y_Xa0aJbSsXgy2Wl1bMC0nm0l9LxvEsDWLMKVouffhLmzsLIyzn84svAoQ?key=iDsHD7FiYhNJdhYVK1OmNw)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe8vlE4RGTbjMBLjfMK9O_w84xWOM7MF4W-m5WFryP8Yox8B3x8RFVFmnjUMLPBjkrQwEg8KrD5KiDIpK2jFMrVguNxRAYnV6R3xN6NhYRBZo_7pVAZjNrQymzBUOX_noa33vC8YKvHjmzcNDlJLuQsAIcO?key=iDsHD7FiYhNJdhYVK1OmNw)
