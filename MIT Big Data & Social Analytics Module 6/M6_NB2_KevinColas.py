
# coding: utf-8

# <div align="right">Python 2.7 Jupyter Notebook</div>
# 
# # Privacy
# <br>
# <div class="alert alert-warning">
# <b>This notebook should be opened and completed by students completing both the technical and non-technical tracks of this course.</b>
# </div>
# 
# ### Your completion of the notebook exercises will be graded based on your ability to:
# 
# > **Understand**: Does your pseudo-code and/or comments show evidence that you recall and understand technical concepts?
# 
# > **Apply**: Are you able to execute code, using the supplied examples, that perform the required functionality on supplied or generated data sets? 
# 
# # Notebook introduction
# 
# In Video 2, Cameron Kerry indicated that the law lags too far behind technology to answer many of the hard questions. He then went on to elaborate that in many cases the question becomes not just what you must do, but rather, what you *should* do in order to establish and maintain a trust relationship.
# 
# Sharing data collected about individuals between entities poses a risk to privacy and trust, and is regulated in most parts of the world. The European Union recently passed the [General Data Protection Regulation (GDPR)](http://www.allenovery.com/SiteCollectionDocuments/Radical%20changes%20to%20European%20data%20protection%20legislation.pdf), which addresses the treatment of personal information, as well as the rights of the individuals whose information has been collected. Penalties are based on a tiered approach, and some infringements can result in fines of up to 4% of annual worldwide turnover, and €20 million. It is often the case that the information to be shared needs to be anonymous. In some cases, ensuring anonymity removes the data from the jurisdiction of certain laws. Application of the laws is a complex task that needs to be carefully implemented to ensure compliance. Please refer to [this](http://www.fieldfisher.com/publications/2016/03/the-new-eu-data-protection-regime-from-an-hr-perspective#sthash.cmQBwBnz.rUXLBjmR.dpbs) post for additional context.
# 
# Pseudonymization - the removal of direct identifiers - is the first step to anonymize data. This is achieved by removing direct identifiers such as names, surnames, social insurance numbers, and phone numbers; or by replacing them with random or hashed (and salted, see the NYC taxi cab example) values.
# 
# However, cases like [William Weld's](http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2076397) show that pseudonymization is not sufficient to prevent the re-identification of individuals in pseudonymized datasets. In 1990, the Massachusetts Group Insurance Commission (GIC) released hospital data to researchers for the purpose of improving healthcare and controlling costs. At the time GIC released the data, William Weld, then Governor of Massachusetts, assured the public that GIC had protected patient privacy by deleting identifiers. 
# 
# > **Note**:
# 
# > Sweeney was a graduate student at MIT at that stage. She bought the data, re-identified Governor Weld's medical records, and sent these to him.
# 
# ![Weld case](privacy/Weld-case.png "William Weld case venn diagram.")
# 
# Sweeney (2002) later demonstrated that 87% of Americans can be uniquely identified by their zip code, gender, and birth date.
# 
# This value - i.e. the percentage of unique and, thus, identifiable members of the dataset knowing a couple of quasi-identifiers - has been conceptualized as  ***Uniqueness***.
# 
# While the numerous available sources of data may reveal insights into human behavior, you need to be sensitive to the legal and ethical considerations when dealing with them. These sources of data include: census data, medical records, financial and transaction data, loyalty cards, mobility data, mobile phone data, browsing history and ratings, research-based or observational data, and others.
# 
# You can review the seven principles of privacy by design in this [blog post](https://blog.varonis.com/privacy-design-cheat-sheet/).
# 
# > **Note**: 
# 
# > It is strongly recommended that you save a checkpoint after applying significant changes or completing exercises. This allows you to return the notebook to a previous state should you wish to do so. On the Jupyter menu, select "File", then "Save and Checkpoint" from the dropdown menu that appears.
# 

# # 1. Uniqueness and k-anonymity
# 
# > **Uniqueness** refers to the fraction of unique records in a particular dataset. In other words, the number of individuals who are identifiable given the fields.
# 
# The available fields in your dataset can typically contain:
# 
# > **Identifiers** which refer to attributes that can be used to explicitly identify individuals. These are typically removed from datasets prior to release.
# 
# > **Quasi-identifiers** which refer to a subset of attributes that can uniquely identify most individuals in the dataset. They are not themselves unique identifiers, but are sufficiently well-correlated with an individual that they can be combined with other quasi-identifiers to create a unique identifier.
# 
# Anonymization has been chosen as a strategy to protect personal privacy. K-anonymity is the measure used for anonymization, and is defined below according to Sweeney (2002).
# 
# > **K-anonymity of a dataset** (given one or more fields) is the size of the smallest group in the dataset sharing the same value of the given field(s) or the number of persons having identical values of the fields yielding them indistinguishable.
# 
# For k-anonymity the person anonymizing the dataset needs to decide what the quasi-identifiers are and what a potential attacker could extract from the provided dataset.
# 
# Generalization and suppression are the core tools used to anonymize data, and make a dataset k-anonymous (Samarati and Sweeney 1998). The privacy-securing methods employed in this paradigm are optimized for the high *k-anonymity* versus precision of the data. One of the biggest problems experienced is that optimization is use case specific and therefore depends on the application. Typical methods include:
# 
# - **Generalization (also called coarsening)**: reducing the resolution of the data, for example date of birth -> year of birth -> decade of birth.
# - **Suppression**: removing the rows from groups with k lower than desired from the dataset.
# 
# ![Date of Birth Coarsening.](privacy/DOB-coarsening.png "Date of Birth Coarsening.")
# 
# These heuristics typically come with trade-offs. Other techniques such as noise addition and translation exist, but provide similar results.
# 
# Technical examples of such methods are not of central importance in this course, therefore only basic components will be repeated to illustrate the fundamentals of the elements discussed above. 

# ### 1.1 Load dataset
# In this example we use a synthetic dataset created for 100 000 fictional people from Belgium. The zip codes are random numbers adhering to the same standards observed in Belgium, with the first two characters indicating the district.

# In[1]:

import pandas

# Load the dataset.
df = pandas.read_csv('privacy/belgium_100k.csv')
df = df.where((pandas.notnull(df)), None)
df['birthday'] = df['birthday'].astype('datetime64')
df.head()


# ### 1.2 Calculate uniqueness
# In order to calculate the uniqueness as defined earlier, we define a function that accepts and input dataset and list of features to be used to evaluate the dataset. The output indicates the number or records in the dataset that can be uniquely identified using the provided features.

# In[2]:

# Define function to evaluate uniqueness of the provided dataset.
def uniqueness(dataframe, pseudo):
    groups = dataframe.groupby(pseudo).groups.values()
    return sum(1. for g in groups if len(g) == 1) / len(dataframe)


# In[3]:

print uniqueness(df, ['zip'])
print uniqueness(df, ['sex', 'birthday'])
print uniqueness(df, ['sex', 'birthday', 'zip'])


# The results indicate that 20% of the individuals could potentially be identified using two features ("sex" and "birthday") and 99% of the population could potentially be re-identified using three features ("sex", "birthday" and "zip").

# ### 1.3 K-anonymity
# As per the earlier definition of k-anonymity, we calculate uniqueness as the smallest group of records that is returned based on the grouping parameters provided. In the code cell below we define a function that takes an input dataset and list of features to group the recordset when performing the evaluation. The function provides the minimum count of records grouped by these features as output.

# In[4]:

# Define function to evaluate k-anonymity of the provided dataset.
def k_anonymity(dataframe, pseudo):
    return dataframe.groupby(pseudo).count().min()[0]


# In[5]:

print k_anonymity(df, ['sex', 'birthday', 'zip'])


# In this example we observe a value of one which implies that the minimum number of individuals with a unique combination of the provided features is one and that there is a significant risk of potential attackers being able to re-identify individuals in the dataset.
# 
# Typically, the goal would be to not have any groups with size less than k values, as defined by your organizational or industry standards. Typical target values observed ranges from six to eight.
# 
# > **Note**:
# 
# > You can experiment with different combinations of features or repeat the test with single features to review the impact on the result produced.
# 
# > Example: `print k_anonymity(df, ['sex'])`

# In[10]:

# Code cell to review k_anonymity function with different input parameters.
print k_anonymity(df, ['sex'])
print k_anonymity(df, ['birthday'])
print k_anonymity(df, ['zip'])
print k_anonymity(df, ['cancer'])
print k_anonymity(df, ['sex', 'birthday'])
print k_anonymity(df, ['sex', 'zip'])
print k_anonymity(df, ['birthday', 'zip'])
print k_anonymity(df, ['cancer', 'zip'])
print k_anonymity(df, ['cancer','sex'])


# ### 1.4 Coarsening of data
# In this section we coursen the data using a number of different techniques. It should be noted that we lose granularity or accuracy of the dataset in order to preserve the privacy of the records for individuals in the dataset.
# 
# #### 1.4.1 Remove the zip code
# **District** is contained in the first two characters of the zip code. In order to retain the district when coarsening the data, a simple programmatic transformation (such as the below) can be applied. After applying this transformation you can choose to expose the "zip_district" to end users instead of the more granular "zip".

# In[11]:

# Reduce the zip code to zip district.
df['zip_district'] = map(lambda z: z / 1000, df['zip'])
df[['zip', 'zip_district']].head(3)


# In[12]:

print k_anonymity(df, ['zip'])
print k_anonymity(df, ['zip_district'])


# #### 1.4.2 Coarsen data from birthday to birth year
# Similar to the previous exercise, we can expose the birth year instead of the birthday as demonstrated in the code cell below.

# In[13]:

# From birthday to birth year.
df['birth_year'] = df['birthday'].map(lambda d: d.year)
df[['birthday', 'birth_year']].head(3)


# In[14]:

print k_anonymity(df, ['birthday'])
print k_anonymity(df, ['birth_year'])


# #### 1.4.3 Coarsen data from birthday to birth decade
# You can reduce granularity to decade level instead of yearly as per section 1.4.2 with the code demonstrated below.

# In[15]:

# From birthday to birth decade.
df['birth_decade'] = df['birth_year'] // 10 * 10
df[['birthday', 'birth_year', 'birth_decade']].head()


# In[16]:

print k_anonymity(df, ['birth_year'])
print k_anonymity(df, ['birth_decade'])


# ### 1.5 Suppression
# This refers to the suppression of all groups smaller than the desired k. In many cases you will reach a point where you will have to coarsen data to the point of destroying its utility. Removing records can be problematic as you may remove the records of interest to a particular question (such as 1% of data with a link to a particular feature).

# In[17]:

print k_anonymity(df, ['sex', 'birth_year', 'zip_district'])
grouped = df.groupby(['sex', 'birth_year', 'zip_district'])
df_filtered = grouped.filter(lambda x: len(x) > 5)
print 'Reducing size:', len(df), '> ', len(df_filtered)
print 'K-anonymity after suppression:', k_anonymity(df_filtered, ['sex', 'birth_year', 'zip_district'])


# In[18]:

df_filtered.head()


# # 2. Privacy considerations for big data
# Big data sets typically differ from traditional datasets in terms of:
# - **Longitude**: Data is typically collected for months, years, or even indefinitely, in contrast to snapshots or defined retention periods.
# - **Resolution**: Data points collected with frequencies down to single seconds.
# - **Features**: Features with unprecedented width and detail for behavioral data, including location and mobility, purchases histories, and more.
# 
# Many of the traditional measures used to define the uniqueness of individuals, and strategies to preserve users' privacy, are no longer sufficient. Instead of *Uniqueness* usually being used for fields consisting of single values, *Unicity* has been proposed (de Montjoye et al. 2015; de Montjoye et al. 2013). Unicity can be used to measure the ease of re-identification of individuals in sets of metadata such as a user's location over a period of time. Instead of assuming that an attacker knows all of the quasi-identifiers and none of the data, unicity assumes that any datapoint can either be known to the attacker or useful for research and focuses on quantifying the amount of information that would be needed to uniquely re-identify people. In many cases, data is poorly anonymized, and you also need to consider the richness of big data sources when evaluating articles such as [identifying famous people](http://bits.blogs.nytimes.com/2015/01/29/with-a-few-bits-of-data-researchers-identify-anonymous-people/).
# 
# 
# ## 2.1 Unicity of a dataset at *p* datapoints (given one or more fields)
# Given one or more fields, the unicity of a dataset at *p* datapoints refers to:
# - The fraction of users who can be uniquely identified by *p* randomly chosen points from that field.
# - Approximate number of datapoints needed to reconcile two datasets.
# 
# > **Note**: 
# 
# > Unicity is specifically designed for big data and its metadata, meaning that it's applicable to features containing numerous values - such as a trace, for example a history of GPS coordinates.
# 
# #### Unicity levels in big data datasets and it's consequences
# 
# de Montjoye et al. (2015) have shown that for 1.5 million people (over the course of a year), four visits to places (location + timestamps) are enough to uniquely identify 95% of the users. While for another 1 million people (over 3 months), unicity reached 90% at four points (shop + day) or even 94% at only three points (shop + day + approximate price). Such an ease of identification means that if someone anonymizes the big data of individuals effectively, they will strip it of its utility.
# 
# > **Note**: The 'Family and Friends' dataset transformed the location data of each user individually, which preserved the users' privacy very well, yet rendered the data unusable for our purposes. The 'Student Life' dataset, on the other hand, left the GPS records intact, which enabled us to use this as input for the Module 3 exercises. This introduces the risk of attacks to re-identify individuals by reconciliation with location services such as Foursquare, Twitter, and Facebook.
# 
# ![Unicity algorithm.](privacy/Unicity-algorithm.png "Unicity algorithm.")
# 
# #### Example 1: Assessing the unicity of a dataset.
# We will use a synthetic dataset which simulates the mobility of 1000 users. The dataset contains mobile phone records based on hourly intervals.

# ### 2.1.1 Sampling

# In[19]:

# Load the data.
import pandas as pd
import numpy as np
from scipy.stats import rv_discrete
from tqdm import tqdm

get_ipython().magic(u'pylab inline')


# In[20]:

# Load samples of the dataset.
samples = pd.read_csv('privacy/mobility_sample_1k.csv', index_col='datetime')

samples.index = samples.index.astype('datetime64')
samples.head(3)


# ### 2.1.2 Computing the unicity
# Below you will find an implementation of the unicity assessment algorithm defined earlier. A single estimation of unicity is performed using the compute_unicity function defined earlier, for a more robust result.
# 
# > **Note**:
# 
# > You do not need to understand the code in the cells below. It is provided as sample implementation for advanced users.

# In[21]:

def draw_points(user, points):
    '''IN: a Series; int'''
    
    user.dropna(inplace=True)

    indeces = np.random.choice(len(user), points, replace=False)
    return user[indeces]        
    
def is_unique(user_name, points):
    '''IN: str, int'''
    drawn_p = draw_points(samples[user_name], points) 
    for other_user in samples.loc[drawn_p.index].drop(user_name, axis=1).as_matrix().T:
        if np.equal(drawn_p.values, other_user).all():

            return False
    return True

def compute_unicity(samples, points):
    '''IN:int'''
    unique_count = .0
    
    users = samples.columns
    for user_name in users:
        if is_unique(user_name, points): 
            unique_count += 1

    return unique_count / len(samples.columns)

def iterate_unicity(samples, points=4, iterations=10):
    
    unicities = []
    for _ in tqdm(range(iterations)):
        unicities.append(compute_unicity(samples, points))
    
    return np.mean(unicities)


# #### Example calculation: Compute the unicity for a single data point. (Iterate 3 times)
# In this example we use one datapoint and three iterations. The result will vary based on the selected sample but will indicate that about 35% of the individuals in the sample could potentially be identified using a single data point.

# In[22]:

## Compute unicity.
iterate_unicity(samples, 1, 3)


# <br>
# <div class="alert alert-info">
# <b>Exercise 1 Start.</b>
# </div>
# 
# ### Instructions
# 
# > Calculate the unicity at four data points. Iterate five times for some additional accuracy. You can find the syntax in the example calculation above and change the parameters as required.
# 
# > **Question**: Is it big or small? What does it mean for anonymity?

# In[23]:

iterate_unicity(samples, 4, 5)


# It's larger than at 1 data point and 3 iterations and close to 1, meaning it's not secure at all since 99.98% of the individuals in the sample could potentially be identified using 4 data points. The anonymity is very insufficient to make sure that reidentification can be avoided.

# <br>
# <div class="alert alert-info">
# <b>Exercise 1 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# ## 2.2 Coarsening
# Similarly, here we could try coarsening in order to anonymize the data. This approach has been shown to be insufficient in making the dataset anonymous.
# 
# You can read more about the implementation and interpretation of results of the "Unique in the shopping mall" study [here](http://www.nature.com/articles/srep01376). 
# 
# ![Unique in the mall](privacy/Unique_in_the_Mall2.jpg "Unique in the mall")
# 
# Please review the paper and pay special attention to figure 4 which demonstrates how the uniqueness of mobility traces ε depends on the spatial and temporal resolution of the data. The study found that traces are more unique when coarse on one dimension and fine along another than when they are medium-grained along both dimensions. (Unique means easier to attack for re-identification of individuals)
# 
# The risk of re-identification decreases with the application of these basic techniques, however this decrease is not nearly fast enough. An alternate solution for this specific use case is to merge the antennas into (big) groups of 10 in an attempt to lower the unicity.

# > **Note**:
# 
# > The two code cells below is used to prepare your dataset, but does not produce any output. It will generate the input dataset required for exercise 2. The second code cell will also produce an error which you can safely ignore.

# In[24]:

# Load antenna data.
antennas = pandas.read_csv("privacy/belgium_antennas.csv")
antennas.set_index('ins', inplace=True)

cluster_10 = pandas.read_csv('privacy/clusters_10.csv')
cluster_10['ins'] = map(int, cluster_10['ins'])
mapping = dict(cluster_10[['ins', 'cluster']].values)


# In[28]:

# Reduce the grain of the dataset.
samples_10 = samples.copy()
samples_10 = samples_10.applymap(lambda k: np.nan if np.isnan(k) else mapping[antennas.index[k]])


# In[29]:

samples_10.head(3)


# <br>
# <div class="alert alert-info">
# <b>Exercise 2 Start.</b>
# </div>
# 
# ### Instructions
#     
# > Calculate the unicity of the coarsened mobility dataset (samples_10) with the same number of datapoints (four) and iterations (five) as in Exercise 1. You need to execute the same function and replace the input dataset, "samples", with the newly created, "samples_10" dataset.  
# 
# > a) Is there any difference to your answer in the previous exercise? 
# 
# > b) How much does it improve anonymity (if at all)?
# 
# > c) Is the loss of spatial resolution worth the computational load and effort?

# In[30]:

iterate_unicity(samples_10, 4, 5)


# a) No.
# 
# b) It decreases anonymity by 0.02% to reach zero. It worsen the result.
# 
# c) No. When lowering the temporal or the spatial resolution of the dataset, the uniqueness of traces decreases as a power function Uniqueness of Trace U = a - x^b.(E). While U decreases according to a power function, its exponent b decreases linearly with the number of points p. Accordingly, a few additional points might be all that is needed to identify an individual in a dataset with a lower resolution. Traces are more unique when coarse on one dimension and fine along another than when they are medium-grained along both dimensions.

# <br>
# <div class="alert alert-info">
# <b>Exercise 2 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# ## 2.3 Big data privacy: conclusion
# 
# In the context of big data, existing concepts and methods of privacy preservation are inadequate. Even the basic measure of how unique an individual is within the dataset needs to be replaced. Perhaps, more importantly, the old measure of privacy, i.e. k-anonymity, is unattainable unless the majority of information has been removed from the data (compare the unusable location data of the “Friends and Family” dataset). This leads us to conclude that:
# 
# **Anonymity is no longer a solution to the privacy problem in the big data context.**
# 
# The answer lies in the paradigm of data handling and can only be solved by software architecture changes. Solutions providing finely-grained access control and *remote computation* like [Open PDS](link) by Yves-Alexandre de Montjoye, or the [new initiative](http://www.digitaltrends.com/web/ways-to-decentralize-the-web/) by the inventor of the world wide web (WWW) Sir Tim Berners-Lee, show the way by effectively changing the privacy problem into a security one. You can also review the [Opal project](http://opalproject.org/) and [Solid](https://solid.mit.edu/) for more initiatives related to big data and privacy.

# <br>
# <div class="alert alert-info">
# <b>Exercise 3 Start.</b>
# </div>
# 
# ### Instructions
#     
# > It has been shown that data anonymization is no longer a practical solution in the context of big data. 
# 
# > a) Describe the typical problems experienced with the anonymization approach in the context of big data in your own words. Your description should be two or three sentences in length.
# 
# > b) What is the best alternative approach to ensure privacy of sensitive data in the context of big data?

# a) Big data analytics often use mobility data which appear to be highly unique, representing a high risk of re-identification since, as shown in the article of reference of de Montjoye, Hidalgo, Verleysen and Blondel, uniqueness of mobility traces decays approximately as the 1/10 power of their resolution, so very very slowly. Which means that even coarse datasets provide little anonymity, especially with the existence nowadays of many different public datasets to cross-reference. To reach a degree of k-anonymity sufficient enough to prevent re-identification is unattainable unless most inforation is removed from the dataset.
# 
# b) Alternatives:
# 
# - Data trasformation:
# 
# Transforming the sensitive data in randomized key codes, like in the "Family and Friends" study preserve well the data privacy.
# 
# - Personal metadata frameworks or Personal Data Stores (PDS) like OpenPDS/SafeAnswers and SOLID:
# 
# OpenPDS/SafeAnswer is a personal metadata management framework following WEF, NSTIC and European Commission rules that allow individuals to collect, store, and give fine-grained access to their metadata to third parties. As explained in "openPDS: Protecting the Privacy of Metadata through SafeAnswers" by de Montjoye, Shmueli1, Wang and Pentland, "under the openPDS/SafeAnswers mechanism, a piece of code would be installed inside the user’s PDS. The installed code would use the sensitive raw metadata (...) to compute the relevant piece of information within the safe environment of the PDS. In practice, researchers and applications submit code (the question) to be run against the metadata, and only the result (the answer) is sent back to them. openPDS/SafeAnswers is similar to differential privacy, both being online privacy-preserving systems." As said, SafeAnswers turns a hard anonymization problem into a more tractable security one.
# 
# Differential Privacy is designed for a centralized setting where a database contains metadata about numerous individuals and answers are aggregate across these individuals.
# 
# In the bottom up system of Tim Berners-Lee Social LInked Data (SOLID), derived from his concept of Linked Data, personal data is stored in Personal Online Data Stores (PODS) and the user/individual chooses how, when and to whom share it, and can split his data on a multiplicity of PODS corresponding to any degree of granularity of his data if he wishes to. And it is up to him to distribute each PODS independently or in groups. SOLID lets applications ask for data and it is up to individuals to authorize it or not by giving their permission. Once the permission is given, SOLID delivers the authorized PODS to the application requiring them.
# 
# - Recentralization methods like IPFS or applied blockchain:
# 
# The InterPlanetary File System (IPFS) of Juan Benet and developed as an open source project by Protocol Labs with help from the open source communities, is a content-addressable, peer-to-peer hypermedia distribution protocol that follow the peer-to-peer approach of BIt-Torrent and let multiple peered computers supply parts of a the information and deliver it to the authorized target/client.
# 
# The blockchain concept of Satoshi Nakamoto is a distributed database that maintains a continuously-growing list of data records secured from tampering and revision. Like the PODS above, it consists of data structure blocks containing data or programs, each block holding batches of combinations of both with a timestamp and a link to a previous block.
# 
# - Encryption currently widely used:
# 
# Here, to read an encrypted file, the user must have access to a secret key or password that authorizes him to decrypt it. Encrypted data is referred to as cipher text. The whole data or file or the bits of data are encrypted using an encryption algorithm, generating cipher text that can only be read if decrypted.
# 

# <br>
# <div class="alert alert-info">
# <b>Exercise 3 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 3. Submit your notebook
# 
# Please make sure that you:
# - Perform a final "Save and Checkpoint";
# - Download a copy of the notebook in ".ipynb" format to your local machine using "File", "Download as", and "IPython Notebook (.ipynb)"; and
# - Submit a copy of this file to the online campus.

# # 4. References
# Arrington, Michael. 2006. “AOL Proudly Releases Massive Amounts of Private Data.” TechCrunch, August 6. Accessed August 21, 2016. https://techcrunch.com/2006/08/06/aol-proudly-releases-massive-amounts-of-user-search-data/.
# 
# de Montjoye, Yves-Alexandre, César A. Hidalgo, Michel Verleysen, and Vincent D. Blondel. 2013. “Unique in the Crowd: The privacy bounds of human mobility.” Scientific Reports 3. doi:10.1038/srep01376.
# 
# de Montjoye Yves-Alexandre, Laura Radaelli, Vivek Kumar Singh, Alex “Sandy” Pentland. 2015. “Unique in the Shopping Mall: On the Re-identifiability of Credit Card Metadata.” Science 347:536- 539. doi:10.1126/science.1256297.
# 
# Golle, Philippe. 2006. “Revisiting the Uniqueness of Simple Demographics in the US Population.” Proceedings of the 5th ACM Workshop on Privacy in Electronic Society, Alexandria, Virginia, October 30.
# 
# Gymrek, Melissa, Amy L. McGuire, David Golan, Eran Halperin, and Yaniv Erlich. 2013. “Identifying Personal Genomes by Surname Inference.” Science 339: 321–24. doi:10.1126/science.1229566.
# 
# Narayanan, Arvind, and Vitaly Shmatikov. 2006. “How To Break Anonymity of the Netflix Prize Dataset.” arXiv [cs.CR]. arXiv. http://arxiv.org/abs/cs/0610105.
# 
# Samarati, Pierangela, and Latanya Sweeney. 1998. “Protecting Privacy When Disclosing Information: K-Anonymity and Its Enforcement through Generalization and Suppression.” epic.org. http://epic.org/privacy/reidentification/Samarati_Sweeney_paper.pdf.
# 
# Sweeney, Latanya. 2002. “K-Anonymity: A Model for Protecting Privacy.” International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems 10:557–70.

# In[ ]:



