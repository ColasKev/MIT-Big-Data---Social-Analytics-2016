
# coding: utf-8

# <div align="right">Python 2.7 Jupyter Notebook</div>
# 
# # Big data projects
# <br>
# <div class="alert alert-warning">
# <b>This notebook should be opened and completed by students completing both the technical and non-technical tracks of this course.</b>
# </div>
# 
# ### Your completion of the notebook exercises will be graded based on your ability to:
# 
# > **Understand**: Does your pseudo-code and/or comments show evidence that you recall and understand technical concepts?
# 
# # Introduction
# 
# While you have been introduced to the many technical concepts, tools, and technologies, very little attention has been paid to people and processes. These are critical elements to consider when undertaking projects using data. Technological advances mean that tools and data collection methods are becoming more accessible, and that projects that previously required significant investment in terms of human resources and technology can be undertaken by a wider range of audiences. Your use cases and reasons for undertaking projects will vary greatly, but you will still need to interact with a variety of stakeholders.
# 
# In this notebook we are introducing you to a big data project methodology that will allow you to set up big data projects in your own social or commercial context. We use the existing project to demonstrate some of the key concepts which you can use as input in setting up your own projects in future.
# 
# In the video content, Professor Pentland referred to specific concepts which can also be applied to your project. Some of these include:
# - Creating social context which involves surrounding people with others who are thinking about trying certain things to get them to adopt those behaviors.
# - Playing with ideas does not change behavior.
# - You may need to implement multiple strategies to deal with the various parties in the social context (early adopters and close networks).
# 
# > **Note**:
# 
# > While the content introduced in the course is intended to provide you with insights typically applied in social analytics projects, you can also use these to ensure the successfull completion of your project. As example, projects where multiple parties work on similar projects with a shared vision typically have a higher likelihood of success than those performed by individuals in isolation. From a systems point of view this can be attributed to items such as shared tasks (data collected once and used multiple times), but we typically underestimate the impact of social context when executing projects. Please keep all the content introduced in the course in mind while working through the content presented in this notebook and ensure that you apply these concepts when setting up or interacting with other parties in your future endeavours.
# 
# Refer to the abstract of the [paper](http://realitycommons.media.mit.edu/download.php?file=FriendsAndFamily.pdf) titled “Social fMRI: Investigating and shaping social mechanisms in the real world”. This abstract is repeated below and you are encouraged to work through the detail of the paper over the next two weeks.
# 
# > **ABSTRACT:** We introduce the Friends and Family study, a longitudinal living laboratory in a residential community. In this study, we employ a ubiquitous computing approach, Social Functional Mechanism-design and Relationship Imaging, or Social fMRI, that combines extremely rich data collection with the ability to conduct targeted experimental interventions with study populations. We present our mobile-phone-based social and behavioral sensing system, deployed in the wild for over 15 months. Finally, we present three investigations performed during the study, looking into the connection between individuals’ social behavior and their financial status, network effects in decision making, and a novel intervention aimed at increasing physical activity in the subject population. Results demonstrate the value of social factors for choice, motivation, and adherence, and enable quantifying the contribution of different incentive mechanisms. (Aharony et al. 2011)
# 
# <div class="alert alert-warning">
# <b>Note</b>:<br>The first week (Module 6) will touch on various aspects of big data projects in general and provide some technical considerations, while the second week (Module 7) will focus on the specific study and its results.
# </div>

# # 1. Project overview
# 
# Taking a "People, Process, Technology, Data" high level view of projects can be a useful framework when planning or communicating projects at a concept level. Most project management methodologies will contain similar elements to those depicted in the image below.
# 
# ![Data: People, Process, Technology](peopleprocesstechnology.png "Data: People, Process, Technology")
# (Source: [Analytics Infrastructure: 15 Considerations](http://blogs.sas.com/content/datamanagement/2012/05/09/analytics-infrastructure-15-considerations/))
# 
# ## 1.1 People
# Reuse existing skills and budget time and training for your human resources. Many of the items require additional learning curves and training. The nature of the types of analysis performed means that it is an iterative process by definition and many of the steps are extremely difficult to budget for using traditional project management methodologies. Make sure that you have the relevant skills available and ensure that you deal appropriately with the introduction of new tools and methodologies. Ensure that the analysts work in an environment where other users in the organization can support them or work with tools where there is a rich community of support (social context).
# 
# ** Typical roles include**.
# - Stakeholders and decision makers.
#     - Within an organization: Chief information officers (CIO), chief data officers (CDO), governance and compliance officers, and business unit representatives.
#     - Within academia: Lecturers and study leads or department representatives of your institution.
# - Analytic professionals.
#     - Data scientists to create new functions and perform ad-hoc analyses (typically using scripting languages such as R and Python, but may also use other existing tools where appropriate).
#     - Data scientists and analysts to repeat and refine analyses. This is similar to the above step, but also typically includes the use of more accessible tools such as SQL (Structured Query Language) and BI (business intelligence) tools.
#     - Business analysts and knowledge workers are supported by data analysts and data scientists, and typically use apps and end-to-end tools.
# - Infrastructure specialists.
#     - Architects, database administrators, and system owners.
# 
# ## 1.2 Process
# The “process” component of projects refers to the following aspects:
# - Define the processes and implement standards to ensure that your activities are repeatable. Adhere to the standards (academic or commercial) required by your organization, and meet the requests of your analytic needs.
# - Academics have a well defined and rigorous process in place to deal with many of the issues. In the business and commercial world, many organizations are moving from waterfall-based approaches to more agile and "fail-fast" or "[data-driven approaches](http://andrewchen.co/know-the-difference-between-data-informed-and-versus-data-driven/)". 
# 
# ## 1.3 Technology
# The “technology” component of projects is characterized by the need to:
# - Choose the analytic platform components as required. Reuse and be critical of limitations. Consider open ecosystems and environments that can be used to speed up progress, but perform due diligence prior to implementing in production, as the checks and balances exist for a reason.
# - Be careful of moving to this step too quickly. Resolve the function and architecture before making technology choices.
# 
# ## 1.4 Data
# The “data” component of projects involves carrying out  the following steps:
# - Identify, acquire, store, and provide access to the data required to support your analytic needs.
# - Apply the appropriate governance and privacy protection standards.
# - Use the data to uncover insights, trends, and patterns from data.

# # 2. Change management
# Approaching projects purely from a technical or analytical point of view typically delivers underwhelming results. You need to be aware that once you reach an insight, you will also have to act on that insight, and that action needs to be performed in the social context of your organization or environment. This may be a simple change in an existing business process, however, in many cases there may be significant changes in existing processes, new processes, or products and services that require some form of change management in order to implement the proposed changes. You can review topics such as [organizational transformation](http://www.iienet2.org/Details.aspx?id=24456) and “data-driven approaches” for further information and guidance.
# 
# Many of the approaches advocate iterative or 'fail-fast' approaches to test the concept before embarking on full-scale projects. The approach you follow will be highly dependent on the type and size of organization, and potentially on the tools and resources at your disposal. The majority of approaches contain similar elements, which will be explored in this notebook.
# 
# - Phase 0: Pilot: Preparation and feasibility checks.
# - Phase 1: Project execution and maintenance.
# - Phase 2: Project termination.

# # 3. Project initiation
# > **Vision**:
# 
# > Imagine the ability to place an imaging chamber around an entire community. Imagine the ability to record and display nearly every facet and dimension of behavior, communication, and social interaction among the members of the said community. Moreover, envision being able to conduct interventions in the community, while measuring their effect — by both automatic sensor tools as well as qualitative assessment of the individual subjects. Now, think about doing this for an entire year, while the members of the community go about their everyday lives.
# 
# You may find inspiration in science, technology, attempting to solve a business need, or your inspiration may be driven by personal interest. Whatever your reason for choosing to initiate a project, you will still need to communicate the ideas to multiple stakeholders and plan your project to ensure success.
# 
# Starting with a clearly-stated and referenceable vision is a good way to ensure that you can communicate the project’s intention to multiple stakeholders, and get them excited and focused on the topic being studied. When setting the vision, keep the potential stakeholders and decision makers in mind. This is usually championed by an individual, but may form part of formal structures such as thesis proposals or workshops with multiple potential stakeholders. In terms of your documentation: abstracts, introductions, and overviews are typically completed last as these items are frequently revisited and updated throughout the course of the project.
# 
# The processes in academia, and specifically postgraduate studies, are set up to ensure that prospective students research their topic of interest, and clearly define their intended area of research. The research methodologies are carefully validated to ensure that all relevant aspects are addressed. Business applications tend to be less rigorous regarding processes while having an increased focus on achieving monetary results. You are urged to work through the detail of the referenced paper by Aharony et al. (2011), and think carefully about where each of the described sections may be applicable to your potential use cases.

# # 4. Research and context
# > **Review section 2 of the referenced paper**: Related work and context.
# 
# Research will enable you to benefit from incorporating the latest research in your analytic efforts. At this stage, many of your questions may have already been addressed to some extent, allowing you to build on the work completed by others in order to accelerate your efforts, while other questions may not be answered directly.

# # 5. Project framework
# You can use project methodologies that are applicable to your industry, or that you are familiar with, but it is highly advisable to keep a couple of topics in mind.
# 
# In your research you will likely encounter subject material relevant to your project which you can use as input. This may include subject matter or business value frameworks, project plans, typical questions, and plans to answer these questions.
# 
# #### Value perspective
# Focusing on the value perspective from an early point helps to keep you focused. This involves considering the following questions:
# - How do you define success?
# - What is the intended outcome?
# - Monetization of results?
# - How will you measure social impact?
# - Will you be using incentives such as gamification?
# 
# #### Accelerators
# Accelerators may include using programming libraries - such as pandas for data analysis, or Bandicoot for the interrogation of mobile phone metadata - but you will also be able to leverage the content from studies and service providers in order to accelerate your efforts.
# 
# #### Skills review
# New technological options, and applying the best tool or process ("fit for purpose"), may achieve results in dramatically-reduced timeframes or at a fraction of the cost typically associated with large-scale studies. However, you should start building a view of the resources (available and required) at an early stage.
# 
# #### Communicating your proposal
# Lastly, you should decide on a strategy to visualize and communicate your project.
# 
# > **Review**: Section 3 of the referenced paper has a clear example of framing your intentions in a manner that can be easily communicated to stakeholders.

# # 6. Methodology
# 
# > **Review**: Section 4 of the referenced paper.

# <div class="alert alert-info">
# <b>Exercise 1 Start.</b>
# </div>
# 
# ### Instructions
# 
# > Can you list two advantages of the multimodal approaches similar to that discussed in the paper?

# Commonly, computational social science studies of large scale have very large sample size, collecting information on a very large number of subjects. But in general, the amount of information in the dataset or "throughput" is limited by several factors:
# 
# . The use of only very few data-crumbs or "domains" like emails, credit card transactions or browsing histories and patterns,
# 
# . Time-interval information collect sessions creating offline confusion of causes and effect relationships vs. real-time online continuous data collect,
# 
# . A lack of contextual information on the individuals subjects themselves. 
# In consequence, the results of those domain-limited studies are hard to generalize to larger populations.
# 
# Trhoughput if defined by data dimensionality (number of signals collected or data-crumbs), resolution (granularity and structuration of the data), sampling rate (frequency of data collection) and uniqueness (specificities of the data and its methods of collection). 
# 
# Increasing the dimensionality and throughput of datasets, multimodal approaches including bottom up data collection from the users or subjects of the study using simultaneously multiple data-crumbs (phone and internet browsing logs and sensors readings like accelerometer, gyro-meter, bluetooth, wifi, GPS, email, credit card transactions, app installed, app activity, Facebook, Twitter or LinkedIn logs, mood, stress, activity and sleep polling or psychological personality test, phone and web surveys, video and media consumption) combining automatic digital data collection and interventions by interactions with the subjects through questionnaires, have several advantages with two major: higher throughput, online continuous data without offline effect, highly dimensional and contextual information on users or subjects and information on interrelations between data-crumbs or signals.

# <br>
# <div class="alert alert-info">
# <b>Exercise 1 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# The paper describes the methodology employed in detail. When embarking on your journey there will be additional items to consider which typically do not make it into the paper or the final product. Some of the items outside of the scope of the course are included here for your consideration:
# 
# ## 6.1 Data collection
# In an earlier video within the course, Arek Stopczynski referred to data collection being expensive. Carefully review the data sources already available, as well as potential new sources of data. You will need to balance the need for trusted and high-accuracy or low-grain data with the costs and overheads associated with obtaining the data. In many cases, your pilot project will focus on easier-to-obtain data and you may need to use proxies or subsets of data to demonstrate the usefulness of the concept being studied. Many of the questions that you will need to answer may potentially be answered by a small subset of data (the [Pareto principle](http://www.investopedia.com/terms/p/paretoprinciple.asp)). There would typically be a number of use cases for the same set of data. Think of social network data. While many organizations, or units within organizations, are excited about the prospect of using this rich source of data, few are able to obtain value from it.
# 
# ## 6.2 Data preparation
# Data preparation can be a tedious process and typically requires more time than is budgeted for. Accelerators aid in transformation, and the governance of data can help you to reduce ongoing required efforts. When thinking about the creation of data products or implementing production systems, you typically want to automate these steps, in order to free up capacity of your data scientists, ensuring that they can spend their time on useful tasks rather than repetitive ones.
# 
# ## 6.3 Analytics
# In addressing analytics concerns, you would build up a "toolbox" as you practise and implement your analytic capabilities. These may include:
# * Statistical analyses and predictive modeling, including:
#     * Generalized linear models (GLMs);
#     * K-nearest neighbor (KNN) classification algorithms; and
#     * Naive bayes models to train models for churn prediction.
# * Pattern and distribution matching.
# * Time series analysis.
# * Segmentation.
# * Graph and cluster analysis.
# * Data transformation for advanced analysis or input to existing tools.
# * Text analytics to derive patterns and features.
# * Sentiment analysis.
# * Geospatial analysis.
# * Machine and deep learning techniques.
# * Visualization of random forests and single decision trees.

# <br>
# <div class="alert alert-info">
# <b>Exercise 2 Start.</b>
# </div>
# 
# ### Instructions
# 
# > Section 4.2 of the referenced paper contains a number of data sources such as mobile phone call and sensor records, surveys, purchasing behavior, and Facebook data. Describe what these type of records would typically tell you about the behavior of the individuals.
# 
# > **Hint**: Refer to content from earlier modules where the difference between how you want to be perceived and what you commit to is described.

# Telecom networks data like phone and text or sms logs provide very strong signal of the social structure, stronger than Facebook, Twitter or LinkedIn contact logs since on the contrary of interactions on social networks often with strangers or network specific connections without previous or even future physical contacts, people rarely call of text people they don't know. Telecom network are therefore more important to understand social relationships.
# 
# Credit card informations or like in the study financial statements, are informing on actual actions of subjects vs. just interaction or browsing on web or mobile apps showing people needs and interests in a realized view, not simply at the stage of wish or curiosity. This information shows at what people commit more strongly.
# 
# Physical interaction networks that can be collected through GPS, Bluetooth or Wifi positioning and proximity scanning bring information on diffusion phenomenons like diseases or virus propagation and are obviously interesting to study movements. Different degrees of commitment can be shown in this information or those relations, from just passing by to interracting physically.
# 
# Facebook, Twitter or LinkedIn and other social networks are interesting to study rumors, their trends and their modalities of diffusion. Those network correspond more at the way people want to be perceive than at what they actually are.

# <br>
# <div class="alert alert-info">
# <b>Exercise 2 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 7. System architecture
# > **Review**: Section 5 of the referenced paper.
# 
# The focus of the provided paper is on the mobile phone platform. It defines the sensors utilized and describes the data formats, data movement protocols, and the data storage structure.
# 
# The **system architecture** is a conceptual model that defines and describes the technical components contained in the system structure as well as the system’s behavior. It is typically used for planning and communication purposes where the components need to be described to interested parties.
# 
# Typically you would want to create a system that delivers fast and scalable results, and the components selected would vary greatly based on available existing resources as well as new requirements and options. You would typically start with a conceptual or logical model that describes the functions and flow of information in the system. At a later stage of the process you would revisit granular details, such as the physical implementation and specific components utilized.
# 
# Many get stuck on trying to select the technology first, then extending and revisiting the logical and conceptual models at later stages. This approach carries the risk of diluting the purpose and getting lost in technicalities rather than focusing on the defined purpose.
# 
# Privacy considerations will be addressed in the second notebook of this module. However, at this stage it is important to note that traditional methods applied to anonymize data in relational or file-based systems are usually no longer adequate. This necessitates approaches such as “privacy by design” to ensure that you deal with sensitive and potentially sensitive data appropriately. Consider data collected for internal and external use. Your internal researchers may require access to granular and unmasked data, while other business users or applications should only be able to access anonymized datasets.
# 
# While the technical components of systems architecture fall outside of the scope of the course, the following section briefly outlines a number of items that you should carefully consider when defining your architecture.
# 
# ## 7.1 Dealing with Data
# Consider the following points on dealing with data before defining your target system architecture:
# * Sourced data can be stored as files or in dedicated structures such as databases. These databases can either be relational or non-relational, and their physical implementation should be based on function and cost considerations. A number of implementation alternatives exist, and you can review two of the most popular structures - logical data warehouses and data lakes.
# * Interacting with the data can be done via a wide variety of tools, and the choice of technology will determine the available options. The data can be accessed via BI tools, Excel, SQL or APIs (application program interfaces). These concepts are not described in detail here, however, you are encouraged to perform your own research should the topics be of interest to you.
# * Data governance and data definitions need to be maintained over time and are typically of use to multiple stakeholders in your organization. While controlling these in the analytic phase often ensures quick time to action, it is important to hand these over to stable structures to avoid having to repeat the actions on an ongoing basis. The advantage of this approach is that you will start building rich, "trusted" data structures that you can access for future use cases. In addition to this, your insights can start to form part of the internal reporting structures for your organization (if they exist).
# * Methodologies applied in traditional business intelligence (BI) approaches are usually based on waterfall methodologies. Analytic projects tend to be better suited to agile or "fail-fast" methodologies that promote quicker time-to-value and iterative approaches. Once you have determined that your analysis holds value, you can decide on the appropriate action to implement on a more permanent basis. This is instead of committing to large projects before being able to ascertain or confirm that the project is feasible and can be implemented.

# <br>
# <div class="alert alert-info">
# <b>Exercise 3 Start.</b>
# </div>
# 
# ### Instructions
# 
# > Provide a short description of a recent technological component or trend that you think is significant. This should be selected based on your personal experience or interests, as many of the topics fall outside of the scope of the course.
# 
# > a) State your hypothetical social analytics use case (for context).
# 
# > b) State the technological component or trend and provide a short, relevant motivation for selecting this component. The description should contain at least five key points and can include benefits or potential risks to the technology or trend.
# 
# > **Hint**: API's, mobile applications, cloud-based computing, and interactive computing are examples of topics that are not covered in detail in this course. Should you wish to do so, you can also base your answer on technological concepts introduced in this course.

# **a.** An API dedicated to survey sector or industry specific groups of professional like C-levels, heads of brands or business units, or managers in product development, product innovation, finance, HR, compliance or legal departments would be very useful, time and budget saving for strategy consulting firms vs the current state of practice consisting in sending periodically questionnaires, or doing phone an interviews to those large audiences. And it would allow consulting firms to run those survey more often on large scale samples of firms either focusing on small and medium size businesses, targeting large companies like Fortune 500s or specifically startups. Besides, once built, the database on contacts underlying the app could be use to automatically find Twitter and LinkedIn accounts of those subjects to bring additional datasets through the collect and analysis of their postings and the postings of their companies. Anonymized the data collection of such surveys could be an incentive to respondents to be more open and express more personal opinion than in a PR controlled environment, and social network postings text analysis could help to confirm or infirm the data collected in the survey. The risk is the fear of respondent of de-identification of the data, privacy and security of the dataset, non diffusion of their responses for commercial purpose and their fear to breach NDAs and contractual or other legal limits. Such API does not exist yet and main survey tools remain Survey Monkey, SurveyGizmo, Google Forms, FluidSurveys or Qualtrics phone and onsite interviews or email/mailing questionnaires, harder to process.
# 
# **b.** The technological components would be the following I believe:
# 
# - Mobile survey application: to allow mobile responses or notifications and accommodate usual high mobility of consultants. The risk here is a low installation rate, data leaks if stolen or lost phones, hacking and crashing/debugging.
# 
# - Online survey website: to give an alternative to mobile use when in sedentary mode. The risk is hacking or maintenance issues.  
# 
# - Sequel Pro, MySQL or Microsoft SQL database: to record respondents data and contact information. A data physically secured and de-identified are the most important risk here.
# 
# - Open public API: to sets of requirements that govern how the application can talk to others and enable expose content to internal or external audiences. Low use from developers and security could be the risks here.
# 
# - Server-side back-end: to processes incoming files and inserts them into the central database, sends email reports to the research team about the status of survey application installation and alerts of any issues, or remotely patch and update application and website. Additional services provide data for survey management and personal data visualization for participants. Data storage, security and de-identification are at risk of overheating or power outage.
# 
# - An object-relational-mapper (ORM): to enables representing all data as code objects, ensure application persistence and simplify the development of applications using the data. Risk is probably slow time response of queries. 

# <br>
# <div class="alert alert-info">
# <b>Exercise 3 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 8. Project checklist
# When carrying out the phases of a project (plan), it is helpful to make use of a checklist to ensure each step in the plan is executed, and each aspect accounted for. The following sections outline typical phases of a project plan, and highlight some of the items that should be added to your checklists.
# 
# ## 8.1 Phase 0
# * Research the topic under review as well as related topics that can be used to add context.
# * Define your project framework.
#     * Define the vision and scope of the project.
#     * Review your value perspective. 
#         * List value drivers. How will you measure success? (monetization, ROI, etc.)
#     * Define security and access policies.
#     * Identify use cases and stakeholders.
#     * Feasibility checks. 
#     * High-level conceptual architecture.
# - Pilot projects typically focus on prioritized use cases.
#     * Describe and prioritize use cases.
#         * Often based on business impact, timing (availability and access to data), complexity, and effort.
#     * Source and transform data.
#     * Data analysis typically follows the same steps as listed in previous modules with short iterations.
# 
# ## 8.2 Phase 1
# * Create a roadmap or an implementation plan.
#     * Plan the activities and resources required to execute the plan.
#     * The data analysis cycle includes:
#         * Collection;
#         * Pre-processing;
#         * Hygiene;
#         * Analysis (a combination of analytic techniques);
#         * Visualization;
#         * Interpretation; and
#         * Intervention, which may include operationalized insights or a policy definition.
# 
# ## 8.3 Phase 2
# Project termination generally requires dependencies to be removed and data to be dealt with appropriately. Ensure that you are aware of the legal requirements when disposing of or archiving sensitive data. Architecture documents describing both systems and interactions between different systems can be of significant value during this phase.

# > **Note**: In the second part of this notebook, which you will review in Module 7, you will revisit technical sections six to eleven as well as the conclusion of the referenced paper.

# <br>
# <div class="alert alert-info">
# <b>Exercise 4 Start.</b>
# </div>
# 
# ### Instructions
# Project management methodologies and best practices can ensure that you get started quickly, but can also introduce a significant amount of overheads. These "best practices" can be used to guide you in unknown areas and help you deal with the complexities of running large projects.
# 
# > List two problems that you expect to experience when attempting to scale small-scale or ad-hoc analyses to large-scale implementations; or when incorporating these analyses within large organizations. Propose simple corrective actions for each.
# 
# > **Note**: These concepts are not dealt with in detail in this course. We are looking for insights based on the tools and technologies introduced in this course rather than attempting to test your knowledge of the various frameworks that exist.

# **Scaling up is a guided process of diffusion of innovation, in contrast to a spontaneous process.** A scaling-up strategy is a combination of plans and subsequent actions, controls and corrections necessary to establish the innovation in corporate policies, programmes and product/service delivery processes. A report of the UN World Health Organization and ExpandNet lists "Nine steps for developing a scaling-up strategy" (http://www.expandnet.net/PDFs/ExpandNet-WHO%20Nine%20Step%20Guide%20published.pdf):
# 
# >"Step 1.	 Planning actions to increase the scalability of the innovation.
# >Step 2.	 Increasing the capacity of the user organization to implement scaling up.
# >Step 3.	 Assessing the environment and planning actions to increase the potential for scaling-up success.
# >Step 4.	 Increasing the capacity of the resource team to support scaling up.
# >Step 5.	 Making strategic choices to support vertical scaling up (institutionalization).
# >Step 6.	 Making strategic choices to support horizontal scaling up (expansion/replication).
# >Step 7.	 Determining the role of diversification.
# >Step 8.	 Planning actions to address spontaneous scaling up.
# >Step 9.	 Finalizing the scaling-up strategy and identifying next steps."
# 
# The report defines different types of scaling up:
# 
# - Vertical scaling up—institutionalization through policy, political, legal, budgetary or other systems change.
# 
# - Horizontal scaling up—expansion or replication,
# 
# - Diversification,
# 
# - Spontaneous.
# 
# Problems can arise for example when the planning is weak lacking definition of the innovation, user organizations, its environment or the resource team. The analysis of those elements should conclude on recommendations to simplify the innovation to ease the transfer to user organizations, build training capacities inside user organizations, improve adherence to sectorial rregulations and monitoring reforms, or better plan workforce management and shortages.
# 
# Failing to capture the business requirements is a common source of failure in project management. Indeed, apropriate concertation and internal/external interviews and meetings are needed to make sure allt the elements are taken into consideration from definition of the vision, goals, scope and requirements to the expected results by milestones and deliverable with conrresponding deadlines, storage, security, compliance or legal aspects.
# 
# Other issues can be ignoring or underestimating costs and resource mobilization or not defining properly methods and metrics of monitoring and evaluation.

# <br>
# <div class="alert alert-info">
# <b>Exercise 4 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# ## 9. Submit your notebook
# 
# Please make sure that you:
# - Perform a final "Save and Checkpoint";
# - Download a copy of the notebook in ".ipynb" format to your local machine using "File", "Download as", and "IPython Notebook (.ipynb)", and
# - Submit a copy of this file to the online campus.

# # 10. References
# Aharony, Nadav, Wei Pan, Cory Ip, Inas Khayal, Alex Pentland. 2011. “SocialfMRI: Investigating and shaping social mechanisms in the real world.” Pervasive and Mobile Computing 7:643-659.

# In[ ]:



