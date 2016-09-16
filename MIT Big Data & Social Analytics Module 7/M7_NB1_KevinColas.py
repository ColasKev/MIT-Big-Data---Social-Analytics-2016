
# coding: utf-8

# <div align="right">Python 2.7 Jupyter Notebook</div>
# 
# # "Friends and Family" study review
# <br>
# <div class="alert alert-warning">
# <b>This notebook should be opened and completed by students completing both the technical and non-technical tracks of this course.</b>
# </div>
# 
# ### Your completion of the notebook exercises will be graded based on your ability to:
# 
# > **Understand**: Does your pseudo-code and/or comments show evidence that you recall and understand technical concepts?
# 
# # Notebook introduction
# The first notebook of Module 6 dealt with a holistic project overview. As a continuation of that overview, this notebook will highlight selected portions of the “[Friends and Family](http://realitycommons.media.mit.edu/friendsdataset.html)” study (Aharony et al. 2011). This is in order to prepare you to create a blueprint for your own project in Module 8.
# 
# > **Note**: Please refer to the sections of the referenced [paper](http://realitycommons.media.mit.edu/download.php?file=FriendsAndFamily.pdf), indicated in parentheses, for the content discussed in the following sections of the notebook.
# 
# # 1. The social fMRI approach (Section 6)
# The social fMRI’s data-rich methodology allows for the initial exploration of specific questions, as discussed in the paper, as well as the design of subsequent components and sub-experiments in the longitudinal study. In scientific studies you typically start with the project definition, specific questions, and hypotheses to be investigated. The data and insights that are gained as you analyze available data will lead you to form and investigate new hypotheses. You will then need to revisit and review these hypotheses against the high-level goals of the study, and communicate with the various project stakeholders.
# 
# ![Scientific method](Scientific_Method.png "Overview of scientific method.")
# 
# (Source: [Scientific method overview](http://idea.ucr.edu/documents/flash/scientific_method/story.htm))

# # 2. Investigating behavior choices (Section 7)
# The discovery of the strong correlation between social interaction patterns and the wealth and economical development of a community has attracted significant attention (Eagle et al. 2010). A current challenge is to understand the causality of this finding. In their article, titled “Network Diversity and Economic Development”, Eagle et al. (2010) find that users’ social diversity patterns only correlate with their current income, as illustrated in figures 5 and 6. Thus, these observations suggest the opposite: individuals will quickly lose their diversity in social interaction when their financial status gets worse. Consequently, individuals will quickly gain their social interaction diversity when their financial status improves. This result demonstrates how social fMRI-type studies can provide novel perspectives on long-standing debates in social science.

# # 3. Investigating the social fabric using decisions and choice (Section 8)
# While it is not possible to capture all of the decisions made by individuals, you would typically find a proxy behavior that is representative of the action to be measured. In the “Friends and Family” study, the apps installed by users were used to test their decision-making process(es). Physical co-location and self-reported closeness were used to better understand social support and behavior choices. Observations based on the information above suggest that the diffusion of apps relies more on face-to-face interaction ties than on self-perceived friendship ties.

# # 4. Active intervention and discussion (Sections 9-12)
# Intervention requires novel approaches. In the study, a third experimental condition was defined. It was aimed at generating increased incentives for social influence, and potentially leveraging social capital. Another design consideration that you would usually have to take into account is designing less accurate, but more robust, algorithms. In the study, three activity level groupings were created based on minimal data collection. This design allowed for minimal intrusion, as well as the ability to monitor an increase in activity levels.
# 
# The design of the reward structure, as well as the richness of the available data, enabled the study of networks (rather than a closed team structure) for social interventions. For example, A receives a reward for B and C’s performance, while D and E receive a reward for A’s performance. It is then possible to disambiguate and focus on the dyadic and asymmetric relationship of the person doing the activity versus the person receiving the reward, who is motivated to convince the former to be active.
# 
# > Typical [gamification](http://blogs.gartner.com/brian_burke/2014/04/04/gartner-redefines-gamification/) approaches focus on the use of rewards like points and badges to change the behavior of users, which can cause long-term damage to intrinsic motivation. Meaningful gamification is the use of design concepts from games and play to help people find personal connections to a real-world setting. You can refer to this
# 
# > (Eberhardt 2013)
# 
# > **Note**: You can refer to [this](https://www.cs.uic.edu/~jzhang2/files/2016_asonam_slides2.pdf) overview of social badge reward system analysis based on Pokemon Go for more information.

# <div class="alert alert-info">
# <b>Exercise 1 Start.</b>
# </div>
# 
# ### Instructions
# 
# > Based on the content presented in this course:
# 
# > a) Describe how the approach followed in the “Friends and Family” study (as described in section 4 of this notebook) differs from typical gamification approaches.
# 
# > b) List two potential advantages of this design.

# a) The system designed in Friends & Family is novel to the extent that the mechanism rewards subjects based on their peers’ performance and not their own like in most gamification models. The action of my network peers will determine my performance and I have to motivate and influence them in order to get the benefit of their goal achievements. And their results suggest that social factors have an effect individual behaviors, motivation, and commitment over time, that social incentives and particularly similar peer-reward mechanisms command higher ROI and that a complex contagion or network effect exist between participants with pre-existing social ties favoring sometimes face to face interaction to trigger decisions (application downloads for example in the study).
# 
# Moreover, the study compares rewards systems splitting the population between three conditions:
# - The control condition: where subjects can only see their own progress and are rewarded only on their own activity.
# - A first experimental condition called "peer-view": where subjects were shown their own progress and the progress of two "Buddies" of the same experimental group and vice-versa with still a reward depending on each subject own activity.
# - A second experimental condition called "PeerReward": where subjects were shown their own progress as well as that of two "Buddies", but this time subjects’ rewards depend solely on the performance of their "Buddies".
# 
# b) This way the study of Aharonya, Pana, Ipa, Khayal and Pentland can leverage the network effect between network peers focusing on the influence of each participant and positive results in a non-competitive environment instead of relying solely on peer pressure in a competitive environment like most gamification systems. Also, results seem to show that embedding the social aspects in a non-competitive game adds in performance over time compared to classic gamification models represented by the control condition. Besides, if both control and peer-view or peer-see conditions deteriorates as time passes, the performance of a peer-reward system is slower to start but steadier in increase over time showing more sustainability or higher return on investment (ROI).
# 
# 

# <br>
# <div class="alert alert-info">
# <b>Exercise 1 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 5. Submit your notebook
# 
# Please make sure that you:
# - Perform a final "Save and Checkpoint";
# - Download a copy of the notebook in ".ipynb" format to your local machine using "File", "Download as", and "IPython Notebook (.ipynb)", and
# - Submit a copy of this file to the online campus.

# # 6. References
# Aharony, Nadav, Wei Pan, Cory Ip, Inas Khayal, Alex Pentland. 2011. “SocialfMRI: Investigating and shaping social mechanisms in the real world.” Pervasive and Mobile Computing 7:643-659.
# 
# Eagle, Nathan, Michael Macy, and Rob Claxton. 2010. “Network Diversity and Economic Development.” Science 328:1029-1031. doi:10.1126/science.1186605. 
# 
# Eberhardt, Rik. 2013. “Meaningful Gamification: Motivating through Play instead of Manipulating through Rewards.” MIT Game Lab Blog Archives, December 12. http://gamelab.mit.edu/tag/gamification/. 

# In[ ]:



