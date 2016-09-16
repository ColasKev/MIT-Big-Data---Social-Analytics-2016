
# coding: utf-8

# <div align="right">Python 2.7 Jupyter Notebook</div>

# # Sources of data

# ### Your completion of the Notebook exercises will be graded based on your ability to: 
# 
# > **Apply**: Are you able to execute code, using the supplied examples, that perform the required functionality on supplied or generated data sets? 
# 
# > **Evaluate**: Are you able to interpret the results and justify your interpretation based on the observed data?
# 
# > **Create**: Your ability to produce notebooks that serve as computational record of a session that can be used to share your insights with others? 

# # Notebook introduction
# 
# Data collection is expensive and time consuming, as Arek Stopczynski alluded to in the video 2 resource on the learning path. 
# In some cases you will be lucky enough to have existing datasets available to support your analysis. You may have datasets from previous analyses, access to providers, or curated datasets from your organization. In many cases, however, you will not have access to the data that you require to support your analysis, and you will have to find alternate mechanisms. 
# The data quality requirements will differ based on the problem that you are trying to solve. Taking the hypothetical case of geocoding a location that was introduced in Module 1, the accuracy of the geocoded location does not need to be exact when you are simply trying to plot the locations of students on a map. Geocoding a location for an automated vehicle to turn off the highway, on the other hand, has an entirely different accuracy requirement.
# 
# > **Note**:
# 
# > Those of you who work in large organizations may be privileged enough to have company data governance and data quality initiatives. These efforts and teams can generally add significant value both in terms of supplying company-standard curated data, and making you aware of the internal policies that need to be adhered to.
# 
# As a data analyst or data scientist, it is important to be aware of the implications of your decisions. You need to choose the appropriate set of tools and methods to deal with sourcing and supplying data.
# 
# Technology has matured in recent years, and allowed access to a host of sources of data that can be used in our analyses. In many cases you can access free resources, or obtain data that has been curated, is at a lower latency, or comes with a service-level agreement at a cost. Some governments have even made datasets publicly available.
# 
# You have been introduced to [OpenPDS](http://openpds.media.mit.edu/) in the video content where the focus shifts from supplying raw data - where the provider needs to apply security principles before sharing datasets - to supplying answers rather than data. OpenPDS allows users to collect, store, and control access to their data, while also allowing them to protect their privacy. In this way, users still have ownership of their data, as defined in the new deal on data. 
# 
# This notebook will demonstrate another example of sourcing external data to enrich your analyses. The Python ecosystem contains a rich set of tools and libraries that can help you to exploit the available resources.
# 
# This course will not go into detail regarding the various options to source and interact with social data from sources such as Twitter, LinkedIn, Facebook, and Google Plus. However, you should be able to find libraries that will assist you in sourcing and manipulating these sources of data.
# 
# Twitter data is a good example as, depending on the options selected by the twitter user, every tweet contains not just the message or content that most users are aware of. It also contains a view on the network of the person, home location, location from which the message was sent, and a number of other features that can be very useful when studying networks around a topic of interest. Professor Pentland pointed out the difference in what you share with the world (how you want to be seen) compared to what you actually do and believe (what you commit to). Ensure you keep these concepts in mind when you start exploring the additional sources of data. Those who are interested in the topic can start to explore the options by visiting the [twitter library on pypi](https://pypi.python.org/pypi/twitter). 
# 
# Start with the five Rs introduced in module 1, and consider the following questions:
# - How accurate does my dataset need to be?
# - How often should the dataset be updated?
# - What happens if the data provider is no longer available?
# - Do I need to adhere to any organizational standards to ensure consistent reporting or integration with other applications?
# - Are there any implications to getting the values wrong?
# 
# You may need to start with “untrusted” data sources as a means of validating that your analysis can be executed. Once this is done, you can replace the untrusted components with trusted and curated datasets as your analysis matures.
# 
# > **Note**: 
# 
# > It is strongly recommended that you save a checkpoint after applying significant changes or completing exercises. This allows you to return the notebook to a previous state should you wish to do so. On the Jupyter menu, select "File", then "Save and Checkpoint" from the dropdown menu that appears.

# #### Load libraries and set options

# In[1]:

import pandas as pd
from pandas_datareader import data, wb
import numpy as np
import matplotlib
import folium
import geocoder
#import urllib2
get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (10, 8)


# # 1. Source additional data from public sources 
# ## 1.1 World-bank
# 
# This example will demonstrate how to source data from an external source to enrich your existing analyses. You will need to combine the data sources and add additional features to the example of student locations plotted on the world map in Module 1, Notebook 3.
# 
# The specific indicator chosen has little relevance other than to demonstrate the process that you will typically follow in completing your projects. Population counts, from an untrusted source, is added to our map and we use scaling factors combined with the number of students and population size of the country to demonstrate adding external data with minimal effort.
# 
# You can read more about the library that is utilized in this notebook [here](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#world-bank).

# In[2]:

# Load the grouped_geocoded dataset from Module 1.
df1 = pd.read_csv('data/grouped_geocoded.csv',index_col=[0])

# Prepare the student location dataset for use in this example.
# We use the geometrical center by obtaining the mean location for all observed coordinates per country.
df2 = df1.groupby('country').agg({'student_count': [np.sum], 'lat': [np.mean], 
                                  'long': [np.mean]}).reset_index()
# Reset the index.
df3 = df2.reset_index(level=1, drop=True)

# Get the external dataset from worldbank
#  We have selected indicator, "SP.POP.TOTL"
df4 = wb.download(
                    # Specify indicator to retrieve
                    indicator='SP.POP.TOTL',
                    country=['all'],
                    # Start Year
                    start='2008',
                    # End Year
                    end=2016
                )

# The dataset contains entries for multiple years.
#    We just want the last entry and create a separate object containing the list of maximum values
df5 = df4.reset_index()
idx = df5.groupby(['country'])['SP.POP.TOTL'].transform(max) == df4['SP.POP.TOTL']

# Create a new dataframe where entries corresponds to maximum year indexes in previous list.
df6 = df5[idx]

# Combine the student and population datasets.
df7 = pd.merge(df3, df6, on='country', how='left')

# Rename the columns or our merged dataset.
df8 = df7.rename(index=str, columns={('lat', 'mean'): "lat_mean", 
                                ('long', 'mean'): "long_mean", 
                                ('SP.POP.TOTL'): "PopulationTotal_Latest_WB",
                                ('student_count', 'sum'): "student_count"}
           )


# > **Note**:
# 
# > The cell above will complete with a warning message the first time that you execute the cell. You can ignore the warning and continue to the next cell to plot the indicator added.
# 
# > The visualization below does not have any meaning. The scaling factors selected is used to demonstrate the difference in population sizes and number of students on this course per country.

# In[3]:

# Plot the combined dataset

# Set map center and zoom level
mapc = [0, 30]
zoom = 2

# Create map object.
map_osm = folium.Map(location=mapc,
                   tiles='Stamen Toner',
                    zoom_start=zoom)

# Plot each of the locations that we geocoded.
for j in range(len(df8)):
    # Plot a blue circle marker for country population.
    folium.CircleMarker([df8.lat_mean[j], df8.long_mean[j]],
                    radius=df8.PopulationTotal_Latest_WB[j]/500,
                    popup='Population',
                    color='#3186cc',
                    fill_color='#3186cc',
                   ).add_to(map_osm)
    # Plot a red circle marker for students per country.
    folium.CircleMarker([df8.lat_mean[j], df8.long_mean[j]],
                    radius=df8.student_count[j]*10000,
                    popup='Students',
                    color='red',
                    fill_color='red',
                   ).add_to(map_osm)
# Show the map.
map_osm


# <br>
# <div class="alert alert-info">
# <b>Exercise 1 Start.</b>
# </div>
# 
# ### Instructions
# 
# > Copy the code from the previous two cells into the cells below. After you've reviewed the available indicators in the [worldbank](http://data.worldbank.org/indicator) dataset, replace the population indicator with an indicator of your choice. Add comments (lines starting with #) giving a brief description of your view on the observed results. Make sure to provide the tutor with a clear description of why you selected the indicator, what your expectation was when you started and what you think the results may indicate.
# 
# > **Note**: Advanced users are welcome to source data from alternate data sources or manually upload files to be utilized to their virtual analysis environment.
# 

# In[25]:

# Load the grouped_geocoded dataset from Module 1.
df1 = pd.read_csv('data/grouped_geocoded.csv',index_col=[0])

# Prepare the student location dataset for use in this example.
# We use the geometrical center by obtaining the mean location for all observed coordinates per country.
df2 = df1.groupby('country').agg({'student_count': [np.sum], 'lat': [np.mean], 
                                  'long': [np.mean]}).reset_index()
# Reset the index.
df3 = df2.reset_index(level=1, drop=True)

# Get the external dataset from worldbank
# We have selected indicator, "EN.POP.DNST" on http://data.worldbank.org/indicator/EN.POP.DNST Population density=people per sq. km of land area
# The population density of the students' countries of origin will give a slightly different picture than the map above where by the way the 
# showing if the students enrolled ar coming from high or low population density areas, which could be an indicator of future migrations to
# Go from high density to low density countries... In the map above, the population is divided by 500 and the number of students by country is 
# multiplied by 10000, which for me does not make much sense: a percentage of student in the total cohort compared with the percentage of the 
# world total population represented by the contribution of the population of each country in the total would have make more sense.
# The difference in population sizes and number of students on this course per country does not bring anything to the table.
# Replacing population by population density just shows that students in high density population have mechanically more job market competition
# and could have an incentive to move to lower density population: and it's actually what is happening today in the world. Chinese and Indians
# are fleeing highly competitive job markets and low average income countries to settled in low density population areas with less competitive
# job markets and higher avarage household income... Adding the average household annual income could bring an additional layer of information.
df4 = wb.download(
                    # Specify indicator to retrieve
                    indicator='EN.POP.DNST',
                    country=['all'],
                    # Start Year
                    start='2008',
                    # End Year
                    end=2016
                )

# The dataset contains entries for multiple years.
#    We just want the last entry and create a separate object containing the list of maximum values
df5 = df4.reset_index()
idx = df5.groupby(['country'])['EN.POP.DNST'].transform(max) == df4['EN.POP.DNST']

# Create a new dataframe where entries corresponds to maximum year indexes in previous list.
df6 = df5[idx]

# Combine the student and population datasets.
df7 = pd.merge(df3, df6, on='country', how='left')

# Rename the columns or our merged dataset.
df8 = df7.rename(index=str, columns={('lat', 'mean'): "lat_mean", 
                                ('long', 'mean'): "long_mean", 
                                ('EN.POP.DNST'): "Population_Density_WB",
                                ('student_count', 'sum'): "student_count"}
           )

df8

# Note that Canada and the US are represented by only 1 point in the above map, India representing 17 students does not appear on the map
# Only a single point appres in China and wee can wonder if it's just for China or all China and South-East Asia including the 64 
# students of HK and 80 of Singapore... The map lacks of precision/accuracy since it uses geolocation based on mean lattitude and longitude of
# all coordinates within each country...


# In[31]:

# Plot the combined dataset

# Set map center and zoom level
mapc = [0, 30]
zoom = 2

# Create map object.
map_osm = folium.Map(location=mapc,
                   tiles='Stamen Toner',
                    zoom_start=zoom)

# Plot each of the locations that we geocoded.
for j in range(len(df8)):
    # Plot a blue circle marker for country population density.
    # We accentuate the population density representation to show clearer differences but these changes in population or population density 
    # number of students by countries could be subject to discussion...
    folium.CircleMarker([df8.lat_mean[j], df8.long_mean[j]],
                    radius=df8.Population_Density_WB[j]*10000,
                    popup='Population_Density_WB',
                    color='#3186cc',
                    fill_color='#3186cc',
                   ).add_to(map_osm)
    # Plot a red circle marker for students per country.
    folium.CircleMarker([df8.lat_mean[j], df8.long_mean[j]],
                    radius=df8.student_count[j]*10000,
                    popup='Students',
                    color='red',
                    fill_color='red',
                   ).add_to(map_osm)
# Show the map.
map_osm

# We obtained the researched effect showing Europe, China, Central America and Ethiopia reprensenting hubs of students where population
# density is high and job markets very competitive, having a potential incentive to move to the red spots concentrating students but 
# having low population and consecutively lower job market compatitiveness. India does not appear in the plot though and we are 
# noticing the same lack of accuracy of the map... using mean locations


# In[ ]:

####Step by step:

### Prepare

## Read dataset selected
# df1 = pd.read_csv('data/grouped_geocoded.csv',index_col=[0])

## Prepare geolocation dataset attention to precision of method in line with expected reults
# df2 = df1.groupby('country').agg({'student_count': [np.sum], 'lat': [np.mean], 
#                                  'long': [np.mean]}).reset_index()

## Reset index
# df3 = df2.reset_index(level=1, drop=True)

## Download external dataset and select what is needed
# df4 = wb.download(
                    # indicator='EN.POP.DNST',
                    # country=['all'],
                    # start='2008',
                    # end=2016
#)

## Reset index
# df5 = df4.reset_index()

## Set constraints here setting max values
# idx = df5.groupby(['country'])['EN.POP.DNST'].transform(max) == df4['EN.POP.DNST']

## Create new dataframe
# df6 = df5[idx]

## Merge data from separate sources
# df7 = pd.merge(df3, df6, on='country', how='left')

## Rename
# df8 = df7.rename(index=str, columns={('lat', 'mean'): "lat_mean", 
                               # ('long', 'mean'): "long_mean", 
                               # ('EN.POP.DNST'): "Population_Density_WB",
                               # ('student_count', 'sum'): "student_count"}
#)

## Show table
# df8

### Plot

## Setup map center and zoom level
# mapc = [0, 30]
# zoom = 2

## Create map object
# map_osm = folium.Map(location=mapc,
                   # tiles='Stamen Toner',
                   # zoom_start=zoom)

## Plot locations created with a blue circle marker for country population density and a red for number of students with set 
## multipliers/divisers
# for j in range(len(df8)):
# folium.CircleMarker([df8.lat_mean[j], df8.long_mean[j]],
                   # radius=df8.Population_Density_WB[j]*10000,
                   # popup='Population_Density_WB',
                   # color='#3186cc',
                   # fill_color='#3186cc',
                   # ).add_to(map_osm)    
# folium.CircleMarker([df8.lat_mean[j], df8.long_mean[j]],
                   # radius=df8.student_count[j]*10000,
                   # popup='Students',
                   # color='red',
                   # fill_color='red',
                   # ).add_to(map_osm)
## Plot map
# map_osm


# <br>
# <div class="alert alert-info">
# <b>Exercise 1 End.</b>
# </div>

# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# ## 1.2 Wikipedia
# 
# To demonstrate how quickly data can be sourced from public, "untrusted" data sources, you have been supplied with a number of sample scripts below. While these sources contain extremely rich datasets that you can acquire with minimal effort, they can be amended by anyone and may not be 100% accurate. In some cases you will have to manually transform the datasets, while in others you might be able to use pre-built libraries.
# 
# Execute the code cells below before completing exercise 2.

# In[32]:

#!pip install wikipedia
import wikipedia

# Display page summary
print wikipedia.summary("MIT")


# In[33]:

# Display a single sentence summary.
wikipedia.summary("MIT", sentences=1)


# In[34]:

# Create variable page that contains the wikipedia information.
page = wikipedia.page("List of countries and dependencies by population")

# Display the page title.
page.title


# In[ ]:




# In[35]:

# Display the page URL. This can be utilised to create links back to descriptions.
page.url


# <br>
# <div class="alert alert-info">
# <b>Exercise 2 Start.</b>
# </div>
# 
# ### Instructions
# 
# > After executing the cells for the Wikipedia example in section 1.2, think about the potential implications of using this "public" and in many cases "untrusted" data sources when doing analysis or creating data products.
# 
# > **Please compile and submit a short list of pros and cons (three each). Your submission will be evaluated.**
# 
# > Your submission can be a simple markdown list or you can use the table syntax provided below.

# Add your answer in this markdown cell. The contents of this cell should be replaced with your answer.
# 
# **Submit as a list:**
# 
# ListType
# - Pro: Ease of use - The process of manipulating data, creating a page and posting on Wikipedia is made easy by Python.
# - Pro: Global exposure - Considering the global users base of Wikipedia, a posting on this site assure an fast exposure.
# - Pro: Peer reviews - Each posting is made by registered contributors tracked by Wikipedia and peer checked for errors & abuses.
# - Pro: Community test - The Wikipedia community size allow to test reactions on a large scale to new findings or researches.
# - Con: No garantee of validity - Even with data & content peer checks, codes of ethics & conduct, Wikipedia makes no garantees. 
# - Con: No formal peer review - Editors use Recentchanges or Newpages feeds to monitor new & changing content, no uniform review.
# - Con: No expert check - Anyone can peer review content, sources & references required by Wikipedia, not experts of the field.
# - Con: Legal issues - Copyrighting, data privacy and respect of the law is controlled on the basis of American laws.
# 

# <br>
# <div class="alert alert-info">
# <b>Exercise 2 End.</b>
# </div>

# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# ## 2. Submit your notebook
# 
# Please make sure that you:
# - Perform a final "Save and Checkpoint";
# - Download a copy of the notebook in ".ipynb" format to your local machine using "File", "Download as", and "IPython Notebook (.ipynb)"; and
# - Submit a copy of this file to the online campus.

# In[ ]:



