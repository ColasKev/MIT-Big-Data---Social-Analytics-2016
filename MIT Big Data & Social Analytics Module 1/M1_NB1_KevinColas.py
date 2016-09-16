
# coding: utf-8

# <div align="right">Python 2.7 Jupyter Notebook</div>

# # Introduction to Python

# ### Your completion of the Notebook exercises will be graded based on your ability to:
# 
# > **Understand**: Does your pseudo-code and/or comments show evidence that you recall and understand technical concepts?
# 
# > **Apply**: Are you able to execute code, using the supplied examples, that perform the required functionality on supplied or generated data sets? 

# # Notebook introduction
# 
# In the Orientation Module, you were given a number of links that provided Python and additional library documentation and tutorials. This section will serve as a summarized version of useful commands to get non-technical users started, and equip them with the basics required to complete this course.
# 
# This course is not intended to be a Python training course, but rather to showcase how Python can be utilized in analysis projects, and more specifically in analyzing social data. You will be provided with links that you can explore in your personal time to further your expertise and you are welcome to offer additional suggestions to your peers in the online forums.
# 
# You should execute the cells with sample code and write your own code in the indicated cells in this notebook. When complete, ensure that you save and checkpoint the notebook, download a copy to your local workstation and submit a copy to the Online Campus.
# 
# You can also visit [Python Language Reference](https://docs.python.org/2/reference/index.html) for a more complete reference manual that describes the syntax and "core semantics" of the language.
# 
# > **Note to new Python users**: 
# 
# > This notebook will introduce a significant amount of new content. You do not need to be able to master all components, but we urge you to work through the examples below and attempt to follow the logic. In the first notebook of this section we will introduce you to Python syntax. The second notebook will start to introduce various components of data analysis basics, and the third will guide you through a basic example from beginning to end. The focus will shift from programming to subject related examples in subsequent modules. 
# 

# > **Note**: 
# 
# > We strongly recommend that you save a checkpoint after applying significant changes or completing exercises. This allows you to return the notebook to a previous state should you wish to do so. On the Jupyter menu, select "File", then "Save and Checkpoint" from the dropdown menu that appears.

# # 1. Python Basics
# Execute the cells below and follow the instructions in the comments where applicable.

# In[1]:

# We introduced the the first line of code in any new language
print "hello world!"


# In[2]:

# Setting a variable
var = 1
print var


# In[6]:

# Set your name as a variable and say hello to yourself
# Type your name in the input block that appears and press Enter to continue
yourname = raw_input('Enter your name: ')
print 'Hello', yourname


# # Exercise 1
# Say hello to yourself 5 times. You can set your name as a variable, or use the input function demonstrated above.

# In[14]:

# Your code here
yourname = raw_input('Enter your name: ')
print 'Hello', yourname,'!'
# Variant 1
print "hello Kevin Colas !"
# Variant 2
yourname = 'Kevin Colas !'
print 'Hello', yourname










# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 2. Data Types and Operators

# Working with multiple data types and being aware of what they are is often overlooked in analysis. You will typically use int (integers) and float (floating-point) numeric types, str (characters or string) and bool (true-false values). Each of these data types have common operators that can be applied to them.
# 
# | Data Type  | Typical Operators |
# | ------------- | ------------- |
# | int  | + - * // % **  |
# | float  | + - * / % **  |
# | bool  | and or not  |
# | str  | +  |
# 
# 

# ## 2.1 Type
# 
# When working with new data sources or performing ad-hoc analysis, the "type" function can be employed to provide more information on the input or output objects. Execute the cells below to see how this function can assist you.

# In[15]:

a = 1
type(a)


# In[16]:

b = '1'
type(b)


# In[17]:

c = 'one'
type(c)


# In[18]:

d = 1.125
type(d)


# In[19]:

e = True
type(e)


# ## 2.2 Numbers
# Before continuing, it’s a good time to view the list of all currently defined variables to ensure that you do not reuse existing variables that you may require at a later stage. Retrieving the list of currently defined variables can be achieved with the function "who".

# In[20]:

who


# In[21]:

# Integers and floats
f = 5
g = 2
h = 2.1

k = f/g
# cast 'f' to float and perform the same calculation
l = float(f)/g
m = f/h

print 'f = ', f, 'and the type of f is:',type(f)
print 'g = ', g, 'and the type of g is:',type(g)
print 'h = ', h, 'and the type of h is:',type(h)
print '\n'
print 'Integer division: \nThe result of integer, f = 5, divided by integer, g = 2, is of type ', type(k), ' and has a value of ', k
print '\n'
print 'Float division:\nWe can cast the integer to float to produce the result of type float for the same calculation.'
print 'The result is of type ', type(l), ' and has a value of ', l
print '\n'
print 'Integer divided by float: \nThe result is of type ', type(m), ' and has a value of ', m


# Python can also perform calculations with [Complex Numbers](https://docs.python.org/2/library/cmath.html).

# In[22]:

# Complex numbers
r = 1.5 + 2j  # complex number.
s = 4         # int
t = r + s     # complex + int
print 'Calculations with complex numbers: Result is of type ', type(t), ' and has a value of ', t


# > **Note **:
# 
# > You may have to use different libraries for different data types, for example the "math" library can be used to calculate the sine of float data types, while you will require "cmath" to calculate the sine of complex numbers.
# 
# A great place to continue your journey outside the course is on the [Scipy.org website](https://www.scipy.org/), which contains links to [Pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/), [Numpy](http://www.numpy.org/) and [Scipy](http://docs.scipy.org/doc/) documentation.

# ## 2.3 Strings
# The value of a string data type is a sequence of characters. Strings are typically used for text processing.
# You can perform concatenation and a number of other functions on strings.

# In[23]:

# String example
x = 'Hello'
y = 'World'

# String concatenation
z = x + ' ' + y + '!'
print z

# String concatenation utilizing formatting options
z = '{} {}!'.format(x, y)
print z

# String information
print 'The length of z is', len(z)

# String repetition and concatenation
print 3*x+' '+y


# ## 2.4 Lists and tuples
# 
# Variables store information that may change over time. When you need to store longer lists of information, there are additional options available. You will be introduced to lists and tuples as basic elements and can read more about native Python data structures in the [Python documentation](https://docs.python.org/2/tutorial/datastructures.html#). During this course you will be exposed to examples using the [Pandas](http://pandas.pydata.org/) and [Numpy](http://www.numpy.org/) libraries that offer additional options for data analysis and manipulation. 
# 
# ### 2.4.1 [Lists](https://docs.python.org/2/tutorial/datastructures.html#more-on-lists)
# Lists are changeable or mutable sequences of values. Lists are stored within square brackets and items within a list are separated by commas. Lists can also be modified, appended, and sorted amongst other methods.

# #### Create and print a list

# In[24]:

# lists
lst = ['this', 'is', 'a', 'list']
print lst

# print the number of elements in the list
print 'The list contains', len(lst), 'elements.'


# #### Selecting an element in a list
# Remember that Python's index starts at zero. You can use a negative index to select the last element in the list.

# In[27]:

# print the first element in the list
print lst[0]

# print the third element in the list
print lst[1]

# print the third element in the list
print lst[2]

# print the last element in the list
print lst[-1]


# #### Append to a list

# In[28]:

# appending a list
print lst
lst.append('with')
lst.append('appended')
lst.append('elements')
lst.append('and')
lst.append('show')
lst.append('the')
lst.append('list')
lst.append('content')
print lst

# print the number of elements in the list
print 'The list updates list contains', len(lst), 'elements.'


# > **Note**:
# 
# > When selecting and executing the cell again, you will continue to add values to the list.
# 
# > Try this now: Select the cell above again and execute it to see how the input and output content changes.
# 
# > This can come in handy when working with loops.

# **Changing a list**
# 
# We will not cover string methods in detail. You can read more about [string methods](https://docs.python.org/2/library/stdtypes.html?highlight=upper#sequence-types-str-unicode-list-tuple-bytearray-buffer-xrange) if you are interested.

# In[29]:

# Changing a list
# Note: Remember that Python starts with index 0
lst[0] = 'THIS'
lst[3] = lst[3].upper()
print lst


# #### Define a list of numbers

# In[30]:

# define a list of numbers
numlist = [0, 10, 2, 7, 8, 5, 6, 3, 4, 1, 9]
print numlist


# #### Sort and filter a list

# In[31]:

# sort and filter list
sorted_numlist = sorted(i for i in numlist if i >= 5)
print sorted_numlist


# #### Remove the last element from a list

# In[32]:

# Remove the last element from the list
list.pop(sorted_numlist)
print sorted_numlist


# ### 2.4.2 [Tuples](https://docs.python.org/2.7/tutorial/datastructures.html?highlight=tuples#tuples-and-sequences)
# Tuples are similar to lists, except that they are typed in parentheses and are immutable. Tuples therefore cannot have their values modified.

# In[33]:

tup = ('this', 'is', 'a', 'bigger', 'tuple')
print(tup)


# In[34]:

tup[3]


# In[35]:

# tuples cannot be changed and will fail with an error if you try to change an element
tup[3] = 'new'


# ## 2.5 Boolean operators
# A boolean (logical) expression evaluates to true or false. Python provides the boolean type that can either be set to true or false. Many functions and operations return boolean objects. You have been introduced to some of the basic operators in earlier examples. A brief overview of boolean operators is provided here, as they can be very useful when comparing different elements.

# In[36]:

# equal to
x = 'Left'
y = 'Right'
x == y


# In[37]:

# not equal to
x = 'Left'
y = 'Right'
x != y


# In[38]:

# less than
x = 3
y = 4
x < y


# In[39]:

# greater than or equal to
x >= y


# # 3. Loops, sequences and conditionals
# Conditions and loops can be utilized to repeat statements based on the conditions or loops specified. This can be employed to cycle through data sets or perform a sequence of steps on, or based on, input variables.

# ## 3.1 [Range](https://docs.python.org/2/library/functions.html#range)
# Range(start, stop, step) is used to create lists containing arithmetic progressiongs. If you call a list with only one argument specified, it will use the value as stop value and default to zero as start value. The step argument is optional and can be a positive or negative integer.

# In[40]:

# Generate a list of 10 values.
myrange = range(10)
myrange


# In[41]:

# Generate a list with start value equal to one, stop value equal to ten that increments by three.
myrange2 = range(1, 10, 3)
myrange2


# In[42]:

# Generate a negative list
myrange3 = range(0, -10, -1)
myrange3


# ## 3.2 Basic loops
# Python uses indentation to repeat items. The example below demonstrates:
# - Cycling through the generated list.
#  - Printing the current element in the list.
#  - Printing X's, repeated per element.
# - Exiting the loop and printing a line that is not repeated.

# In[43]:

# You can specify the list manually or use a variable containg the list as input
# The syntax for manual input is: `for item in [1, 2, 3]:`

for item in myrange:
    print item
    print item * 'X'
print 'End of loop (not included in the loop)'


# ## 3.3 Conditionals
# Conditionals are used to determine which statements are executed. In the example below we import the "random" library and generate a random number smaller than 2. We assume 0 means heads and 1 means tails. The conditional statement is then used to print the result of the coin flip as well as "Heads" or "Tails".

# In[44]:

# Flip a coin

import random
coin_result = random.randrange(0,2)

print coin_result

if coin_result == 0:
    print 'Heads'
else:
    print 'Tails'


# # 4. Functions
# Functions allow us to avoid duplicating code. When defining a function, the def statement is used. The desired function code is then placed into the body of the statement. A function usually takes in an argument and returns a value as determined by the code in the body of the function. Whenever referring to the function outside of the function body itself, this action is known as a function call.

# In[45]:

# function 'firstfunction' with argument 'x'
def firstfunction(x):
    y = x * 6
    return y

# call your function    
z = firstfunction(6)
print(z)


# Function definitions, loops and conditionals can be combined to produce something useful. In the example below we will simulate a variable number of coin flips and then produce the summary of results as output.

# In[57]:

# function 'coinflip' with argument 'x' for the number of repetitions that returns the number of 'tail' occurrences
def coinflip(x):
    # set all starting variables to 0
    heads = 0
    tails = 0
    flip = 0
    # start a loop that executes statements while the conditional specified results in 'True'
    while flip < x:
        # generate a random number smaller than 2
        flipx = random.randrange(0,2)
        # increment heads by 1 if the generated number is 0
        if flipx == 0:
            heads = heads + 1
        # increment tails by 1 if the generated number is larger than 0
        if flipx > 0:
            tails = tails + 1
        # increment the flip variable by 1
        flip += 1
    return [heads, tails]

# set the number of repetitions
rep = 100

# call the function and set the output to the variable 'tails'
coinflip_results = coinflip(rep)

# print results of the function
coinflip_results


# # Exercise 2: Rolling Dice
# 
# Use the coinflip example above and change it to simulate rolling dice. Your output should contain summary statistics for the number of times you rolled the dice and occurrences of each of the 6 sides.
# 
# > **Hints**:
# 
# > - Replace "Heads" and "Tails" with "Side_1" to "Side_6".
# > - Increase the maximum of the random number generated.
# > - Test for additional states of the random variable, and increase the counter for the relevant variable.
# > - Add additional output variables.

# In[1]:

def rollingdice(x):
    Side_1 = 0
    Side_2 = 0
    Side_3 = 0
    Side_4 = 0
    Side_5 = 0
    Side_6 = 0
    roll = 0
while roll < x:
    rollx = random.randrange(0,6)
    if rollx == 0:
        Side_1 = Side_1 + 1
        Side_2 = Side_2 + 1
        Side_3 = Side_3 + 1
        Side_4 = Side_4 + 1
        Side_5 = Side_5 + 1
        Side_6 = Side_6 + 1
    if rollx > 0:
        Side_1 = Side_1 + 1
        Side_2 = Side_2 + 1
        Side_3 = Side_3 + 1
        Side_4 = Side_4 + 1
        Side_5 = Side_5 + 1
        Side_6 = Side_6 + 1
    rollx += 1
return [Side_1, Side_2, Side_3, Side_4, Side_5, Side_6]
rep = 200
rollingdice_results = rollingdice(rep)
rollingdice_results




# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# The basic random function was used in the example above, but you can view [this site](http://bigdataexaminer.com/data-science/how-to-implement-these-5-powerful-probability-distributions-in-python/) to see an example of implementing other probability distributions.

# # 5. Loading files
# 
# A lot of the input data that will be available will be stored in either files or databases. This section will introduce limited examples to show the syntax and functionality, but will restrict use to CSV files in most cases. You can read from and write to a multitude of data storage structures, including relational databases and non-relational data storage structures such as Hadoop. When you start to approach the boundaries of the available memory and compute capacities of your infrastructure, it is likely time to switch to a data storage system that better supports your needs. However, from an analysis perspective, the most important aspect is being able to interrogate and analyze the data.
# 
# ## 5.1 Loading CSV files
# Python's [CSV](https://docs.python.org/2/library/csv.html?highlight=csv) module allows the user to load data from a CSV file. In the first example we load a sample CSV file that contains four columns and three rows.

# In[2]:

# loading CSV files
import csv
egFile = open('workbook.csv')
egReader = csv.reader(egFile)
egData = list(egReader)
egData


# ## 5.2 JSON strings
# JSON (JavaScript Object Notation) is a lightweight data-interchange format that is easy to read and write for humans, and easy to parse and generate for computers. You can read more about the Python encoder and decoder [here](https://docs.python.org/2/library/json.html).

# In[3]:

# loading JSON data
import json
jsonStr = '{"Name": "John", "Surname": "Smith", "Age": 30}'
jsonStrToPy = json.loads(jsonStr)
jsonStrToPy


# ## 5.3 Reading and writing to SQLite databases
# [SQLite](https://docs.python.org/2/library/sqlite3.html?highlight=sqlite#module-sqlite3) is a lightweight, disk-based database that does not require a separate server process. Other databases would require different libraries and connection strings, but are very similar to the example demonstrated below. The structure of the example is based on the link earlier in this paragraph and provides good input with regards to the use and best practices of SQLite databases.

# In[4]:

# import the sqlite library and open the database
import sqlite3


# #### Connect to your database, create a table and add some sample data.

# In[5]:

# connect to your database
conn = sqlite3.connect('example.db')

# set the connection cursor
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS students
             (id INTEGER PRIMARY KEY, industry text, country text, city text)''')

# Insert 3 rows
c.execute("INSERT INTO students VALUES (NULL, 'Agriculture', 'Australia', 'Perth')")
c.execute("INSERT INTO students VALUES (NULL, 'Arts & Education', 'Greece', 'Thessaloniki')")
c.execute("INSERT INTO students VALUES (NULL, 'Public Sector', 'United States','San Francisco')")

# Save (commit) the changes
conn.commit()

# Make sure that any changes have been committed before closing the connection. 
# If the previous step was not completed, any changes that you have made will be lost.
conn.close()


# > **Note**: 
# 
# > If you choose a filename, "example.db" in the example above, does not exist, it will be created for you.
# 
# > We have not added any checks so executing the cell above multiple times will keep on adding duplicates of the same three rows.

# #### Retrieve the data in your database.

# In[6]:

# connect to your database and set the cursor
import sqlite3
conn = sqlite3.connect('example.db')
c = conn.cursor()

# Fetch the records
c.execute('SELECT * FROM students')

# Print the result set row by row
for row in c.execute('SELECT * FROM students ORDER BY id'):
        print row

# Close the connection
c.close()


# #### Delete a subset of records from your database.

# In[7]:

# connect to your database and set the cursor
import sqlite3
conn = sqlite3.connect('example.db')
c = conn.cursor()

# Delete all records where the id is greater than 1
c.execute('DELETE FROM students WHERE id > 1')

# Fetch the records
c.execute('SELECT * FROM students')


# Print the result set row by row
for row in c.execute('SELECT * FROM students ORDER BY id'):
        print row

# Close the connection
c.close()


# # 6. Additional library examples
# 
# This section contains examples of additional libraries that will be utilized in subsequent notebooks. The Anaconda Python distribution includes a number of libraries that you typically require for data analysis. You can also install your own libraries from a wide variety of sources. Some of these will be demonstrated when you set up for your weekly tasks and refresh your virtual analysis environment.
# 
# > **Note**:
# 
# > If you are not already familiar with Matplotlib and Pandas, it would be worthwhile to review the documentation and complete some of the tutorials available. The section below provides links and a brief overview of their use and functionality.

# ## 6.1 Matplotlib
# 
# The examples below come from the [Matplotlib Screenshots](http://matplotlib.org/users/screenshots.html) page.
# 
# > **Note**: 
# 
# > One of the options that you will need to remember to set when you use Matplotlib is "%matplotlib inline" which instructs the notebook to plot inline instead of opening a separate window for your graph.
# 
# You can find additional information on Matplotlib here:
# - [Matplotlib Documentation](http://matplotlib.org/)
# 
# - [Matplotlib Beginners](http://matplotlib.org/users/beginner.html)
# 
# - [Matplotlib Gallery](http://matplotlib.org/gallery.html)

# ### 6.1.1 Simple Plot
# The graph below does not contain any real significance other than demonstrating how easy it is to plot graphs. 

# In[9]:

# Import matplotlib library and set notebook plotting options
import matplotlib.pyplot as plt
import numpy as np
# Instruct the notebook to plot inline. 
get_ipython().magic(u'matplotlib inline')

# Generate data
t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*np.pi*t)

# Create plot
plt.plot(t, s)

# Set plot options
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)

# Saving as file can be achieved by uncommenting the line below
# plt.savefig("test.png")

# Display the plot in the notebook.
# The '%matplotlib inline' option set earlier ensure that the plot is displayed inline
plt.show()


# ### 6.1.2 Matplotlib - Demo with subplots
# Subplots is another useful feature where you can display multiple plots. This is typically used to visually compare data sets.

# In[11]:

"""
Simple demo with multiple subplots.
"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('Time (s)')
plt.ylabel('Undamped')

plt.show()


# ### 6.1.3 Matplotlib - Histogram (with a few features)
# Here is another example to be utilized in future. The intention is to demonstrate syntax and provide ideas of what is possible.

# In[14]:

"""
Demo of the histogram (hist) function with a few features.

In addition to the basic histogram, this demo shows a few optional features:

    * Setting the number of data bins
    * The ``normed`` flag, which normalizes bin heights so that the integral of
      the histogram is 1. The resulting histogram is a probability density.
    * Setting the face color of the bars
    * Setting the opacity (alpha value).

"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# Example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)

num_bins = 50
# The histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# Add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()


# ### 6.2 Pandas

# Pandas is an open source, BSD-licensed library that provides high-performance, easy-to-use data structures and data analysis tools for the Python programming language. For more information on the Pandas library, explore the list of resources provided below.
# 
# - [Pandas web page](http://pandas.pydata.org/)
# - Pandas [Documentation](http://pandas.pydata.org/pandas-docs/stable/) and [Tutorial](http://pandas.pydata.org/pandas-docs/stable/tutorials.html)
# - [External 10 Minute overview video on Vimeo](http://vimeo.com/59324550)

# The [Pandas cookbook](https://github.com/jvns/pandas-cookbook) is an excellent resource. Work through the set of examples in the cookbook directory to familiarize yourself with Pandas.

# One of the many useful features of Pandas is the ability to create arrays with sample data. In the example below we create an array with random data in a single statement. You will learn how to utilize the relevant features at a later stage. Should you be interested in learning more about the specific random function, you can read more [here](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html)

# In[15]:

# example of generating an array of values
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(10,4), index=np.arange(0, 100, 10), columns=list('WXYZ'))
df


# ## 7. Submit your notebook
# 
# Please make sure that you:
# - Perform a final "Save and Checkpoint";
# - Download a copy of the notebook in ".ipynb" format to your local machine using "File", "Download as" and "IPython Notebook (.ipynb)";
# - Submit a copy of this file to the online campus.

# In[ ]:




# In[ ]:



