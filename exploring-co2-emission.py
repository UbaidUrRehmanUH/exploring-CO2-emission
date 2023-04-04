#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('../input/Indicators.csv')
# describe function for describing the value in dataset
data.describe()  


# 

# In[3]:


# Shape function for checking the value in dataset
data.shape


# In[4]:


#  function for dropping duplicate values from dataset
data = data.drop_duplicates() 


# In[ ]:


# describe function for reading the value from dataset
def read_wb_data():
    # Read the CSV file 
    df = pd.read_csv("./Indicators.csv")

    df = df.drop(columns=['CountryCode', 'IndicatorCode'])

    # Pivot the dataframe to have years as columns
    df_years = df.pivot(index='CountryName', columns='Year', values='Value')

    df_countries = df.pivot(index='Year', columns='CountryName', values='Value')

    # Clean the dataframes
    df_years = df_years.dropna()
    df_countries = df_countries.dropna()

    return df_years, df_countries


# In[ ]:


# Create a dataframe with years as columns
df_years = df.pivot_table(index='CountryName', columns='Year', values='Value', aggfunc='first')


# In[ ]:


# Create a dataframe with countries as columns
df_countries = df.pivot_table(index='Year', columns='CountryName', values='Value', aggfunc='first')


# In[11]:


countries = data['CountryName'].unique().tolist()
len(countries)


# In[12]:


# How many unique country codes are there ? (should be the same #)
countryCodes = data['CountryCode'].unique().tolist()
len(countryCodes)


# In[13]:


# How many unique indicators are there ? (should be the same #)
indicators = data['IndicatorName'].unique().tolist()
len(indicators)


# In[14]:


# How many years of data do we have ?
years = data['Year'].unique().tolist()
len(years)


# In[15]:


print(min(years)," to ",max(years))


# In[16]:


# select CO2 emissions for the United States
hist_indicator = 'CO2 emissions \(metric'
hist_country = 'USA'

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['CountryCode'].str.contains(hist_country)

# stage is just those indicators matching the USA for country code and CO2 emissions over time.
stage = data[mask1 & mask2]


# In[17]:


stage.head()


# In[18]:


# get the years
years = stage['Year'].values
# get the values 
co2 = stage['Value'].values

# create
plt.bar(years,co2)
plt.show()


# In[19]:


# switch to a line plot
plt.plot(stage['Year'].values, stage['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel(stage['IndicatorName'].iloc[0])

#label the figure
plt.title('CO2 Emissions in USA')

# to make more honest, start they y axis at 0
plt.axis([1959, 2011,0,25])
#plt.plot(stage['Year'].values, stage['Value'].values)

plt.show()


# In[21]:


# 
hist_data = stage['Value'].values


# In[22]:


print(len(hist_data))


# In[23]:


# the histogram of the data
plt.hist(hist_data, 10, normed=False, facecolor='green')

plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Years')
plt.title('Histogram Example')

plt.grid(True)

plt.show()


# In[24]:


# select CO2 emissions for all countries in 2011
hist_indicator = 'CO2 emissions \(metric'
hist_year = 2011

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['Year'].isin([hist_year])

# apply our mask
co2_2011 = data[mask1 & mask2]
co2_2011.head()


# For how many countries do we have CO2 per capita emissions data in 2011

# In[26]:


print(len(co2_2011))


# In[27]:


# let's plot a histogram of the emmissions per capita by country

# subplots returns a touple with the figure, axis attributes.
fig, ax = plt.subplots()

ax.annotate("USA",
            xy=(18, 5), xycoords='data',
            xytext=(18, 30), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

plt.hist(co2_2011['Value'], 10, normed=False, facecolor='green')

plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Countries')
plt.title('Histogram of CO2 Emissions Per Capita')

#plt.axis([10, 22, 0, 14])
plt.grid(True)

plt.show()


# So the USA, at ~18 CO2 emissions (metric tons per capital) is quite high among all countries.
# 
# An interesting next step, which we'll save for you, would be to explore how this relates to other industrialized nations and to look at the outliers with those values in the 40s!

# ### Relationship between GPD and CO2 Emissions in USA

# In[28]:


# select GDP Per capita emissions for the United States
hist_indicator = 'GDP per capita \(constant 2005'
hist_country = 'USA'

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['CountryCode'].str.contains(hist_country)

# stage is just those indicators matching the USA for country code and CO2 emissions over time.
gdp_stage = data[mask1 & mask2]

#plot gdp_stage vs stage


# In[29]:


gdp_stage.head(5)


# In[30]:


stage.head(5)


# In[31]:


# switch to a line plot
plt.plot(gdp_stage['Year'].values, gdp_stage['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel(gdp_stage['IndicatorName'].iloc[0])

#label the figure
plt.title('GDP Per Capita USA')


plt.show()


# In[32]:


# switch to a Bar plot
plt.bar(gdp_stage['Year'].values, gdp_stage['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel(gdp_stage['IndicatorName'].iloc[0])

#label the figure
plt.title('GDP Per Capita USA')



plt.show()


# So although we've seen a decline in the CO2 emissions per capita, it does not seem to translate to a decline in GDP per capita

# ### ScatterPlot for comparing GDP against CO2 emissions (per capita)
# 
# First, we'll need to make sure we're looking at the same time frames

# In[33]:


print("GDP Min Year = ", gdp_stage['Year'].min(), "max: ", gdp_stage['Year'].max())
print("CO2 Min Year = ", stage['Year'].min(), "max: ", stage['Year'].max())


# We have 3 extra years of GDP data, so let's trim those off so the scatterplot has equal length arrays to compare (this is actually required by scatterplot)

# In[34]:


gdp_stage_trunc = gdp_stage[gdp_stage['Year'] < 2012]
print(len(gdp_stage_trunc))
print(len(stage))


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.set_title('CO2 Emissions vs. GDP \(per capita\)',fontsize=10)
axis.set_xlabel(gdp_stage_trunc['IndicatorName'].iloc[10],fontsize=10)
axis.set_ylabel(stage['IndicatorName'].iloc[0],fontsize=10)

X = gdp_stage_trunc['Value']
Y = stage['Value']

axis.scatter(X, Y)
plt.show()


# This doesn't look like a strong relationship.  We can test this by looking at correlation.

# In[36]:


np.corrcoef(gdp_stage_trunc['Value'],stage['Value'])

