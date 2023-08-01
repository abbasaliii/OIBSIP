#!/usr/bin/env python
# coding: utf-8

# ## UNEMPLOYMENT ANALYSIS WITH PYTHON

# ## Author Syed Abbas Ali

# ### Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# ### Load and Prepare the Data

# In[2]:


unemployment_rate_data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Unemployment_Rate_upto_11_2020.csv")
unemployment_rate_data.head(5)


# In[3]:


# Remove leading and trailing whitespaces from column names
unemployment_rate_data.columns = unemployment_rate_data.columns.str.strip()
print(unemployment_rate_data.columns)


# In[4]:


# Convert 'Date' column to datetime format and sort the data by date
unemployment_rate_data['Date'] = pd.to_datetime(unemployment_rate_data['Date'])
unemployment_rate_data = unemployment_rate_data.sort_values(by='Date')


# ### Data Exploration

# In[5]:


print("Data Shape:")
print(unemployment_rate_data.shape)
print("Data Info:")
print(unemployment_rate_data.info())
print("Missing Values:")
print(unemployment_rate_data.isnull().sum())


# ### Exploratory Data Analysis

# In[6]:


# Overall trend of the unemployment rate over time
fig1 = px.line(unemployment_rate_data, x='Date', y='Estimated Unemployment Rate (%)', 
               title='Overall Trend of Unemployment Rate')
fig1.show()


# In[7]:


# Bar plot comparing the average unemployment rate among different regions
average_unemployment_rate = unemployment_rate_data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()
fig2 = px.bar(average_unemployment_rate, x='Region', y='Estimated Unemployment Rate (%)', 
              title='Average Unemployment Rate by Region')
fig2.show()


# In[8]:


# Scatter plot to show how COVID-19 affected the employment rate
fig3 = px.scatter(unemployment_rate_data, x='Date', y='Estimated Employed', color='Region',
                  title='Impact of COVID-19 on Employment Rate', trendline='ols')
fig3.update_layout(xaxis_title='Date', yaxis_title='Estimated Employed (%)')
fig3.show()


# In[9]:


# Line plot to show the unemployment rate in each region over time
fig5 = px.line(unemployment_rate_data, x='Date', y='Estimated Unemployment Rate (%)', color='Region',
               title='Unemployment Rate by Region over Time')
fig5.show()


# In[10]:


# Bar plot to show the estimated employed percentage in each region
fig6 = px.bar(unemployment_rate_data, x='Region', y='Estimated Employed', color='Region',
              title='Estimated Employed Percentage by Region')
fig6.show()


# In[11]:


fig8 = px.bar(unemployment_rate_data, x='Region', y='Estimated Labour Participation Rate (%)',
              title='Labour Participation Rate by Region', template='plotly')
fig8.show()


# In[12]:


# Scatter plot to show the relationship between the unemployment rate and estimated employed percentage
fig9 = px.scatter(unemployment_rate_data, x='Estimated Employed', y='Estimated Unemployment Rate (%)',
                  color='Region', title='Unemployment Rate vs. Estimated Employed Percentage')
fig9.show()


# In[ ]:




