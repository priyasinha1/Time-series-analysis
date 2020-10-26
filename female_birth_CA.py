#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error
import numpy as np


# In[41]:


f_birth = pd.read_csv('female-births-CA.csv', index_col=[0],parse_dates=[0])


# In[42]:


f_birth.head()


# In[43]:


f_birth.describe()


# In[44]:


series_value = f_birth.values


# In[45]:


f_birth.tail()


# In[46]:


f_birth.plot()


# In[47]:


f_birth_mean = f_birth.rolling(window = 25).mean()


# In[48]:


f_birth.plot()
f_birth_mean.plot()


# In[49]:


value = pd.DataFrame(series_value)
birth_df= pd.concat([value,value.shift(1)],axis = 1)


# In[54]:


birth_df.columns = ['Actual_birth','Forecast_birth']


# In[55]:


birth_df.head()


# In[56]:


birth_test = birth_df[1:]


# In[57]:


birth_test.head(2)


# In[58]:


birth_error = mean_squared_error(birth_test.Actual_birth, birth_test.Forecast_birth)


# In[59]:


birth_error


# In[60]:


np.sqrt(birth_error)


# In[61]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[63]:


plot_acf(f_birth)


# In[64]:


plot_pacf(f_birth)


# In[65]:


birth_train = f_birth[0:330]
birth_test = f_birth[330:365]


# In[67]:


from statsmodels.tsa.arima_model import ARIMA


# In[70]:


birth_model = ARIMA(birth_train, order=(2,1,3))


# In[71]:


birth_model_fit = birth_model.fit()


# In[72]:


birth_model_fit.aic


# In[73]:


birth_forecast = birth_model_fit.forecast(steps = 35)[0]


# In[74]:


birth_forecast


# In[75]:


np.sqrt(mean_squared_error(birth_test,birth_forecast))


# In[ ]:




