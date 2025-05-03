
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np               # For exp, log etc.,


# # Step Activation Function
# 

# In[2]:


def step(x):
  if x>=0:
    return 1
  else:
    return 0
     

x=np.arange(-6,6,0.01)
step_output = [step(i) for i in x]

# Plotting
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x,step_output, color="#307EC7", linewidth=3, label="Step Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # Sigmoid Activation Function
# 

# In[3]:


def sigmoid(x):
  s = (1/(1+np.exp(-x)))   # sigmoid function
  return s
     

x=np.arange(-6,6,0.01)
# Plotting
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x,sigmoid(x), color="#307EC7", linewidth=3, label="Sigmoid Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # Tanh Activation Function
# 

# In[4]:


def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    # t = np.tanh(x)
    return t
     

x=np.arange(-6,6,0.01)
# Plotting
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x,tanh(x), color="#307EC7", linewidth=3, label="Tanh Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # ReLU Activation Function

# In[5]:


def relu(x):
  y = max(0,x)
  return y
     

x=np.arange(-6,6,0.01)
relu_output = [relu(i) for i in x]
# Plotting
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x,relu_output, color="#307EC7", linewidth=3, label="ReLU Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # Leaky ReLU Activation Function

# In[6]:


def leaky_relu(x):
  y = max(0.1*x,x)
  return y
     

x=np.arange(-6,6,0.01)
leaky_relu_output = [leaky_relu(i) for i in x]
# Plotting
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x,leaky_relu_output, color="#307EC7", linewidth=3, label="Leaky ReLU Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # Softplus Activation Function

# In[7]:


def softplus(x):
  s = np.log(1+np.exp(x))
  return s
     

x=np.arange(-6,6,0.01)
# Plotting
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x,softplus(x), color="#307EC7", linewidth=3, label="Softplus Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # GeLU Activation Function

# In[8]:


def sigmoid(x):
  s = (1/(1+np.exp(-x)))   # sigmoid function
  return s

def gelu(x):
  g = x * sigmoid(1.702*x)
  return g
     

z=np.arange(-6,6,0.01)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(z,gelu(z), color="#307EC7", linewidth=3, label="GeLU Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # Swish Activation Function

# In[9]:


def swish(x):
  s = x* (1/(1+np.exp(-x)))
  return s
     

z=np.arange(-6,6,0.01)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(z,swish(z), color="#307EC7", linewidth=3, label="Swish Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # Mish Activation Function

# In[10]:


def mish(x):
  m = x * np.tanh(np.log(1+np.exp(x)))
  return m
     

z=np.arange(-6,6,0.01)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(z,mish(z), color="#9621E2", linewidth=3, label="Mish Activation Function")
ax.legend(loc="upper left", frameon=False)
fig.show()


# # All At A Place

# In[11]:


z=np.arange(-6,6,0.01)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(z,step_output, linewidth=3, label="step")
ax.plot(z,sigmoid(z), linewidth=3, label="sigmoid")
ax.plot(z,tanh(z), linewidth=3, label="tanh")
ax.plot(z,relu_output, linewidth=3, label="relu")
ax.plot(z,leaky_relu_output, linewidth=3, label="leaky_relu")
ax.plot(z,softplus(z), linewidth=3, label="softplus")
ax.plot(z,gelu(z), linewidth=3, label="gelu")
ax.plot(z,swish(z), linewidth=3, label="swish")
ax.plot(z,mish(z), linewidth=3, label="mish")

ax.legend(loc="upper left", frameon=False)
fig.show()
