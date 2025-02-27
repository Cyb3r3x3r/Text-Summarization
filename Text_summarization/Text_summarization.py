#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset


# In[2]:


dataset = load_dataset("cnn_dailymail","3.0.0")


# In[3]:


dataset['train']


# In[4]:


dataset['train'][5]['article'][:200]


# In[5]:


dataset['train'][5]['highlights']


# In[6]:


from transformers import pipeline


# In[7]:


import torch
device = 0 if torch.cuda.is_available() else -1


# In[8]:


#Creating the pipeline
pipe = pipeline("text-generation",model="gpt2-medium",device=device)


# In[9]:


#Preparing the data and inserting into pip
dataset['train'][5]['article'][:2500]
input_text = dataset['train'][5]['article'][:2500]
query = input_text + "\nTL;DR:\n"
pipe_out = pipe(query,max_length=1024,clean_up_tokenization_spaces=True)


# In[10]:


summary = pipe_out[0]['generated_text'][len(query):]


# In[11]:


all_summary ={}
all_summary['gpt2-medium'] = summary
#print(f"Summary :{summary}")


# In[12]:


#Trying out the google pegasus model
pipePegasus = pipeline('summarization',device=device,model='google/pegasus-cnn_dailymail')


# In[13]:


result = pipePegasus(input_text)


# In[14]:


all_summary['pegasus_cnn'] = result[0]['summary_text']


# In[15]:


# for model in all_summary:
#     print(model.upper())
#     print(all_summary[model])
#     print("")


# In[16]:


#Trying with custom text
input_text = "Neelam Shinde, who hails from Maharashtra's Satara district, was critically injured in the accident on February 14 and has been in a coma at a hospital since then. Neelam Shinde has reportedly suffered severe injuries to her head, hand, and chest. Her family has sought an urgent visa to travel to the US to be by her side.The MEA has taken up the matter with the US.The US side is looking into the formalities for early grant of visa for the applicant's family,the sources cited in the PTI report said.Nationalist Congress Party (Sharadchandra Pawar) leader Supriya Sule flagged the case on Wednesday.Student Neelam Shinde has met with an accident in the USA and is hospitalised in a local hospital, she said in a post on X."


# In[17]:


result = pipePegasus(input_text)
print(f"Summary with pegasus_cnn: {result[0]['summary_text']}")





