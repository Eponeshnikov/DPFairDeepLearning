#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from clearml import Task, Dataset, TaskTypes
import time
import pytz
import datetime
from dateutil.tz import tzutc
import configparser
import json


# In[ ]:


def get_credentials(dir_, file):
    if file is not None:
        config = configparser.ConfigParser()
        config.read(f'{dir_}/{file}.conf')

        # Get the values of the different settings
        api_server = config.get('settings', 'api_server')
        web_server = config.get('settings', 'web_server')
        files_server = config.get('settings', 'files_server')

        # Parse the credentials JSON string
        credentials_json = config.get('settings', 'credentials')
        credentials = json.loads(credentials_json)
        access_key = credentials['access_key']
        secret_key = credentials['secret_key']
        return api_server, web_server, files_server, access_key, secret_key
    else:
        return None, None, None, None, None


# In[ ]:


configuration = {
    'config_dir': 'configs',
    'server': None,
    'time_threshold': 60,
    'project_name': "AI Fairness",
    'task_filter': {'system_tags': ['archived']}
}


# In[ ]:


api_server, web_server, files_server, access_key, secret_key = get_credentials(configuration['config_dir'], configuration['server'])
if api_server is not None:
    Task.set_credentials(
        api_host=api_server, web_host=web_server, files_host=files_server,
        key=access_key, secret=secret_key
     )
else:
    print('Use default clearml.conf')


# In[ ]:


# Set up the ClearML task
task = Task.init(project_name='Automation', task_name='Archive Drafts Tasks Deleting', task_type=TaskTypes.service)
task.connect(configuration)


# Define a function to check the status of a task and delete it if it has been in 'created' status for more than the time threshold
def check_and_delete_task(task_id, time_threshold):
    current_time = datetime.datetime.utcnow().replace(tzinfo=tzutc())
    task_to_check = Task.get_task(task_id=task_id)
    if task_to_check.status == 'created' and (current_time - task_to_check.export_task()['last_change']).total_seconds() > time_threshold:
        print(f"Task {task_id} has been in 'created' status for more than {time_threshold} seconds. Deleting task...")
        task_to_check.delete()
        print(f"Task {task_id} deleted.")
        
def archive_and_delete_tasks(project_name, task_filter, time_threshold):
    # Get all archived tasks in the project
    archived_tasks = Task.get_tasks(project_name=project_name, task_filter=task_filter)
    # Loop through each task in the archive dataset
    for task in archived_tasks:
        check_and_delete_task(task.id, time_threshold)


# In[ ]:


# Call the archive_and_delete_tasks function
while True:
    archive_and_delete_tasks(configuration['project_name'], configuration['task_filter'], configuration['time_threshold'])
    time.sleep(30)


# In[ ]:




