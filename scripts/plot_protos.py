import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame from CSV data
df = pd.read_csv('/share/portal/pd337/xskill/experiment/pretrain/orig_batch/robot_proto_task_relation_79.csv')

# Filter data for tasks 'kettle', 'microwave', 'hinge cabinet', and 'your_task'
tasks_of_interest = ['kettle', 'microwave', 'hinge cabinet', 'slide cabinet']
filtered_df = df[df['task'].isin(tasks_of_interest)]

grouped_df = filtered_df.groupby(['task', 'max_proto']).size().reset_index(name='count')

# Get unique tasks
tasks = grouped_df['task'].unique()

# Define colors for each task
colors = {'kettle': 'blue', 'microwave': 'black', 'hinge cabinet': 'red', 'slide cabinet': 'orange'}

# Plot counts for each task
plt.figure(figsize=(10, 6))
for task in tasks:
    task_data = grouped_df[grouped_df['task'] == task]
    task_data_sorted = task_data.sort_values(by='count', ascending=False).head(5)  # Select top 5 most frequent max_proto
    task_data_sorted = task_data_sorted.sort_values(by='max_proto', ascending=True)
    plt.scatter(task_data_sorted['max_proto'], task_data_sorted['count'], label=task, color=colors[task], s=100)  # Increase the size of dots

plt.title('Count of max_proto Numbers for Each Task (Top 5) - ROBOT')
plt.xlabel('max_proto')
plt.ylabel('Count')
plt.legend(title='Task')
plt.grid(True)
plt.tight_layout()
plt.savefig('robot.png')
plt.show()
