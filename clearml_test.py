import time
import numpy as np
from clearml import Task

# Connect to the ClearML server and create a task
task = Task.init(project_name="Remote Execution Test", 
                 task_name="Test Task", 
                 task_type=Task.TaskTypes.optimizer,  # You can also use TaskTypes.container, or TaskTypes.custom
                 reuse_last_task_id=False)

# Log some parameters and metrics
task.connect({'learning_rate': 0.001, 'batch_size': 32})
task.execute_remotely(queue_name="default")

# Example: Simulate a training loop
for epoch in range(5):
    # Simulating some training progress
    loss = np.random.random()
    accuracy = np.random.random()

    # Log metrics to ClearML
    task.logger.report_scalar("Loss", "train", iteration=epoch, value=loss)
    task.logger.report_scalar("Accuracy", "train", iteration=epoch, value=accuracy)

    # Sleep to simulate training time
    time.sleep(1)


print("Test task completed!")
