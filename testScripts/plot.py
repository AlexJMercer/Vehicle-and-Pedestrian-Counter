import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import defaultdict
from datetime import datetime, timedelta

# Data provided
data = [(6, '15:24:39'), (1, '15:24:39'), (2, '15:24:39'), (3, '15:24:39'), (7, '15:24:41'), (8, '15:24:44'), 
        (10, '15:24:50'), (11, '15:24:50'), (9, '15:24:50'), (12, '15:25:01'), (13, '15:25:01'), (14, '15:25:01'), 
        (15, '15:25:05'), (16, '15:25:07'), (19, '15:25:12'), (18, '15:25:12'), (17, '15:25:12'), (21, '15:25:14'), 
        (20, '15:25:14'), (22, '15:25:16'), (24, '15:25:23'), (23, '15:25:23'), (26, '15:25:29'), (25, '15:25:29'), 
        (29, '15:25:36'), (28, '15:25:36'), (27, '15:25:36'), (30, '15:25:36'), (32, '15:25:40'), (31, '15:25:40'), 
        (34, '15:25:43'), (35, '15:25:45'), (36, '15:25:51'), (37, '15:25:52'), (39, '15:25:56'), (40, '15:25:58'), 
        (41, '15:26:02')]

interval_time = 10

# Process data
interval = timedelta(seconds=interval_time)
time_group = defaultdict(set)

# Grouping the vehicles by time
for vehicle_id, time in data:
    timeStamp = datetime.strptime(time, '%H:%M:%S')
    group_time = timeStamp - timedelta(seconds = timeStamp.second % interval.seconds)
    time_group[group_time].add(vehicle_id)

# Sort groups before plotting
times, counts = zip(*[(time, len(vehicle_ids)) for time, vehicle_ids in sorted(time_group.items())])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(times, counts, marker='o', linestyle='-', color='b')

plt.title('Number of Vehicles vs Time')
plt.xlabel('Time')
plt.ylabel('Number of Vehicles')

plt.xticks(rotation=45)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=interval_time))

plt.grid(True)
plt.tight_layout()

plt.show()