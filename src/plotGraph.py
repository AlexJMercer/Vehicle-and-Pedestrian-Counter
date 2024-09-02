
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from collections import defaultdict


def plot_graph_vehicle_count(vehicleCountData):

    # Set the interval of the time axis
    interval_time = 10
    
    # Process vehicleCountData
    interval = timedelta(seconds=interval_time)
    time_group = defaultdict(set)


    # Grouping the vehicles by time
    for vehicle_id, time in vehicleCountData:
        timeStamp = datetime.strptime(time, '%H:%M:%S')
        group_time = timeStamp - timedelta(seconds = timeStamp.second % interval.seconds)
        time_group[group_time].add(vehicle_id)


    # Sort groups before plotting
    times, counts = zip(*[(time, len(vehicle_ids)) for time, vehicle_ids in sorted(time_group.items())])


    # Plot the final graph
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
    