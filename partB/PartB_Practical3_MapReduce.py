# ============================================================
# FILE: PartB_Practical3_MapReduce_Weather.py
# STANDALONE FILE — No other files needed.
# SUBJECT: Distributed Computing (DC)
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install numpy matplotlib pandas
#   (No Hadoop needed — we simulate MapReduce in pure Python)
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open colab.research.google.com → New Notebook
#   2. First cell: !pip install numpy matplotlib pandas
#   3. Paste entire code into next cell → Shift+Enter
#   4. Output: mapper output, reducer output, hottest/coolest year
#   5. Bar chart saved as 'mapreduce_weather.png'
#
# OPTION B — PyCharm / Terminal:
#   pip install numpy matplotlib pandas
#   python PartB_Practical3_MapReduce_Weather.py
#
# REAL HADOOP VERSION (optional, if Hadoop is installed):
#   The Java code from the lab manual requires:
#   1. javac -classpath `hadoop classpath` -d weather_classes *.java
#   2. jar -cvf weather.jar -C weather_classes/ .
#   3. hdfs dfs -mkdir /weather && hdfs dfs -put weather.csv /weather
#   4. hadoop jar weather.jar WeatherDriver /weather /weather_output
#   5. hdfs dfs -cat /weather_output/part-r-00000
#   This Python version simulates the exact same MapReduce logic.
#
# EXPECTED OUTPUT:
#   Mapper output: (year, temperature) pairs
#   Shuffle/Sort: grouped by year
#   Reducer output: (year, average_temperature) pairs
#   Hottest Year: XXXX with avg temp XX.X°C
#   Coolest Year: XXXX with avg temp XX.X°C
#   Bar chart showing yearly average temperatures
# ============================================================

# Import collections for grouping key-value pairs (simulating shuffle/sort)
from collections import defaultdict
# Import random for generating synthetic weather data
import random
# Import numpy for average computation
import numpy as np
# Import matplotlib for bar chart visualization
import matplotlib.pyplot as plt
# Import itertools for grouping (simulates Hadoop's shuffle and sort)
import itertools
# Import pandas for real dataset loading from CSV URL/local file
import pandas as pd

# Set random seed for reproducible weather data
random.seed(42)
np.random.seed(42)

# ============================================================
# STEP 1: DATASET LOADING (REAL DATA FIRST, SYNTHETIC FALLBACK)
# ============================================================
def load_real_weather_data():
    """
    REAL DATASET SECTION (preferred):
    Loads daily minimum temperatures dataset and converts it to
    "Year,Month,Day,Temperature" CSV records for MapReduce.
    """
    dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(dataset_url)

    # Local fallback path (keep commented; use if URL is unavailable)
    # df = pd.read_csv(r"U:\ci_dc_practical\data\daily-min-temperatures.csv")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Temp"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    records = [
        f"{int(row.Year)},{int(row.Month)},{int(row.Day)},{float(row.Temp):.1f}"
        for _, row in df.iterrows()
    ]
    return records

def generate_weather_data(start_year=2001, end_year=2015, records_per_year=50):
    """
    SYNTHETIC DATASET SECTION (backup / offline simulation):
    Generates synthetic weather data in CSV format.
    Format: Year, Month, Day, Temperature
    In real Hadoop usage: this data would come from NOAA or Kaggle datasets
    stored in HDFS (Hadoop Distributed File System).
    """
    weather_records = []  # List to store all weather records as strings
    print("Generating synthetic weather dataset...")

    for year in range(start_year, end_year + 1):
        # Base temperature varies by year to create interesting hottest/coolest pattern
        base_temp = 20 + (year - start_year) * 0.5 + random.uniform(-5, 5)

        for _ in range(records_per_year):
            month = random.randint(1, 12)       # Random month 1-12
            day = random.randint(1, 28)         # Random day 1-28 (safe for all months)
            # Temperature varies around the base with seasonal + random noise
            temp = base_temp + random.uniform(-15, 15)  # ±15°C variation
            temp = round(temp, 1)               # Round to 1 decimal place

            # Format as CSV string: Year,Month,Day,Temperature
            record = f"{year},{month},{day},{temp}"
            weather_records.append(record)

    print(f"Generated {len(weather_records)} weather records "
          f"({start_year}–{end_year}, {records_per_year} per year)")
    return weather_records  # Return list of CSV strings

# ============================================================
# STEP 2: MAPPER FUNCTION
# ============================================================
def mapper(record):
    """
    MAP PHASE: Processes a single input record and emits key-value pairs.
    Input:  A single CSV line — "2001,01,15,32.5"
    Output: A (key, value) pair — ("2001", 32.5)
    
    In Hadoop: each mapper runs on a separate data node, processing
    a split (chunk) of the input file in parallel. Results are
    written to local disk before shuffle phase.
    """
    try:
        # Split CSV record into fields
        fields = record.strip().split(',')

        # Validate that we have exactly 4 fields
        if len(fields) != 4:
            return None  # Skip malformed records

        # Extract year (key) and temperature (value)
        year = fields[0].strip()           # Year is the grouping key
        temperature = float(fields[3].strip())  # Temperature is the value

        # Emit (key, value) pair — year as key, temperature as value
        return (year, temperature)

    except (ValueError, IndexError):
        # Skip records with invalid format or non-numeric temperature
        return None

# ============================================================
# STEP 3: SHUFFLE AND SORT PHASE (Hadoop does this automatically)
# ============================================================
def shuffle_and_sort(mapped_pairs):
    """
    SHUFFLE AND SORT PHASE: Groups all values by their key.
    Input:  List of (year, temperature) pairs from all mappers
    Output: Dictionary {year: [temp1, temp2, temp3, ...]}
    
    In Hadoop: Framework automatically sorts mapper outputs by key
    and sends all values for a given key to the same reducer.
    Data travels over the network from mappers to reducers.
    This is the most expensive phase (network I/O intensive).
    """
    # Group temperature values by year using defaultdict
    grouped = defaultdict(list)  # {year: [list of temperatures]}

    for pair in mapped_pairs:
        if pair is not None:           # Skip any None results from mapper
            year, temp = pair          # Unpack the (year, temperature) tuple
            grouped[year].append(temp) # Add temperature to this year's list

    # Sort the grouped dict by year (key) — simulates Hadoop sort phase
    sorted_grouped = dict(sorted(grouped.items()))

    return sorted_grouped  # Returns {year: [temps]} sorted by year

# ============================================================
# STEP 4: REDUCER FUNCTION
# ============================================================
def reducer(year, temperatures):
    """
    REDUCE PHASE: Aggregates all values for a given key.
    Input:  year (key) and list of all temperatures for that year
    Output: (year, average_temperature) — one result per year
    
    In Hadoop: Each reducer processes one key and its values.
    Multiple reducers can run in parallel (one per unique year).
    Output written to HDFS output directory.
    """
    # Compute average temperature for this year
    avg_temp = sum(temperatures) / len(temperatures)  # Simple mean
    avg_temp = round(avg_temp, 2)  # Round to 2 decimal places

    # Emit the final (year, avg_temp) result
    return (year, avg_temp)

# ============================================================
# STEP 5: MAIN MAPREDUCE DRIVER
# ============================================================
def mapreduce_weather_analysis(weather_data):
    """
    Orchestrates the complete MapReduce pipeline:
    Input Data → Map → Shuffle & Sort → Reduce → Output
    Equivalent to Hadoop's WeatherDriver class from the lab manual.
    """
    print("\n" + "="*60)
    print("MAPREDUCE PIPELINE — Weather Temperature Analysis")
    print("="*60)

    # ── MAP PHASE ──────────────────────────────────────────
    print("\n[MAP PHASE]")
    print("Processing records: extracting (year, temperature) pairs...")

    # Apply mapper to every input record (in Hadoop: runs in parallel on cluster)
    mapped_output = [mapper(record) for record in weather_data]

    # Filter out None results (malformed records)
    mapped_output = [pair for pair in mapped_output if pair is not None]

    # Show sample mapper output (first 10 pairs)
    print(f"Mapper emitted {len(mapped_output)} (key, value) pairs")
    print("Sample mapper output (first 5 pairs):")
    for pair in mapped_output[:5]:
        print(f"  ({pair[0]}, {pair[1]})")

    # ── SHUFFLE AND SORT PHASE ─────────────────────────────
    print("\n[SHUFFLE AND SORT PHASE]")
    print("Grouping temperatures by year...")
    grouped_data = shuffle_and_sort(mapped_output)

    # Show grouped data summary
    print(f"Grouped into {len(grouped_data)} keys (years)")
    print("Sample grouped data:")
    for year, temps in list(grouped_data.items())[:3]:
        print(f"  Year {year}: {len(temps)} temperature records "
              f"[{temps[0]}, {temps[1]}, ... {temps[-1]}]")

    # ── REDUCE PHASE ───────────────────────────────────────
    print("\n[REDUCE PHASE]")
    print("Computing average temperature per year...")

    # Apply reducer to each group (in Hadoop: runs in parallel, one reducer per key)
    reducer_output = {}
    for year, temperatures in grouped_data.items():
        year_key, avg_temp = reducer(year, temperatures)  # Reduce this year's data
        reducer_output[year_key] = avg_temp  # Store result

    # Display full reducer output (like: hdfs dfs -cat /output/part-r-00000)
    print("\nReducer Output (Year → Average Temperature):")
    print("-"*35)
    print(f"{'Year':<8} {'Avg Temp (°C)':>15}")
    print("-"*35)
    for year, avg_temp in reducer_output.items():
        print(f"{year:<8} {avg_temp:>15.2f}")
    print("-"*35)

    return reducer_output  # Return dictionary {year: avg_temp}

# ============================================================
# STEP 6: POST-PROCESSING — FIND HOTTEST AND COOLEST YEAR
# ============================================================
def find_hottest_coolest(reducer_output):
    """
    Post-processing step: identifies the hottest and coolest years
    from the reducer output. In Hadoop: this would be another MapReduce job
    or done in the client after fetching reducer output from HDFS.
    """
    # Find year with maximum average temperature
    hottest_year = max(reducer_output, key=reducer_output.get)
    hottest_temp = reducer_output[hottest_year]

    # Find year with minimum average temperature
    coolest_year = min(reducer_output, key=reducer_output.get)
    coolest_temp = reducer_output[coolest_year]

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"🌡  HOTTEST YEAR: {hottest_year} → Avg Temp = {hottest_temp:.2f}°C")
    print(f"❄️  COOLEST YEAR: {coolest_year} → Avg Temp = {coolest_temp:.2f}°C")
    print(f"Temperature Range: {coolest_temp:.2f}°C to {hottest_temp:.2f}°C")

    return hottest_year, hottest_temp, coolest_year, coolest_temp

# ============================================================
# STEP 7: VISUALIZATION
# ============================================================
def visualize_results(reducer_output, hottest_year, coolest_year):
    """Plots a bar chart of average temperatures per year."""
    years = list(reducer_output.keys())      # X-axis: years
    avg_temps = list(reducer_output.values()) # Y-axis: average temperatures

    # Assign colors: red for hottest, blue for coolest, gray for others
    colors = []
    for year in years:
        if year == hottest_year:
            colors.append('#FF4444')    # Red for hottest
        elif year == coolest_year:
            colors.append('#4444FF')    # Blue for coolest
        else:
            colors.append('#90CAF9')    # Light blue for others

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create bar chart
    bars = ax.bar(years, avg_temps, color=colors, edgecolor='black', linewidth=0.5)

    # Add temperature values on top of each bar
    for bar, temp in zip(bars, avg_temps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{temp:.1f}°C',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add labels and title
    ax.set_title('MapReduce Weather Analysis — Average Temperature per Year\n'
                 '(Red = Hottest, Blue = Coolest)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Temperature (°C)', fontsize=12)
    ax.set_xticklabels(years, rotation=45, ha='right')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF4444', label=f'Hottest: {hottest_year}'),
        Patch(facecolor='#4444FF', label=f'Coolest: {coolest_year}'),
        Patch(facecolor='#90CAF9', label='Other years')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('mapreduce_weather.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nChart saved as 'mapreduce_weather.png'")

# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    # Try real dataset first (URL), then synthetic fallback
    try:
        weather_data = load_real_weather_data()
        print("Dataset source: REAL dataset from URL (daily-min-temperatures)")
    except Exception as e:
        print(f"URL dataset load failed ({e}). Falling back to synthetic weather data...")
        weather_data = generate_weather_data(
            start_year=2001,
            end_year=2015,
            records_per_year=50
        )
        print("Dataset source: SYNTHETIC generated weather data")

    # Run the MapReduce pipeline
    yearly_avg = mapreduce_weather_analysis(weather_data)

    # Find hottest and coolest year
    h_year, h_temp, c_year, c_temp = find_hottest_coolest(yearly_avg)

    # Visualize
    visualize_results(yearly_avg, h_year, c_year)


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Data Generation: Synthetic CSV weather records (Year,Month,Day,Temp)
#    simulating what would be stored in HDFS across multiple data nodes
# 2. MAP PHASE: mapper() processes each record → emits (year, temperature) pair
#    In Hadoop: thousands of mappers run in parallel across the cluster
# 3. SHUFFLE & SORT: defaultdict groups all temperatures by year
#    In Hadoop: framework automatically sorts by key and routes to correct reducer
# 4. REDUCE PHASE: reducer() computes average temperature for each year
#    In Hadoop: one reducer per year runs in parallel
# 5. Post-processing: max/min on reducer output finds hottest/coolest year
# 6. Visualization: bar chart shows all yearly averages with hottest/coolest highlighted
# Python simulation correctly models the Map→Shuffle→Reduce data flow
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: MapReduce for Distributed Data Processing
# MapReduce is a programming model for processing large datasets in parallel
# across a distributed cluster. Developed by Google, implemented in Apache Hadoop.
# Two phases:
#   Map: Transforms input records into (key, value) pairs (parallelizable)
#   Reduce: Aggregates all values for each key (parallelizable per key)
# Hadoop ecosystem:
#   HDFS: Distributed storage — files split into 128MB blocks across nodes
#   YARN: Resource manager — schedules MapReduce jobs across the cluster
#   MapReduce: Processing framework — runs Map and Reduce tasks on data nodes
# Real-world use: log analysis, word count, recommendation systems,
# search index building, financial data aggregation, scientific computing.
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is MapReduce?
# A1. MapReduce is a programming model and framework for processing large datasets
#     in parallel across a distributed cluster. Two main phases:
#     Map: Input records → (key, value) pairs (runs in parallel on all nodes)
#     Reduce: Groups (key, [values]) → aggregated output (parallel per key)
#     Originally developed at Google (2004 paper by Dean & Ghemawat).
#     Apache Hadoop is the open-source implementation.
#
# Q2. What is the difference between HDFS and MapReduce?
# A2. HDFS (Hadoop Distributed File System): The STORAGE layer.
#     - Splits large files into 128MB blocks, replicates across nodes (default 3x)
#     - Provides fault tolerance — if one node fails, blocks available on others
#     - Optimized for large sequential reads, not random access
#     MapReduce: The PROCESSING layer.
#     - Programming model that processes data stored in HDFS
#     - Moves computation to data (data locality) rather than data to computation
#     - HDFS stores; MapReduce computes. Both together form the Hadoop stack.
#
# Q3. What happens during the Shuffle and Sort phase?
# A3. After all Mappers complete, Shuffle and Sort:
#     (i) Collects all (key, value) pairs from all mapper outputs
#     (ii) Sorts all pairs by key across the entire cluster
#     (iii) Groups all values with the same key together
#     (iv) Sends each group to exactly one Reducer (via network transfer)
#     This is the most expensive phase — heavy network I/O.
#     In Hadoop: handled automatically by the framework (no user code needed).
#     Optimization: Combiner function can pre-aggregate on mapper side to reduce data transfer.
#
# Q4. Why is distributed processing needed for weather data?
# A4. Weather datasets are massive:
#     - NOAA global dataset: terabytes of historical data (100+ years, worldwide)
#     - Real-time sensor data: millions of readings per day globally
#     Single machine limitations:
#     - Cannot store petabytes of data (storage limit)
#     - Processing takes too long (days on single machine vs hours on cluster)
#     - Single point of failure — no fault tolerance
#     Distributed MapReduce:
#     - Splits data across hundreds of nodes, processes in parallel
#     - Fault tolerance via HDFS replication and task re-execution
#     - Linear scaling: 2x nodes → ~2x faster processing
#
# Q5. What is a Combiner in MapReduce and why is it used?
# A5. Combiner is a mini-Reducer that runs on the mapper's output BEFORE shuffle.
#     It pre-aggregates values with the same key on the local mapper node.
#     This significantly reduces the amount of data transferred during shuffle.
#     Example: Instead of sending (2001, 32), (2001, 35), (2001, 28) over network,
#     combiner computes local average/sum first: (2001, 31.67) — one value per key.
#     Not always applicable: only when reduce operation is associative and commutative.
#     For average: need sum + count (not just average) to combine correctly.
#
# Q6. What is data locality in Hadoop?
# A6. Data locality means moving computation to where data resides, not vice versa.
#     In HDFS: each 128MB block is stored on specific nodes.
#     Hadoop's scheduler tries to run mapper tasks on the same node storing the data.
#     Levels: node-local (same node), rack-local (same rack), cross-rack (worst).
#     Benefit: avoids expensive network transfer of large data blocks.
#     Network bandwidth is the main bottleneck in distributed systems.
#
# Q7. What is the role of YARN in Hadoop?
# A7. YARN (Yet Another Resource Negotiator) is Hadoop's cluster resource manager.
#     Components:
#     ResourceManager: Master — manages cluster resources, schedules applications
#     NodeManager: Worker daemon on each node — manages containers, reports resources
#     ApplicationMaster: Per-job manager — negotiates resources, monitors tasks
#     YARN allows multiple frameworks (MapReduce, Spark, Tez) to share the same cluster.
#     Before YARN (Hadoop 1.x): JobTracker did everything — single point of failure.
#
# Q8. What is a Mapper class in Java Hadoop MapReduce?
# A8. In Java Hadoop: Mapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT> base class.
#     KEYIN: Input key type (usually LongWritable — byte offset in file)
#     VALUEIN: Input value type (Text — one line of input file)
#     KEYOUT: Output key type (Text — year string in weather example)
#     VALUEOUT: Output value type (FloatWritable — temperature)
#     Override map() method: reads one record, calls context.write(key, value).
#     Each input split (typically one HDFS block) = one Mapper task.
#
# Q9. What is fault tolerance in MapReduce?
# A9. MapReduce handles failures automatically:
#     Task failure: ApplicationMaster detects heartbeat timeout → re-executes task
#     on another node. Mapper outputs stored locally; reducer fetches from surviving mappers.
#     Node failure: HDFS replication ensures data still accessible on other nodes.
#     Master failure: YARN ResourceManager has HA (High Availability) with standby.
#     Speculative execution: Slow tasks get duplicate copies started; first to finish wins.
#     This allows running reliably on cheap commodity hardware that fails frequently.
#
# Q10. How does the Weather MapReduce program work end-to-end?
# A10. 1. Weather CSV data uploaded to HDFS: hdfs dfs -put weather.csv /weather
#      2. Hadoop splits file into blocks; schedules one Mapper per block
#      3. Each Mapper: reads lines, extracts year+temp, emits (year, temp) pairs
#      4. Shuffle & Sort: collects all pairs, sorts by year, groups → (year, [temps])
#      5. Reducers: receive (year, [temps]), compute average → write (year, avg)
#      6. Output: stored in HDFS /weather_output/part-r-00000
#      7. Client: reads output, finds max (hottest) and min (coolest) year
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# MapReduce is a distributed batch-processing paradigm optimized for huge datasets.
#
# Why it scales:
#   - Data split into blocks.
#   - Map tasks run parallel near data (data locality).
#   - Reduce tasks aggregate by key after shuffle/sort.
#
# Cost insight:
#   Shuffle is network-heavy and often dominates runtime.
#   Good mapper output design + combiners reduces shuffle cost.
#
# Practical weather use case:
#   Key = year, Value = temperature.
#   Reducer aggregates temperatures per year into average.
#
# Hadoop ecosystem context:
#   HDFS for storage, YARN for cluster resource scheduling, MapReduce for execution.
#
# Exam-ready line:
# "MapReduce transforms large unstructured records into key-grouped summaries with
# fault-tolerant distributed execution."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. Why must reducer logic be associative/commutative for combiner benefit?
# A11. Because partial local aggregation must remain mathematically valid when merged globally.
#
# Q12. What is an input split?
# A12. A logical chunk of input assigned to one mapper, usually aligned with HDFS blocks.
#
# Q13. What is speculative execution?
# A13. Hadoop runs duplicate slow tasks; first to finish wins, reducing straggler impact.
#
# Q14. How does MapReduce ensure fault tolerance?
# A14. Failed tasks are re-executed on other nodes, and HDFS block replication preserves data.
#
# Q15. Why MapReduce for batch but not interactive analytics?
# A15. Startup/shuffle overhead is high; for low-latency iterative workloads Spark is usually faster.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) MAP STEP
#    - Converts raw record into key-value pair.
#    - Parallel by data splits.
#
# 2) SHUFFLE/SORT STEP
#    - Groups all values of same key.
#    - Network-heavy, often the most expensive phase.
#
# 3) REDUCE STEP
#    - Aggregates grouped values into final summaries.
#    - Parallel by key groups.
#
# 4) WHY HDFS + MAPREDUCE TOGETHER
#    - HDFS stores huge files distributed with replication.
#    - MapReduce runs computation close to those blocks.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "Mapper emits (year,temp), shuffle groups by year, reducer computes average
#     temperature per year; final output is used to find hottest/coolest year."
# ============================================================
