# Amplifier: A Hybrid Quantum-Classical Sampling Framework for Approximate Query Processing

## How to test

### Environment

First, you need to install the following Python dependencies.

```sh
pip install qiskit qiskit_aer numpy pandas argparse
```

### using qiskit-simu

You can use the following code to execute quantum sampling simulated by Qiskit. In this process, the parameters `P`, `G`, `M`, `op`, and `P_V` control a SQL `select G, agg(M) from dataset where P op P_V group by G`.

`s0` represents the number of times Amplifier is initially sampled classically, and `lowest` indicates that Amplifier will sample each group at least `lowest` times.

```bash
python3 amplifier.py \
    -dataset dataset/demo-test.csv \
    -P P \
    -G G \
    -M M \
    -P_V 1 \
    -op '<' \
    -s0 100 \
    -lowest 20 \
    -MultiTest 1 \
    -min_data_prob 0.1 \
    -min_shots 30 \
    -fast-simu 0 \
    -sup_shots 1.3
```

After execution, the program will output the estimation errors of AQP under three aggregation operators: avg, count, and sum (including results from both uniform sampling and stratified sampling for comparison).

```sh
=======================================================================================
quantum sample data have been stored into: sample_dt.csv
all_sample_times  : [100, 189, 39]
all_sample_amplify: [0, 1, 2]
sample rate is    :  0.2607421875
=======================================================================================
Query: select G, mean(M) from dataset where P < 1 group by G
real answer(mean):  [3.125      3.1        3.55555556 3.88888889 2.875      3.
 3.84615385 3.625     ]
Amplifier          (mean) avg error:  0.05221746698766
universe Sampling  (mean) avg error:  0.24923340191223511
Stratified Sampling(mean) avg error:  0.3717538606334588
=======================================================================================
=======================================================================================
Query: select G, count(M) from dataset where P < 1 group by G
real answer(count):  [16 10 18 18 16 13 13 16]
Amplifier          (count) avg error:  0.134642094017094
universe Sampling  (count) avg error:  0.2921476359678607
Stratified Sampling(count) avg error:  0.23648504273504273
=======================================================================================
=======================================================================================
Query: select G, sum(M) from dataset where P < 1 group by G
real answer(sum):  [50 31 64 70 46 39 50 58]
Amplifier          (sum) avg error:  0.15823087684074882
universe Sampling  (sum) avg error:  0.3174976478306564
Stratified Sampling(sum) avg error:  0.2894106325099812
=======================================================================================
```

**Please note that during testing**, it's important to control the size of the dataset and the range of values for each attribute(`P`, `G`, `M`), as the speed of the simulator increases exponentially with the number of qubits used. Excessively large datasets may result in significantly slow simulation speeds.

### multiple predicate

Example SQL: `select G, agg(M) from dataset where P < 1 and P >= 0 group by G`

```bash
python3 amplifier.py \
    -dataset dataset/demo-test.csv \
    -P P P \        # array
    -G G \
    -M M \
    -P_V 1 0 \      # array
    -op '<' '>=' \  # array
    -s0 100 \
    -lowest 20 \
    -MultiTest 1 \
    -min_data_prob 0.1 \
    -min_shots 30 \
    -fast-simu 0 \
    -sup_shots 1.3
```


### using fast-simu

If you want faster execution speed, you can enabling the `-fast-simu` switch, which will utilize a classical method to **simulate** the amplitude amplification to provide faster execution results.

**SQL: `select DayofMonth, agg(Distance) from flights where Origin=19 group by DayofMonth`**

**Query selectivity is about 0.01 and groups are relatively balance**

```bash
python3 amplifier.py \
    -dataset dataset/flights-rep4.csv \
    -P Origin \
    -G DayofMonth \
    -M Distance \
    -P_V 19 \
    -op '=' \
    -s0 1000 \
    -lowest 80 \
    -MultiTest 1 \
    -min_data_prob 0.1 \
    -min_shots 30 \
    -fast-simu 1
```

**SQL: `select officer_race, AVG(driver_age) FROM police WHERE road_number=2 GROUP BY officer_race`**

**Query selectivity is about 0.01 and groups are very imbalance**

```bash
python3 amplifier.py \
    -dataset dataset/police-rep4.csv \
    -P road_number \
    -G officer_race \
    -M driver_age \
    -P_V 2 \
    -op '=' \
    -s0 1000 \
    -lowest 200 \
    -MultiTest 1 \
    -min_data_prob 0.1 \
    -min_shots 30 \
    -fast-simu 1
```