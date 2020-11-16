# HW simulator of SA
**HW (hardware)** simulator of SA (Simulated Annealing)

## What's this?
* **SA-based annealer**; Ising processor (or QUBO solver) that can address a "fully-connected" model
* **HW simulator** which imitates the behavior of HW annealer (*now under development*)
* Including **HW-specific (or HW-friendly) operations** (fixed-point operations, approximated sigmoid operation, and pseudo random number generator)

## Requirements
* python3 (numpy, argparse, tqdm)

## Simulator types
|No.|Name|Input|Algorithm|
|:----|:---------|:-----|:-----|
|(1)|`01_SA_simulator_Ising.py`|Ising model|Typical SA|
|(2)|`02_SA_simulator_QUBO.py`|QUBO|Typical SA|
|(3)|`03_ConstrainedSA_simulator_QUBO.py`|QUBO|Transition-constrained SA|

* The simulator (3), transion-constrained SA, works in such a way as to always satisfy 1-dimensional one-hot constraints.
* On the other hand, when switching to check mode by an option `-c`, it works in the same manner as the simulator (2).

## How to use
#### (a) Read in a pre-defined model from a file
```
python3 01_SA_simulator_Ising.py -i FILE(.dat) [-O 600] [-I 4000] [-S 20.0] [-E 0.5] [-s 12345] [-d] [-v]
python3 02_SA_simulator_QUBO.py -i FILE(.dat) [-O 100] [-I 1000] [-S 100.0] [-E 0.1] [-s 12345] [-d] [-v]
python3 03_ConstrainedSA_simulator_QUBO.py -i FILE(.dat) -C FILE(.con) [-O 100] [-I 1000] [-S 100.0] [-E 0.1] [-s 12345] [-d] [-v]
```

#### (b) Generate a random model
```
python3 01_SA_simulator_Ising.py [-n 256] [-O 600] [-I 4000] [-S 20.0] [-E 0.5] [-s 12345] [-d] [-v]
python3 02_SA_simulator_QUBO.py [-n 256] [-O 100] [-I 1000] [-S 100.0] [-E 0.1] [-s 12345] [-d] [-v]
```

#### Model information
|Option|Description|Default|Support|
|:----:|:---------|:-----:|:-----:|
|`-i`|Model file (Only for (a))|None|ALL|
|`-C`|One-hot constraint file|None|(3)|
|`-n`|#. of spins (Only for (b))|256|(1), (2)|

* The simulator (3) always require a pair of model file (.dat) and one-hot constraint file (.con).

#### Annealing parameters
|Option|Description|Default (Ising model)|Default (QUBO)|
|:----:|:---------|:-----:|:-----:|
|`-O`|#. of outer loops|600|100|
|`-I`|#. of inner loops|4000|1000|
|`-S`|Initial temperature|20.0|100.0|
|`-E`|Final temperature|0.5|0.1|
|`-s`|Seed value|12345|12345|

#### Other options
|Option|Description|Support|
|:----:|:---------|:-----:|
|`-d`|Output a log file (`./energy.log`) recording energy transition|ALL|
|`-v`|Output a log file (`./var.log`) recording final state|ALL|
|`-c`|Switch to check mode which works as typical SA but counts the number of one-hot constraint violations|(3)|

## Input file format

### Model file
* The 1st line shows #. of spins.
* **J** and **h** are specified from the 2nd line (32bit integer).
* Constant value is specified in the last line.

#### Example
```
4
0 0 -1
0 1 2
0 2 1
1 1 -1
1 3 1
2 2 -1
2 3 2
3 3 -1
2
```

### One-hot constraint file 
* The 1st line shows #. of one-hot constraint sets.
* Pairs of start and end indices are specified from the 2nd line.

#### Example
```
2
0 1
2 3
```

## Output example

#### Simulator (1)
```
Model file: Ising_model/G1_Ising.dat
----------------------------------------
-- Model & Parameters ------------------
----------------------------------------
 N = 800
 #Loops  = 600 x 4000
 T(set)  = 20.0 --> 0.5
 T(real) = 20.0 --> 0.500003628459082
 seed = 12345
----------------------------------------
100%|██████████████████████████████████████████████████████████████████████████████| 600/600 [00:30<00:00, 19.58it/s]
----------------------------------------
-- Result ------------------------------
----------------------------------------
 H(init) = 118
 H(fin)  = -4072
----------------------------------------
```

#### Simulator (2)
```
Model file: QUBO_problem/2-FullIns_3_QUBO_5.dat
----------------------------------------
-- Model & Parameters ------------------
----------------------------------------
 N = 260
 #Loops  = 100 x 1000
 T(set)  = 100.0 --> 0.1
 T(real) = 100.0 --> 0.10000066578831485
 seed = 12345
----------------------------------------
100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 80.17it/s]
----------------------------------------
-- Result ------------------------------
----------------------------------------
 H(init) = 662
 H(fin)  = 6
----------------------------------------
```

#### Simulator (3)
```
Model file: QUBO_problem/2-FullIns_3_QUBO_5.dat
One-hot table (size: 52)
[(0, 4), (5, 9), ... , (250, 254), (255, 259)]
----------------------------------------
-- Model & Parameters ------------------
----------------------------------------
 N = 260
 #Loops  = 100 x 1000
 T(set)  = 100.0 --> 0.1
 T(real) = 100.0 --> 0.10000066578831485
 seed = 12345
----------------------------------------
100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:02<00:00, 39.72it/s]
----------------------------------------
-- Result ------------------------------
----------------------------------------
 H(init) = 201
 H(fin)  = 0
 #. One-hot constraint violations = 0
----------------------------------------
```

#### Simulator (3): check mode
```
Model file: QUBO_problem/2-FullIns_3_QUBO_5.dat
One-hot table (size: 52)
[(0, 4), (5, 9), ... , (250, 254), (255, 259)]
----------------------------------------
-- Model & Parameters ------------------
----------------------------------------
 N = 260
 #Loops  = 100 x 1000
 T(set)  = 100.0 --> 0.1
 T(real) = 100.0 --> 0.10000066578831485
 seed = 12345
----------------------------------------
100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 79.07it/s]
----------------------------------------
-- Result ------------------------------
----------------------------------------
 H(init) = 662
 H(fin)  = 6
 #. One-hot constraint violations = 0
----------------------------------------
```

## Acknowledgement
This work is supported by the MITOU Target program from Information-technology Promotion Agency, Japan (IPA).
