# HW simulator of SA
**HW (hardware)** simulator of SA (Simulated Annealing)

## What's this?
* **SA-based annealer**; ground-state-search processor of a fully-connected Ising model
* **HW simulator** which imitates the behavior of HW annealer (*now under development*)
* Including **HW-specific (or HW-friendly) operations** (fixed-point operations, approximated sigmoid operation, and pseudo random number generator)

## Requirements
* python3 (numpy, argparse, tqdm)

## How to use
(a) Read in an Ising model from a file
```
python3 simulator.py -i FILE [-O 600] [-I 4000] [-S 20.0] [-E 0.5] [-s 123] [-d]
```

(b) Generate a random model
```
python3 simulator.py [-n 256] [-O 600] [-I 4000] [-S 20.0] [-E 0.5] [-s 123] [-d]
```

Model information
|Option|Description|Default|
|:----:|:---------:|:-----:|
|`-i`|Model file (Only for (a))|None|
|`-n`|#. of spins (Only for (b))|256|

Annealing parameters
|Option|Description|Default|
|:----:|:---------:|:-----:|
|`-O`|#. of outer loops|600|
|`-I`|#. of inner loops|4000|
|`-S`|Initial temperature|20.0|
|`-E`|Final temperature|0.5|
|`-s`|Seed value|123|

Other options
|Option|Description|
|:----:|:---------:|
|`-d`|Output a log file (`./energy.log`) recording energy transition|

## Input file format
* The 1st line shows #. of spins.
* **J** and **h** are specified from the 2nd line.
* Constant value is not supported.

Example
```
3
1 1 -1
1 2 -3
1 3 -2
2 2 -1
2 3 -1
3 3 -1
```

<img width="441" alt="SampleModel" src="https://user-images.githubusercontent.com/71317410/93204943-eb67fa00-f791-11ea-979d-4eff8a8f2568.png">

## Output
Example
```
Model file: G1_Ising.dat
----------------------------------------
-- Model & Parameters ------------------
----------------------------------------
 N = 800
 #Loops  = 600 x 4000
 T(set)  = 20.0 --> 0.5
 T(real) = 20.0 --> 0.500003628459082
 seed = 123
----------------------------------------
100%|██████████████████████████████████████████████| 600/600 [00:43<00:00, 13.72it/s]
----------------------------------------
-- Result ------------------------------
----------------------------------------
 H(init) = 202
 H(fin)  = -4072
----------------------------------------
```

* `G1_Ising.dat` includes an Ising model transformed from the MAX-CUT problem `G1` (obtained [here](http://web.stanford.edu/~yyye/yyye/Gset/)).

## Acknowledgement
The work is supported by the MITOU Target program from Information-technology Promotion Agency, Japan (IPA).
