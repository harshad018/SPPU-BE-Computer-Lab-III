# SPPU BE Computer Laboratory III (Semester VIII)

This repository contains the practical implementations for **Computer Laboratory III**, a practical course for the BE Artificial Intelligence and Data Science (2019/2020 Pattern) at Savitribai Phule Pune University (SPPU).

The laboratory covers practical implementations for:
1. **Computational Intelligence**
2. **Distributed Computing**
3. **Deep Learning (Elective)**

---

## Part I: Perform Any 6 Assignments

### 1. RPC Calculation — `Distributed_Computing/1_RPC_Factorial.py`
Design a distributed application where a client sends an integer to a server, which calculates and returns the factorial using Python's XML-RPC (Remote Procedure Call).

```bash
# Terminal 1 – start server
python Distributed_Computing/1_RPC_Factorial.py server

# Terminal 2 – run client (factorial of 7)
python Distributed_Computing/1_RPC_Factorial.py client 7

# Demo mode (server + client in one process)
python Distributed_Computing/1_RPC_Factorial.py
```

---

### 2. RMI Concatenation — `Distributed_Computing/2_RMI_Concatenation.py`
Create a distributed application using Python XML-RPC to simulate Java RMI: a client submits two strings to a remote object, which concatenates and returns them.

```bash
# Terminal 1 – start server
python Distributed_Computing/2_RMI_Concatenation.py server

# Terminal 2 – concatenate two strings
python Distributed_Computing/2_RMI_Concatenation.py client "Hello" "World"

# Demo mode
python Distributed_Computing/2_RMI_Concatenation.py
```

---

### 3. Hadoop MapReduce — `Distributed_Computing/3_Hadoop_MapReduce/`
Implement MapReduce for counting word occurrences and character occurrences in text files.

**Files:**
| File | Description |
|---|---|
| `mapreduce_standalone.py` | Standalone Python simulation (no Hadoop required) |
| `word_mapper.py` | Hadoop Streaming compatible word-count mapper |
| `word_reducer.py` | Hadoop Streaming compatible word-count reducer |
| `char_mapper.py` | Hadoop Streaming compatible character-count mapper |
| `char_reducer.py` | Hadoop Streaming compatible character-count reducer |
| `sample_input.txt` | Sample text file for testing |

```bash
# Standalone simulation (no Hadoop needed)
python Distributed_Computing/3_Hadoop_MapReduce/mapreduce_standalone.py

# Pipe-based simulation (Linux/macOS)
cat Distributed_Computing/3_Hadoop_MapReduce/sample_input.txt \
    | python Distributed_Computing/3_Hadoop_MapReduce/word_mapper.py \
    | sort \
    | python Distributed_Computing/3_Hadoop_MapReduce/word_reducer.py
```

---

### 4. Fuzzy Set Operations — `Computational_Intelligence/4_Fuzzy_Set_Operations.py`
Implement union, intersection, complement, difference, and max-min composition for fuzzy sets and fuzzy relations.

```bash
python Computational_Intelligence/4_Fuzzy_Set_Operations.py
```

**Operations covered:**
- Union: `max(A, B)`
- Intersection: `min(A, B)`
- Complement: `1 − A`
- Difference: `min(A, 1 − B)`
- Max-Min Composition: `T[i,j] = max_k(min(R[i,k], S[k,j]))`

---

### 5. Load Balancing Simulation — `Distributed_Computing/6_Load_Balancing_Simulation.py`
Simulate client requests distributed across servers using three strategies: **Round-Robin**, **Least Connections**, and **Random**. Reports imbalance factor and per-server metrics.

```bash
python Distributed_Computing/6_Load_Balancing_Simulation.py
```

---

### 6. Genetic Algorithm Optimisation — `Computational_Intelligence/5_GA_NN_Optimization.py`
Optimise hyperparameters of a neural network using a Genetic Algorithm for a spray-drying coconut milk quality prediction task. Uses a synthetic dataset and scikit-learn's `MLPRegressor`.

```bash
python Computational_Intelligence/5_GA_NN_Optimization.py
```

---

### 7. Clonal Selection Algorithm — `Computational_Intelligence/6_Clonal_Selection.py`
Implement the Clonal Selection Algorithm (CLONALG) from Artificial Immune Systems theory to minimise the 2-D Rastrigin function. Saves a convergence plot.

```bash
python Computational_Intelligence/6_Clonal_Selection.py
```

---

### 8. Neural Style Transfer — `Computational_Intelligence/7_Neural_Style_Transfer.py`
Create art by applying neural style transfer to images using a pre-trained VGG19 deep network (TensorFlow/Keras). Synthetic sample images are generated automatically if none are provided.

```bash
# Demo with auto-generated images
python Computational_Intelligence/7_Neural_Style_Transfer.py

# Use your own images
python Computational_Intelligence/7_Neural_Style_Transfer.py \
    --content content.jpg --style style.jpg --output result.png --iterations 300
```

> **Note:** First run downloads VGG19 weights (~550 MB). Internet access required for initial download.

---

## Additional Practicals (Existing)

### Section A: Computational Intelligence
| # | File | Description |
|---|---|---|
| A1 | `Computational_Intelligence/1_ANN_Classification.py` | ANN (MLP) for Iris dataset classification |
| A2 | `Computational_Intelligence/2_Fuzzy_Logic.py` | Fuzzy Logic controller (tipping problem) |
| A3 | `Computational_Intelligence/3_Genetic_Algorithm.py` | GA to maximise f(x) = x² |

### Section B: Distributed Computing
| # | File | Description |
|---|---|---|
| B1 | `Distributed_Computing/4_RMI_Calculator.py` | RMI calculator using Pyro4 |
| B2 | `Distributed_Computing/5_Clock_Synchronization.py` | Berkeley clock synchronisation algorithm |

---

## Prerequisites
- Python 3.8+
- See `requirements.txt` for the full list of dependencies.

## Installation
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy scikit-learn scikit-fuzzy matplotlib tensorflow Pillow Pyro4
```

## Repository Structure
```
SPPU-BE-Computer-Lab-III/
├── Computational_Intelligence/
│   ├── 1_ANN_Classification.py
│   ├── 2_Fuzzy_Logic.py
│   ├── 3_Genetic_Algorithm.py
│   ├── 4_Fuzzy_Set_Operations.py    ← NEW (Practical 4)
│   ├── 5_GA_NN_Optimization.py      ← NEW (Practical 6)
│   ├── 6_Clonal_Selection.py        ← NEW (Practical 7)
│   └── 7_Neural_Style_Transfer.py   ← NEW (Practical 8)
├── Distributed_Computing/
│   ├── 1_RPC_Factorial.py           ← NEW (Practical 1)
│   ├── 2_RMI_Concatenation.py       ← NEW (Practical 2)
│   ├── 3_Hadoop_MapReduce/          ← NEW (Practical 3)
│   │   ├── mapreduce_standalone.py
│   │   ├── word_mapper.py
│   │   ├── word_reducer.py
│   │   ├── char_mapper.py
│   │   ├── char_reducer.py
│   │   └── sample_input.txt
│   ├── 4_RMI_Calculator.py
│   ├── 5_Clock_Synchronization.py
│   └── 6_Load_Balancing_Simulation.py ← NEW (Practical 5)
├── requirements.txt                 ← NEW
└── README.md
```

*Disclaimer: These problem statements are based on the standard SPPU syllabus for Computer Laboratory III (BE AI & DS, 2019/2020 Pattern).*
