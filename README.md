# SPPU BE Computer Laboratory III (Semester VIII)

This repository contains the practical implementations for **Computer Laboratory III**, a practical course for the BE Artificial Intelligence and Data Science (2019/2020 Pattern) at Savitribai Phule Pune University (SPPU).

The laboratory primarily covers practical implementations for subjects like:
1. **Computational Intelligence**
2. **Distributed Computing**
3. **Electives (e.g., Deep Learning)**

---

## Part I: Perform Any 6 Assignments

| # | Folder | Topic |
|---|---|---|
| 1 | `Distributed_Computing/` | RPC Calculation – Factorial |
| 2 | `Distributed_Computing/` | RMI Concatenation – String concat |
| 3 | `Distributed_Computing/` | Hadoop MapReduce – Character & word count |
| 4 | `Computational_Intelligence/` | Fuzzy Set Operations |
| 5 | `Distributed_Computing/` | Load Balancing Simulation |
| 6 | `Computational_Intelligence/` | Genetic Algorithm Optimization |
| 7 | `Computational_Intelligence/` | Clonal Selection Algorithm |
| 8 | `Computational_Intelligence/` | Neural Style Transfer |

### Section A: Computational Intelligence
1. **Artificial Neural Network (ANN):** Implement a simple ANN (Multi-Layer Perceptron) for classification. → `Computational_Intelligence/1_ANN_Classification.py`
2. **Fuzzy Logic:** Implement a Fuzzy Logic controller (e.g., for tipping based on service and food quality). → `Computational_Intelligence/2_Fuzzy_Logic.py`
3. **Genetic Algorithm:** Implement a Genetic Algorithm to solve an optimization problem (e.g., maximizing a function). → `Computational_Intelligence/3_Genetic_Algorithm.py`

### Section B: Distributed Computing
4. **Remote Method Invocation (RMI):** Implement a distributed system using RMI/RPC (e.g., a simple calculator). → `Distributed_Computing/4_RMI_Calculator.py`
5. **Clock Synchronization:** Implement a distributed algorithm for clock synchronization (e.g., Berkeley algorithm). → `Distributed_Computing/5_Clock_Synchronization.py`

### Section C: Deep Learning (Elective)
6. **Convolutional Neural Network (CNN):** Implement a CNN for image classification using a standard dataset (e.g., MNIST).

---

## Part II: Perform Any 4 Assignments

All 6 Part II assignments are implemented in the [`Part_II/`](Part_II/) directory.

| # | Folder | Topic | Language |
|---|---|---|---|
| 1 | [`Part_II/Assignment_1_AIS/`](Part_II/Assignment_1_AIS/) | Artificial Immune System – Structure Damage Classification | Python |
| 2 | [`Part_II/Assignment_2_DEAP/`](Part_II/Assignment_2_DEAP/) | Distributed Evolutionary Algorithms with DEAP | Python |
| 3 | [`Part_II/Assignment_3_Java_RMI_Hotel/`](Part_II/Assignment_3_Java_RMI_Hotel/) | Distributed Hotel Booking Application (Java RMI) | Java |
| 4 | [`Part_II/Assignment_4_Weather_MapReduce/`](Part_II/Assignment_4_Weather_MapReduce/) | Weather Analysis – Hottest/Coolest Year via MapReduce | Python |
| 5 | [`Part_II/Assignment_5_ACO_TSP/`](Part_II/Assignment_5_ACO_TSP/) | Ant Colony Optimization – Travelling Salesman Problem | Python |
| 6 | [`Part_II/Assignment_6_Neural_Style_Transfer/`](Part_II/Assignment_6_Neural_Style_Transfer/) | Neural Style Transfer – Art with Deep Learning | Python |

### Part II Quick Start

```bash
# Install Python dependencies
pip install numpy scikit-learn deap requests matplotlib tensorflow Pillow

# Assignment 1 – AIS Classifier
cd Part_II/Assignment_1_AIS && python ais_classifier.py

# Assignment 2 – DEAP Distributed EA
cd Part_II/Assignment_2_DEAP && python deap_distributed_ea.py --workers 4

# Assignment 3 – Java RMI Hotel Booking
cd Part_II/Assignment_3_Java_RMI_Hotel
javac *.java
java HotelBookingServer   # Terminal 1
java HotelBookingClient   # Terminal 2

# Assignment 4 – Weather MapReduce
cd Part_II/Assignment_4_Weather_MapReduce && python weather_mapreduce.py

# Assignment 5 – ACO TSP
cd Part_II/Assignment_5_ACO_TSP && python aco_tsp.py --plot

# Assignment 6 – Neural Style Transfer
cd Part_II/Assignment_6_Neural_Style_Transfer && python neural_style_transfer.py
```

---

## Prerequisites

- **Python 3.x**
- **Java JDK 8+** (for Assignment 3)
- Python libraries (see below)

## Installation

```bash
# Part I dependencies
pip install numpy scikit-fuzzy pyro4 tensorflow matplotlib

# Part II additional dependencies
pip install scikit-learn deap requests Pillow
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

## Usage
Navigate to the respective directories and run the Python scripts.
Each assignment folder contains its own `README.md` with detailed usage instructions.

*Disclaimer: These problem statements are generalized based on the standard SPPU syllabus for Computer Laboratory III.*
