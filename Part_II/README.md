# Part II – Advanced Assignments

This folder contains implementations for all 6 assignments from
**Part II: Perform Any 4 Assignments** of the SPPU BE Computer Laboratory III
(Semester VIII, AI & DS 2019/2020 Pattern).

## Assignments

| # | Folder | Topic | Language |
|---|---|---|---|
| 1 | [Assignment_1_AIS](Assignment_1_AIS/) | Artificial Immune System – Structure Damage Classification | Python |
| 2 | [Assignment_2_DEAP](Assignment_2_DEAP/) | Distributed Evolutionary Algorithms with DEAP | Python |
| 3 | [Assignment_3_Java_RMI_Hotel](Assignment_3_Java_RMI_Hotel/) | Distributed Hotel Booking Application | Java (RMI) |
| 4 | [Assignment_4_Weather_MapReduce](Assignment_4_Weather_MapReduce/) | Weather Analysis – Hottest/Coolest Year via MapReduce | Python |
| 5 | [Assignment_5_ACO_TSP](Assignment_5_ACO_TSP/) | Ant Colony Optimization – Travelling Salesman Problem | Python |
| 6 | [Assignment_6_Neural_Style_Transfer](Assignment_6_Neural_Style_Transfer/) | Neural Style Transfer – Art with Deep Learning | Python |

---

## Quick Start

### Python assignments (1, 2, 4, 5, 6)
```bash
# Install all Python dependencies at once
pip install numpy scikit-learn deap requests matplotlib tensorflow Pillow

# Run each assignment from its directory
cd Assignment_1_AIS && python ais_classifier.py
cd Assignment_2_DEAP && python deap_distributed_ea.py
cd Assignment_4_Weather_MapReduce && python weather_mapreduce.py
cd Assignment_5_ACO_TSP && python aco_tsp.py
cd Assignment_6_Neural_Style_Transfer && python neural_style_transfer.py
```

### Java assignment (3)
```bash
cd Assignment_3_Java_RMI_Hotel
javac *.java
# Terminal 1 – server
java HotelBookingServer
# Terminal 2 – client
java HotelBookingClient
```

---

## Dependency Summary

| Library | Used by | Install |
|---|---|---|
| `numpy` | 1, 4, 5, 6 | `pip install numpy` |
| `scikit-learn` | 1 | `pip install scikit-learn` |
| `deap` | 2 | `pip install deap` |
| `requests` | 4 (optional live data) | `pip install requests` |
| `matplotlib` | 5, 6 | `pip install matplotlib` |
| `tensorflow` | 6 | `pip install tensorflow` |
| `Pillow` | 6 | `pip install Pillow` |
| Java JDK 8+ | 3 | system package manager |

See individual assignment READMEs for detailed instructions and expected output.
