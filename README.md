# SCOPE
Self-Correcting Online Log Parsing with Iterable Pattern Extraction

Development is ongoing

run example:
python examples/scopeTest.py

# Run 16 dataset evaluation:
1. change SCOPE/examples/tools.ini to set different configuration
2.  python evaluator.py

# Result

| Dataset           | Group Accuracy | Duration (sec) | Line Number | Rate (lines/sec) | Clusters |
|-------------------|----------------|----------------|-------------|------------------|----------|
| HPC               | 0.8875         | 0.54           | 2000        | 3726.5           | 48       |
| OpenStack         | 0.9925         | 1.16           | 2000        | 1725.7           | 44       |
| BGL               | 0.9855         | 0.52           | 2000        | 3883             | 116      |
| HDFS              | 1.0000         | 0.36           | 2000        | 5606.6           | 14       |
| Hadoop            | 0.9925         | 0.64           | 2000        | 3121.8           | 114      |
| Spark             | 0.9220         | 0.43           | 2000        | 4646.5           | 30       |
| Zookeeper         | 0.9910         | 0.33           | 2000        | 6017.5           | 52       |
| Thunderbird       | 0.9575         | 1.05           | 2000        | 1906.5           | 197      |
| Windows           | 0.9960         | 0.48           | 2000        | 4167.8           | 56       |
| Linux             | 0.9385         | 0.56           | 2000        | 3580.8           | 118      |
| Andriod           | 0.8875         | 1.67           | 2000        | 1199.8           | 167      |
| HealthApp         | 0.9025         | 0.47           | 2000        | 4265.9           | 73       |
| Apache            | 1.0000         | 0.29           | 2000        | 6933.3           | 6        |
| Proxifier         | 0.9900         | 0.43           | 2000        | 4698.2           | 10       |
| OpenSSH           | 0.9985         | 2.16           | 2000        | 924.7            | 26       |
| Mac               | 0.9415         | 1.26           | 2000        | 1583.3           | 334      |
| Thunderbird_expand| 0.9985         | 0.47           | 1941        | 4120.6           | 34       |

