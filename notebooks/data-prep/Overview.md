# Overview about the dataset preprocessing process

| Dataset Collection (folder names)      | Status| Notebook   | Comments |
| :------------------------------------- | :---: | :--------- | :------- |
| ATLAS Higgs Boson Challenge            |   x   | [🗎][ATLAS] | Classification dataset; time component arbitrary |
| Community-NAB                          |   ✓   | [🗎][NAB]   |  |
| IOPS AI Challenge                      |   ✓   | [🗎][IOPS]  |  |
| KDD Robot Execution Failures           |   x   |            | Only very short sequences and annotations are per sequence instead of per point! |
| MIT-BIH Arrhythmia DB                  |   ✓   | [🗎][mitdb] | Complex generation of anomaly-windows to label datasets. |
| MIT-BIH Long-Term ECG Database         |   ✓   | [🗎][ltdb]  | See _MIT-BIH Arrhythmia DB_ for preprocessing explanation. |
| MIT-BIH Supraventricular Arrhythmia DB |   ✓   | [🗎][svdb]  | See _MIT-BIH Arrhythmia DB_ for preprocessing explanation. |
| NASA Spacecraft Telemetry Data         |   ✓   | [🗎][NASA]  | SMAP and MSL datasets |
| Series2Graph                           |  tbd  |            | **No labels ATM!** |
| Server Machine Dataset                 |   ✓   | [🗎][SMD]   |  |
| TSBitmap                               |   x   |            | **No labels!** |
| UCI ML Repository / 3W                 |   x   | [🗎][3W]    | Hard to transform into TS AD task. |
| UCI ML Repository / CalIt2             |   ✓   | [🗎][CalIt2]|  |
| UCI ML Repository / Condition monitoring|      |            |  |
| UCI ML Repository / Daphnet            |       |            |  |
| UCI ML Repository / Dodgers            |       |            |  |
| UCI ML Repository / HEPMASS            |       |            |  |
| UCI ML Repository / Kitsune Network Attack|    |            |  |
| UCI ML Repository / Metro              |       |            |  |
| UCI ML Repository / OPPORTUNITY        |       |            |  |
| UCI ML Repository / Occupancy Detection|       |            |  |
| UCI ML Repository / URLReputation      |       |            |  |
| Webscope-S5                            |   ✓   | [🗎][Yahoo] |  |
| credit-card-fraud                      |   x   |            | Timestamps are not equi-distant. |
| genesis-demonstrator                   |   ✓   | [🗎][gen]   | A single dataset |

## TODO

Check against datasets in [John's benchmark framework](https://github.com/johnpaparrizos/AnomalyDetection/tree/master/benchmark/dataset):

- ECG (source are mitdb, ltdb, and svdb, label source unknown)
- GHL (what is this?)
- NAB ✓
- SMAP ✓
- SMD ✓
- SSA (tbd)
- YAHOO ✓

[gen]: ./Genesis%20Demonstrator.ipynb
[mitdb]: ./MIT-BIH%20Arrhythmia%20Database.ipynb
[ltdb]: ./MIT-BIH%20Long-Term%20ECG%20Database.ipynb
[svdb]: ./MIT-BIH%20Supraventricular%20Arrhythmia%20DB.ipynb
[NAB]: ./NAB.ipynb
[NASA]: ./NASA%20Spacecraft%20Telemtry.ipynb
[SMD]: ./Server%20Machine%20Dataset.ipynb
[Yahoo]: ./YahooWebscopeS5.ipynb
[IOPS]: ./IOPS%20AI%20Challenge.ipynb
[ATLAS]: ./ATLAS%20Higgs%20Boson%20Challenge.ipynb
[3W]: ./UCI-3W.ipynb
[CalIt2]: ./UCI-CalI2.ipynb