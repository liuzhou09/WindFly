# Precision Hovering Stabilization against Commercial Baselines

The `raw/` directory contains 12 Qualisys CSV files from six comparison experiments. Air3S, Mavic3E, and Mini5Pro are each evaluated under constant (`const`) and sinusoidal (`sine`) wind, with one comparison-aircraft recording and one `Ours` recording per experiment.

The recordings are sampled at 150 Hz. The first 12 lines contain Qualisys metadata and headers; numeric frames begin on line 13. The timestamp in column 2 is in milliseconds, and global X, Y, and Z positions in columns 10–12 are in millimeters.

`SHA256SUMS.txt` provides checksums for all files under `raw/`.
