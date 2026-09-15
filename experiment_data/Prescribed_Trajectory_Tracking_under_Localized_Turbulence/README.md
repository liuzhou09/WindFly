# Prescribed Trajectory Tracking under Localized Turbulence

The `raw/` directory contains 48 flight-log CSV files for four controllers at reference speeds of 1, 2, and 3 m/s and wind settings of 0, 5, 10, and 15 m/s.

Files are organized as `raw/speed_<speed>ms/wind_<wind>ms/`. Each filename contains a controller label (`Baseline`, `DiffPhys`, `INDI`, or `Ours`) and the recording timestamp.

Each CSV contains timestamps in seconds; position in meters; velocity in meters per second; quaternion orientation in xyzw order; and acceleration, disturbance, and model-output fields in meters per second squared. `SHA256SUMS.txt` provides checksums for all files under `raw/`.
