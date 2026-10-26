# Results

The CSV files here are transcribed from the written test reports, produced on the evening
of each run from the raw logs. The raw logs themselves (1 Hz power log, per-request
timings, daemon snapshots) stayed on the test machine and are not part of this
repository. Treat the CSVs as the published summary, not as raw data.

| file | content |
|---|---|
| `ladder_ttft.csv` | the full ladder at 200/200 W and 250/250 W: requests, prompt sizes, TTFT, peak power and temperature per card |
| `power_cap_200_250_300.csv` | prefill throughput and decode time at 200, 250 and a one-off 300 W |
| `mtp_speculative_decoding.csv` | decode speed with and without MTP speculative decoding (separate run, different model; see README) |

Ranges such as `40.2-45.9` are min-max over the requests of that stage. Cells with `/`
list the requests of the stage in order.
