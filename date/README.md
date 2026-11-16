# EirGrid 2025 input data

Download the 2025 quarter-hourly system data Excel workbook from the official
[EirGrid system and renewable data reports](https://www.eirgrid.ie/grid/system-and-renewable-data-reports)
and place **one** `.xlsx` workbook in this folder.

The pipeline reads `System Data` and the `DateTime`, `GMT Offset`, and `IE Demand`
columns. It converts local timestamps to UTC using the supplied GMT offset,
rejects duplicate timestamps and missing quarter-hours, and records a SHA256
checksum in `rezultate/data_audit.json`.

The source data remain attributed to EirGrid. This repository does not
redistribute the Excel workbook or assign a new license to the original data.
