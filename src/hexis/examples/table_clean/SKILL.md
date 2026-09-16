---
name: table-clean
description: Clean a CSV table by reading it, checking and repairing its header, and writing the result to a new file.
trigger: When the user asks to tidy up or clean a CSV table file
---

# Table Cleaning

Tidy up a CSV table and write the result to a new file. The whole process revolves around the
header: read the table in, check whether the header is well-formed, and if it is not, repair it
one spot at a time and check again, until it is well-formed or the repair bound is reached;
finally write the result to a new file.

## S1 Read

Use `read_csv` to read the file that `path` points to. Take the first line as the header row
(`header_row`) and the remaining lines as data rows (`rows`).

## S2 Header check

Decide whether the current `header_row` is well-formed.

### S2.1 Well-formedness criterion

A header is well-formed if and only if its fields are separated by commas, every field name is
non-empty, and no field contains the placeholder mark `Unnamed`. A header with an empty field or
with `Unnamed` is malformed.

## S3 Repair

If the header is malformed, repair it with `fix_header`, then **go back to S2 and check again**.
Each repair handles only one spot, so a header with many bad fields needs several rounds. The
number of repairs (`fix_count`) must not exceed 3; if the header is still malformed when the
bound is reached, stop repairing.

## S4 Export

Once the header is well-formed, use `export` to write the result to `output_path`.

## P1 Prohibition

**Never overwrite the source file.** The target path of `export`, `output_path`, must not equal
the source file path `path`. Even if the result is correct, a run whose export target is the
source file is judged as rejected.
